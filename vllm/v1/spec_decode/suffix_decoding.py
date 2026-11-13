# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import time
from collections import OrderedDict
from collections.abc import Hashable

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_input_batch import InputBatch

# Seconds a finished request's local tree stays available to a successor.
FINISHED_TREE_TTL_S = 600.0


class SuffixDecodingProposer:
    """
    Speculative decoding proposer for Suffix Decoding (https://arxiv.org/pdf/2411.04975).
    This class imports and uses the official implementation from Arctic Inference
    (https://github.com/snowflakedb/ArcticInference).
    """

    def __init__(self, vllm_config: VllmConfig):
        config = vllm_config.speculative_config
        assert config is not None, "Speculative config must be set"
        self.num_speculative_tokens = config.num_speculative_tokens
        self.max_tree_depth = config.suffix_decoding_max_tree_depth
        self.max_spec_factor = config.suffix_decoding_max_spec_factor
        self.min_token_prob = config.suffix_decoding_min_token_prob
        self.max_model_len = vllm_config.model_config.max_model_len

        # Lazy import to avoid error when Suffix Decoding is not used.
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        # Initialize and empty cache. This object will take care of caching request
        # outputs, evicting old requests, and manages the per-prompt suffix trees.
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=config.suffix_decoding_max_tree_depth,
            max_cached_requests=config.suffix_decoding_max_cached_requests,
        )
        # The match score of each draft `propose` last returned, in its order.
        self.last_scores: list[float] = []
        # A finished request's local tree is kept for a successor whose prompt
        # starts with every token the tree indexed. Only with the global tree
        # off, since a kept key would keep extending its global response.
        self.max_finished_trees = (
            vllm_config.scheduler_config.max_num_seqs
            if config.suffix_decoding_max_cached_requests == 0
            else 0
        )
        # The key of each running request's tree in `suffix_cache`.
        self._tree_keys: dict[str, Hashable] = {}
        # The prompt and outputs each running tree indexed, by key.
        self._indexed: dict[Hashable, tuple[np.ndarray, list[int]]] = {}
        # The tokens each finished tree indexed and when it stopped, oldest first.
        self._finished: OrderedDict[Hashable, tuple[np.ndarray, float]] = OrderedDict()
        self._next_key = itertools.count()

    def propose(
        self,
        num_speculative_tokens: int,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> list[list[int]]:
        assert num_speculative_tokens == self.num_speculative_tokens
        """
        Propose speculative tokens for each request in the input batch. Suffix Decoding
        will speculate a dynamic number of tokens for each request every decoding step,
        so each entry in the returned list may have different lengths.
        """
        # Stop requests that are not in the input batch first, so a successor
        # starting in this batch can take its predecessor's tree.
        for req_id in self._tree_keys.keys() - input_batch.req_id_to_index.keys():
            self._stop_request(req_id)
        deadline = time.monotonic() - FINISHED_TREE_TTL_S
        while self._finished and next(iter(self._finished.values()))[1] < deadline:
            self.suffix_cache.stop_request(self._finished.popitem(last=False)[0])
        draft_token_ids: list[list[int]] = []
        scores: list[float] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                # Skip speculative decoding for partial prefills.
                draft_token_ids.append([])
                scores.append(0.0)
                continue

            req_id = input_batch.req_ids[i]
            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                draft_token_ids.append([])
                scores.append(0.0)
                continue

            index = input_batch.req_id_to_index[req_id]
            key = self._tree_keys.get(req_id)
            if key is None:
                num_prompt_tokens = input_batch.num_prompt_tokens[index]
                prompt_token_ids = input_batch.token_ids_cpu[index, :num_prompt_tokens]
                key = self._start_request(req_id, prompt_token_ids)

            # Append the newly sampled ids to the suffix cache for this request.
            self.suffix_cache.add_active_response(key, sampled_ids)
            if self.max_finished_trees:
                self._indexed[key][1].extend(sampled_ids)

            # Suffix decoding only uses the most recent tokens up to max_tree_depth, so
            # we extract the pattern from the end of the input.
            start = max(0, num_tokens - self.max_tree_depth)
            pattern = input_batch.token_ids_cpu[i, start:num_tokens]
            draft = self.suffix_cache.speculate(
                key,
                pattern,
                max_spec_tokens=min(
                    self.num_speculative_tokens, self.max_model_len - num_tokens - 1
                ),
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

            draft_token_ids.append(draft.token_ids)
            scores.append(float(draft.score))

        self.last_scores = scores
        return draft_token_ids

    def _start_request(self, req_id: str, prompt_token_ids: np.ndarray) -> Hashable:
        """Start a request's local tree, extending a finished one it continues.

        Args:
            req_id: The request.
            prompt_token_ids: Its prompt.

        Returns:
            The key of the request's tree in `suffix_cache`.
        """
        key = self._take_finished_tree(prompt_token_ids)
        if key is None:
            if self.max_finished_trees:
                key = next(self._next_key)
            else:
                key = req_id
                if req_id in self.suffix_cache.cached_requests:
                    # Reset the suffix cache for this request.
                    self.suffix_cache.evict_cached_response(req_id)
            # Start a new request, this will build the suffix tree for that prompt.
            self.suffix_cache.start_request(key, prompt_token_ids)
        if self.max_finished_trees:
            self._indexed[key] = (prompt_token_ids.copy(), [])
        self._tree_keys[req_id] = key
        return key

    def _take_finished_tree(self, prompt_token_ids: np.ndarray) -> Hashable | None:
        """Extend the finished tree that indexed the longest prefix of a prompt.

        Args:
            prompt_token_ids: The prompt.

        Returns:
            The tree's key, now indexing the whole prompt, or None if no
            finished tree indexed a prefix of it.
        """
        num_tokens = len(prompt_token_ids)
        best_key: Hashable | None = None
        best_len = 0
        for key, (tokens, _) in self._finished.items():
            n = len(tokens)
            if (
                best_len < n <= num_tokens
                and prompt_token_ids[n - 1] == tokens[-1]
                and np.array_equal(prompt_token_ids[:n], tokens)
            ):
                best_key, best_len = key, n
        if best_key is None:
            return None
        del self._finished[best_key]
        if best_len < num_tokens:
            self.suffix_cache.add_active_response(best_key, prompt_token_ids[best_len:])
        return best_key

    def _stop_request(self, req_id: str) -> None:
        key = self._tree_keys.pop(req_id)
        if not self.max_finished_trees:
            self.suffix_cache.stop_request(key)
            return
        prompt, outputs = self._indexed.pop(key)
        tokens = np.concatenate((prompt, np.array(outputs, dtype=prompt.dtype)))
        self._finished[key] = (tokens, time.monotonic())
        if len(self._finished) > self.max_finished_trees:
            self.suffix_cache.stop_request(self._finished.popitem(last=False)[0])

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass

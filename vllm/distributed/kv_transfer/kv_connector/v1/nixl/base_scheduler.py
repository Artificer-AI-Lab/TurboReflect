# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base scheduler-side logic for the NIXL connector."""

import threading
import time
from typing import TYPE_CHECKING, Any

import msgspec
import zmq

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineId,
    yield_req_data,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    HeartbeatInfo,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


def select_mamba_pull_state(
    states: list[list[Any]],
    remote_num_tokens: int,
    num_prompt_tokens: int,
    num_computed_tokens: int,
    block_size: int,
) -> list[Any] | None:
    """The recurrent state a hybrid P loads from a D's held history, if any.

    ``states`` is D's ``remote_mamba_states``: ``[num_tokens, [block id per
    Mamba group]]`` for each position whose exact state D holds. P loads the
    latest one within the offer (``remote_num_tokens``, which a reader may cut)
    that leaves at least its last prompt token to compute and lies past its
    own block-aligned hit; the attention blocks are cut to that position.

    Returns:
        That ``[num_tokens, block_ids]`` entry, or None for no pull.
    """
    if num_computed_tokens % block_size:
        return None
    limit = min(remote_num_tokens, num_prompt_tokens - 1)
    best = None
    for entry in states:
        if num_computed_tokens < entry[0] <= limit and (
            best is None or entry[0] > best[0]
        ):
            best = entry
    return best


class NixlBaseConnectorScheduler:
    """Base implementation of Scheduler side methods shared by pull and push."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self.kv_cache_config = kv_cache_config
        self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        self.side_channel_port = (
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT
            + vllm_config.parallel_config.data_parallel_index
        )
        assert vllm_config.kv_transfer_config is not None
        self._kv_lease_duration: int = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_lease_duration", 30
            )
        )
        # NOTE (NickLucche): For now we use a hardcoded value for a simpler interface.
        self._heartbeat_interval = self._kv_lease_duration // 6
        if current_platform.device_type == "cpu":
            self.use_host_buffer = False
        else:
            self.use_host_buffer = (
                vllm_config.kv_transfer_config.kv_buffer_device == "cpu"
            )
        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            # Also handle unlikely SW-only model case instead of checking num_groups>1.
            and any(
                not isinstance(g.kv_cache_spec, FullAttentionSpec)
                for g in kv_cache_config.kv_cache_groups
            )
        )
        self._has_mamba = any(
            isinstance(g.kv_cache_spec, MambaSpec)
            for g in kv_cache_config.kv_cache_groups
        )

        logger.info("Initializing NIXL Scheduler %s", engine_id)
        if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
            logger.info("Hybrid Memory Allocator is enabled with NIXL")

        # Background thread for handling new handshake requests.
        self._nixl_handshake_listener_t: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, BlockIds]] = {}
        self._reqs_need_save: dict[ReqId, Request] = {}
        # Reqs to send and their expiration time
        self._reqs_need_send: dict[ReqId, float] = {}
        self._reqs_in_batch: set[ReqId] = set()
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[ReqId] = set()

        # Heartbeat tracking: requests needing periodic lease-renewal heartbeats to
        # remote P-side, stored as ready-to-send HeartbeatInfo grouped by remote engine
        self._heartbeat_by_engine: dict[EngineId, HeartbeatInfo] = {}
        # Reverse lookup: local req_id -> (engine_id, remote_req_id) for O(1) removal
        self._heartbeat_req_engine: dict[ReqId, tuple[EngineId, ReqId]] = {}
        self._last_heartbeat_time: float = 0.0

        # Gather Sliding Window sizes for each kv cache group (if any) in number of
        # blocks per KV cache group. This is used to clip the local attention window.
        sw_sizes_tokens: list[tuple[int, int]] = [
            (g.kv_cache_spec.sliding_window, g.kv_cache_spec.block_size)
            if isinstance(g.kv_cache_spec, SlidingWindowSpec)
            else (0, self.block_size)
            for g in kv_cache_config.kv_cache_groups
        ]
        # cdiv(n_tokens, block_size) gives blocks/window; add 1 to conservatively
        # account for boundary overlap eg window isn't fully aligned with blocks.
        self.blocks_per_sw = [
            cdiv(n_tokens, block_size) + 1 if n_tokens else 0
            for n_tokens, block_size in sw_sizes_tokens
        ]

        # Trailing scratch slots that mamba managers co-allocate per request
        # for speculative decoding; None for non-SSM groups.
        self._ssm_spec_blocks = [
            g.kv_cache_spec.num_speculative_blocks
            if isinstance(g.kv_cache_spec, MambaSpec)
            else None
            for g in kv_cache_config.kv_cache_groups
        ]
        # Only "all" mode keeps a state per block position; the other modes
        # keep a single running state in the last non-speculative slot.
        self._ssm_state_slots_are_positional = (
            vllm_config.cache_config.mamba_cache_mode == "all"
        )
        # A D's held history can be pulled into a hybrid P at a block boundary
        # whose exact state D caches (align mode) or at its running state. Only
        # full-attention groups beside the Mamba ones, one block size, and
        # synchronous scheduling, under which every cached block is computed.
        groups = kv_cache_config.kv_cache_groups
        self._mamba_group_ids = [
            i for i, g in enumerate(groups) if isinstance(g.kv_cache_spec, MambaSpec)
        ]
        self._mamba_prefix_pull = bool(self._mamba_group_ids) and (
            not self._ssm_state_slots_are_positional
            and not vllm_config.scheduler_config.async_scheduling
            and all(
                isinstance(g.kv_cache_spec, MambaSpec)
                or (
                    type(g.kv_cache_spec) is FullAttentionSpec
                    and g.kv_cache_spec.sliding_window is None
                )
                for g in groups
            )
            and {g.kv_cache_spec.block_size for g in groups} == {self.block_size}
        )
        # Bound by bind_gpu_block_pool: the cached boundary states a hybrid D
        # exports are pinned in it until the hold on their request ends.
        self._block_pool: BlockPool | None = None
        self._mamba_state_pins: dict[ReqId, list[KVCacheBlock]] = {}

        # Threshold to decide whether to compute kv cache locally
        # or pull from a remote node: minimum number of remote
        # tokens to amortize the xfer latencies
        self.kv_recompute_threshold: int = int(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_recompute_threshold", 64
            )
        )

        # Bi-directional KV transfer feature supports KV block
        # transfers from D node to P node
        self.is_bidirectional_kv_xfer_enabled = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "bidirectional_kv_xfer", False
            )
        )
        self.decoder_kv_blocks_ttl = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "decoder_kv_blocks_ttl", 480
            )
        )

        if self.is_bidirectional_kv_xfer_enabled and self.kv_recompute_threshold > 0:
            logger.info(
                "Bidirectional KV transfer is enabled and the kv "
                "recompute threshold is set to %d tokens."
                "KV blocks on D are released after a TTL of %d seconds.",
                self.kv_recompute_threshold,
                self.decoder_kv_blocks_ttl,
            )

    def shutdown(self):
        self._stop_event.set()
        if self._nixl_handshake_listener_t is not None:
            self._nixl_handshake_listener_t.join()
            self._nixl_handshake_listener_t = None

    def on_new_request(self, request: "Request") -> None:
        """Track a request that may need heartbeats."""
        params = request.kv_transfer_params
        if (
            params is not None
            and params.get("do_remote_decode")
            and not params.get("decode_after_pull")
            and self._has_mamba
        ):
            self._truncate_mamba_request_for_prefill(request)

        # NOTE (NickLucche) This excludes request meant for P, ie heartbeats are
        # effectively disabled for Bidirectional KV transfer.
        if params is None or not params.get("do_remote_prefill"):
            return
        # Only track if all required remote fields are present.
        remote_engine_id = params.get("remote_engine_id")
        remote_request_id = params.get("remote_request_id")
        host = params.get("remote_host")
        port = params.get("remote_port")
        tp_size = params.get("tp_size")
        pp_size = params.get("pp_size", 1)
        if (
            remote_engine_id is None
            or remote_request_id is None
            or host is None
            or port is None
            or tp_size is None
        ):
            return
        if remote_engine_id not in self._heartbeat_by_engine:
            self._heartbeat_by_engine[remote_engine_id] = HeartbeatInfo(
                req_ids=set(),
                host=host,
                port=port,
                tp_size=tp_size,
                pp_size=pp_size,
            )
        self._heartbeat_by_engine[remote_engine_id].req_ids.add(remote_request_id)
        self._heartbeat_req_engine[request.request_id] = (
            remote_engine_id,
            remote_request_id,
        )

    def _stop_heartbeat(self, req_id: ReqId) -> None:
        """Remove *req_id* from heartbeat tracking (if tracked)."""
        if key := self._heartbeat_req_engine.pop(req_id, None):
            engine_id, remote_id = key
            if info := self._heartbeat_by_engine.get(engine_id):
                info.req_ids.discard(remote_id)
                if not info.req_ids:
                    # Clean up empty engines so we don't leak a key when remote dies.
                    del self._heartbeat_by_engine[engine_id]

    def get_exchange_clipped_blocks(
        self, block_ids: BlockIds, clip_ssm: bool = True
    ) -> BlockIds:
        """Clip a request's block lists down to the transferable blocks.

        Sliding-window groups keep only the in-window tail: the KV cache
        manager allocates blocks for the entire sequence length and cleans up
        out-of-window blocks only prior to the `request_finished_all_groups`
        hook.

        SSM groups keep only their state-bearing slots: the trailing
        speculative scratch slots always go, and in single-state cache modes
        so does everything before the running state (null placeholders and
        the previous step's superseded state). "all" mode keeps its remaining
        slots, which the worker pairs position-wise.

        Use this at every block-id exchange point. Pass ``clip_ssm=False``
        for per-step partial lists (host-buffer save), where the SSM strip
        does not apply.
        """
        if len(block_ids) == 0 or not self._is_hma_required:
            # No blocks to clip eg Full prefix cache hit or not a hybrid model.
            return block_ids
        # NOTE (NickLucche) This logic is currently handled at the connector level
        # because offloading connectors might want to receive the whole sequence even
        # for SWA groups. We will abstract this logic once the interface is more stable
        assert len(block_ids) == len(self.blocks_per_sw), (
            "Number of KV cache groups must match"
        )
        clipped = []
        for i, blocks in enumerate(block_ids):
            if n_sw := self.blocks_per_sw[i]:
                blocks = blocks[-n_sw:]
            elif (
                clip_ssm
                and blocks
                and (n_spec_blocks := self._ssm_spec_blocks[i]) is not None
            ):
                if n_spec := min(n_spec_blocks, len(blocks) - 1):
                    blocks = blocks[:-n_spec]
                if not self._ssm_state_slots_are_positional:
                    # Never empty: downstream reads that as a full prefix hit.
                    blocks = blocks[-1:]
            clipped.append(blocks)
        return tuple(clipped)

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        self._block_pool = gpu_block_pool

    def _export_mamba_states(
        self, request: "Request", num_tokens: int, block_ids: BlockIds
    ) -> list[list[Any]]:
        """Return the exact recurrent states exported with a held request.

        On D these are what a P may load of the held history. On P the range
        is the prompt's last boundary, for D to cache (`_mamba_boundary_state`).

        The entries are each block boundary from the prompt's last one up to
        ``num_tokens`` whose state every Mamba group has in the prefix cache
        (align mode caches a block only once it holds the state at its end),
        pinned until the hold ends, and the running state at ``num_tokens``,
        in the request's last state slot, where no speculative step can have
        left the accepted state in a scratch slot and the computed count was
        not cut back by the output's end.

        Args:
            request: The finished request whose blocks are held.
            num_tokens: The tokens whose KV the hold offers.
            block_ids: The request's exchange-clipped block ids.

        Returns:
            ``[num_tokens, [block id per Mamba group]]`` entries, ascending.
        """
        if not self._mamba_prefix_pull:
            return []
        block_size = self.block_size
        states: list[list[Any]] = []
        pool = self._block_pool
        if pool is not None and pool.enable_caching:
            from vllm.v1.core.kv_cache_utils import resolve_block_hashes

            hashes = resolve_block_hashes(
                request.block_hashes, pool.hash_block_size, block_size
            )
            pins: list[KVCacheBlock] = []
            # From the block ending at the prompt's last boundary: earlier ones
            # the P that prefilled the prompt holds itself.
            first = max(request.num_prompt_tokens // block_size - 1, 0)
            for i in range(first, min(num_tokens // block_size, len(hashes))):
                blocks = pool.get_cached_block(hashes[i], self._mamba_group_ids)
                if blocks is None:
                    continue
                pool.touch(blocks)
                pins.extend(blocks)
                states.append([(i + 1) * block_size, [b.block_id for b in blocks]])
            if pins:
                self._mamba_state_pins[request.request_id] = pins
        if (
            not any(self._ssm_spec_blocks[g] for g in self._mamba_group_ids)
            and request.num_computed_tokens <= request.num_tokens
            and num_tokens % block_size
            and all(block_ids[g] for g in self._mamba_group_ids)
        ):
            states.append(
                [num_tokens, [block_ids[g][-1] for g in self._mamba_group_ids]]
            )
        return states

    def _release_mamba_state_pins(self, request_id: ReqId) -> None:
        pins = self._mamba_state_pins.pop(request_id, None)
        if pins:
            assert self._block_pool is not None
            self._block_pool.free_blocks(pins)

    def _mamba_pull_state(
        self, request: "Request", num_computed_tokens: int
    ) -> list[Any] | None:
        """Return the state of D's export that P loads, or None for no pull.

        See `select_mamba_pull_state`. An export from a D that names no exact
        states also gives None.
        """
        params = request.kv_transfer_params
        assert params is not None
        states = params.get("remote_mamba_states")
        if (
            not self._mamba_prefix_pull
            or states is None
            or params.get("remote_block_size") != self.block_size
        ):
            return None
        return select_mamba_pull_state(
            states,
            params.get("remote_num_tokens") or 0,
            request.num_prompt_tokens,
            num_computed_tokens,
            self.block_size,
        )

    def _mamba_boundary_state(
        self, request: "Request", num_tokens: int, num_computed_tokens: int
    ) -> list[Any] | None:
        """Return the state P exported at the last block boundary inside D's load.

        Runs on D. The load covers `num_tokens`, and the boundary must lie past
        D's own hit. The state comes from `_export_mamba_states`; None for no
        such state.

        A prompt prefilled here caches the state at that boundary, which a
        later stage extending the prompt resumes from. A load ending past it
        writes only the running state at its end, so D reads this one beside
        it (`get_external_boundary_tokens`, `_add_mamba_boundary_read`).
        """
        params = request.kv_transfer_params
        assert params is not None
        states = params.get("remote_mamba_states")
        if not states or params.get("remote_block_size") != self.block_size:
            return None
        pool = self._block_pool
        if not self._mamba_prefix_pull or pool is None or not pool.enable_caching:
            return None
        boundary = num_tokens // self.block_size * self.block_size
        if not num_computed_tokens < boundary < num_tokens:
            return None
        return next((entry for entry in states if entry[0] == boundary), None)

    def get_external_boundary_tokens(self, request: "Request") -> int:
        """Return the boundary of the state `_mamba_boundary_state` chose."""
        params = request.kv_transfer_params
        state = params.get("_mamba_boundary_state") if params else None
        return state[0] if state else 0

    def _add_mamba_boundary_read(
        self,
        params: dict[str, Any],
        state: list[Any],
        blocks: "KVCacheBlocks",
        local_block_ids: BlockIds,
    ) -> BlockIds:
        """Pair P's boundary state with the block that backs that boundary on D.

        Runs on D. The block comes from the KV cache manager, and the pair goes
        ahead of the running state in each Mamba group; the worker pairs
        equal-length lists slot for slot. Nothing changes where no such block
        was given.

        Returns:
            The local block ids to read into; `params` gets the remote ones.
        """
        idx = state[0] // self.block_size - 1
        remote = list(params["remote_block_ids"])
        targets = []
        for g in self._mamba_group_ids:
            group = blocks.blocks[g]
            block = group[idx] if idx < len(group) else None
            if (
                block is None
                or block.is_null
                or block.block_hash is not None
                or len(local_block_ids[g]) != 1
                or len(remote[g]) != 1
            ):
                return local_block_ids
            targets.append(block.block_id)
        local = list(local_block_ids)
        for g, local_id, remote_id in zip(self._mamba_group_ids, targets, state[1]):
            local[g] = [local_id, *local[g]]
            remote[g] = [remote_id, *remote[g]]
        params["remote_block_ids"] = remote
        return tuple(local)

    def _cut_to_mamba_state(self, params: dict[str, Any], state: list[Any]) -> None:
        """Point P's read at ``state``.

        Each attention group keeps its blocks up to that position, which the
        worker pairs from the end with the blocks P allocated. Each Mamba group
        keeps the block holding the state, paired with P's one state slot.
        """
        num_blocks = cdiv(state[0], self.block_size)
        mamba_ids = iter(state[1])
        params["remote_block_ids"] = [
            [next(mamba_ids)] if i in self._mamba_group_ids else group[:num_blocks]
            for i, group in enumerate(params["remote_block_ids"])
        ]
        params["remote_num_tokens"] = state[0]

    def set_xfer_handshake_metadata(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        encoded_data: dict[tuple[int, int], bytes] = {}
        encoder = msgspec.msgpack.Encoder()
        for (pp_rank, tp_rank), rank_metadata in metadata.items():
            if not isinstance(rank_metadata, NixlHandshakePayload):
                raise ValueError(
                    "NixlConnectorScheduler expects NixlHandshakePayload for "
                    "handshake metadata."
                )
            encoded_data[(pp_rank, tp_rank)] = encoder.encode(rank_metadata)
            logger.debug(
                "PP rank %d, TP rank %d: encoded NixlHandshakePayload size: %s bytes",
                pp_rank,
                tp_rank,
                str(len(encoded_data[(pp_rank, tp_rank)])),
            )

        # Only start the listener when we have metadata to serve.
        if self._nixl_handshake_listener_t is None:
            ready_event = threading.Event()
            self._nixl_handshake_listener_t = threading.Thread(
                target=self._nixl_handshake_listener,
                args=(
                    encoded_data,
                    ready_event,
                    self._stop_event,
                    self.side_channel_host,
                    self.side_channel_port,
                ),
                daemon=True,
                name="nixl_handshake_listener",
            )
            self._nixl_handshake_listener_t.start()
            ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    @staticmethod
    def _nixl_handshake_listener(
        encoded_data: dict[tuple[int, int], Any],
        ready_event: threading.Event,
        stop_event: threading.Event,
        host: str,
        port: int,
    ):
        """Background thread for getting new NIXL handshakes."""
        # NOTE(rob): this is a simple implementation. We will move
        # to a better approach via HTTP endpoint soon.

        # Listen for new requests for metadata.
        path = make_zmq_path("tcp", host, port)
        logger.debug("Starting listening on path: %s", path)
        with zmq_ctx(zmq.ROUTER, path) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            ready_event.set()
            while True:
                try:
                    identity, _, msg = sock.recv_multipart()
                except zmq.Again:
                    if stop_event.is_set():
                        break
                    continue
                # Decode (GET_META_MSG, pp_rank, tp_rank).
                msg, target_pp_rank, target_tp_rank = msgspec.msgpack.decode(msg)
                logger.debug(
                    "Received message for pp rank %s, tp rank %s",
                    target_pp_rank,
                    target_tp_rank,
                )
                if msg != GET_META_MSG:
                    logger.warning("Connection listener got unexpected message %s", msg)
                # Echo our perf_counter so P can estimate the clock offset.
                # perf_counter is only comparable within a process, so this
                # listener must run in the same process that stamps the block
                # expiry deadline (`_reqs_need_send`).
                ts = msgspec.msgpack.encode(time.perf_counter())
                sock.send_multipart(
                    (identity, b"", encoded_data[(target_pp_rank, target_tp_rank)], ts)
                )

    def _get_remote_prefill_token_count(self, num_prompt_tokens: int) -> int:
        """D-side only. Returns N-1 for Mamba models since the decoder
        always recomputes the last token and must start from h(N-1)."""
        if self._has_mamba and num_prompt_tokens > 1:
            return num_prompt_tokens - 1
        return num_prompt_tokens

    def _truncate_mamba_request_for_prefill(self, request: "Request") -> None:
        """P-side only: drop the last prompt token so the prefiller computes
        h(N-1) instead of h(N). The decoder recomputes the last token to
        derive h(N) correctly.

        Guarded by ``_p_side_truncated`` to avoid repeated truncation if the
        request is preempted and rescheduled."""
        params = request.kv_transfer_params
        if (
            params is not None
            # Guard against repeated truncation after preemption/reschedule.
            and not params.get("_p_side_truncated")
            and request.num_prompt_tokens > 1
        ):
            if request.prompt_token_ids is not None:
                request.prompt_token_ids.pop()
            elif request.prompt_embeds is not None:
                request.prompt_embeds = request.prompt_embeds[:-1]
            else:
                return

            request._all_token_ids.pop()
            request.num_prompt_tokens -= 1
            request.max_tokens = 1
            params["_p_side_truncated"] = True

    def _build_save_meta(
        self,
        meta: NixlConnectorMetadata,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # only called when use_host_buffer is True to build the save metadata

        # NOTE: For the prefill side, there might be a chance that an early added
        # request is a chunked prefill, so we need to check if new blocks are added
        for req_id, new_block_id_groups, _ in yield_req_data(scheduler_output):
            req_to_save = self._reqs_need_save.get(req_id)
            if req_to_save is None or new_block_id_groups is None:
                continue
            req = req_to_save

            assert req.kv_transfer_params is not None
            clipped_block_id_groups = self.get_exchange_clipped_blocks(
                new_block_id_groups, clip_ssm=False
            )
            meta.add_new_req_to_save(
                request_id=req_id,
                local_block_ids=clipped_block_id_groups,
                kv_transfer_params=req.kv_transfer_params,
            )
            assert scheduler_output.num_scheduled_tokens is not None
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            is_partial = (
                req.num_computed_tokens + num_scheduled_tokens
            ) < req.num_prompt_tokens
            if not is_partial:
                # For non-partial prefills, once new req_meta is scheduled, it
                # can be removed from _reqs_need_save.
                # For partial prefill case, we will retain the request in
                # _reqs_need_save until all blocks are scheduled with req_meta.
                # Therefore, only pop if `not is_partial`.
                self._reqs_need_save.pop(req_id)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = NixlConnectorMetadata()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )

        if self.use_host_buffer:
            self._build_save_meta(meta, scheduler_output)

        meta.reqs_to_send = self._reqs_need_send
        meta.reqs_in_batch = self._reqs_in_batch
        meta.reqs_not_processed = self._reqs_not_processed

        # Package heartbeats, throttled by heartbeat_interval.
        if self._heartbeat_by_engine:
            now = time.perf_counter()
            if now - self._last_heartbeat_time >= self._heartbeat_interval:
                self._last_heartbeat_time = now
                meta.heartbeat_by_engine = self._heartbeat_by_engine

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()
        self._reqs_in_batch = set()
        self._reqs_not_processed = set()
        self._reqs_need_send = {}

        return meta

    def release_hold(self, request_id: ReqId) -> None:
        """End a finished request's hold on its blocks: no reader will come.

        The blocks are freed as if the hold had expired, at the worker's next
        `get_transfer_results`; an id the worker holds nothing for is ignored
        there.
        """
        self._reqs_need_send[request_id] = 0.0

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        """Stop heartbeating for requests whose KV transfer completed."""
        for req_id in connector_output.finished_recving or ():
            self._stop_heartbeat(req_id)

    def has_pending_push_work(self) -> bool:
        return False

    ############################################################
    # Abstract methods that subclasses must implement
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        raise NotImplementedError

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        raise NotImplementedError

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        raise NotImplementedError

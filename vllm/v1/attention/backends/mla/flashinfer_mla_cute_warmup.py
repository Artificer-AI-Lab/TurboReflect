# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode calls that compile every FlashInfer CuTe-DSL MLA kernel a run reaches.

FlashInfer 0.6.16's monolithic CuTe-DSL MLA decode caches one kernel per
dtype, page size, head count, query-length bound, variable-Q use, reducer D
tiling (1, 2 or 4) and whether split-KV needs a workspace
(`_get_compiled_mla_kernel`). Batch, query rows, page count and context length
are dynamic shapes that reach the key only through the split count and the
reducer tiling, so a few small batches per query length cover every kernel.
"""

from vllm.utils.math_utils import cdiv, round_up

# KV tile of the CuTe MLA decode; its page table width must be a multiple of
# 128 / page_size, which vLLM's block table pads to (`BlockTable`).
_KV_TILE = 128


def cute_decode_page_table_width(max_model_len: int, page_size: int) -> int:
    return round_up(cdiv(max_model_len, page_size), _KV_TILE // page_size)


def cute_decode_warmup_calls(
    q_len: int, num_heads: int, num_sms: int, max_batch: int, max_model_len: int
) -> list[tuple[int, int]]:
    """(batch, context length) pairs reaching every kernel variant of `q_len`.

    Reducer tiling stops depending on the batch once its rows fill the SMs.
    Contexts of one KV tile, two tiles and the longest give no split, at most
    two splits and the most splits. For `q_len > 1` the caller lays the batch
    out as one request at `q_len` and the rest at one token, the fewest query
    tiles (so the most splits) that batch can have.
    """
    rows = num_heads * q_len
    batches = range(1, min(max_batch, cdiv(num_sms, rows) + 1) + 1)
    seq_lens = dict.fromkeys(
        min(s, max_model_len) for s in (max(_KV_TILE, q_len), 2 * _KV_TILE)
    )
    seq_lens[max_model_len] = None
    return [(batch, seq_len) for batch in batches for seq_len in seq_lens]

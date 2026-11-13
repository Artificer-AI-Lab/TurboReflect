# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import triton
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.ops.triton_decode_attention import (
    decode_attention_fwd,
    decode_attention_fwd_verify,
)
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

logger = init_logger(__name__)

# num_kv_splits selection (shared by forward_mqa and the workspace reservation
# so the two cannot drift). Both are hardware dependent.
_MIN_WORK_PER_SPLIT = 512
_SPLIT_OCCUPANCY_MULTIPLIER = 2


def _compute_num_kv_splits(max_seq_len: int, sm_count: int) -> int:
    # Power of 2 to avoid excessive kernel instantiations, capped by an SM-based
    # maximum (occupancy multiplier allows multiple blocks per SM
    # for latency hiding).
    ideal_splits = triton.next_power_of_2(max(1, max_seq_len // _MIN_WORK_PER_SPLIT))
    max_splits = sm_count * _SPLIT_OCCUPANCY_MULTIPLIER
    return min(ideal_splits, max_splits)


# Flattened query/head rows per grouped (g32) verify tile.
_VERIFY_TILE_ROWS = 32
# Stable log marker for an explicitly chosen uniform verify under spec decode.
TRITON_MLA_UNIFORM_VERIFY_MARKER = "TRITON_MLA_UNIFORM_VERIFY"
# num_kv_splits variants compiled in this process.
_WARMED_SPLITS: set[int] = set()


def _check_verify_variant(vllm_config) -> None:
    """Refuse speculative decoding unless the verify variant was chosen.

    Runs where the backend is built, so it covers every worker under any
    executor and auto-selected backends. Under the default ``uniform`` a
    ragged verify batch silently runs MLA prefill.

    Raises:
        ValueError: ``VLLM_TRITON_MLA_VERIFY`` is unset under speculative
            decoding.
    """
    if vllm_config.speculative_config is None:
        return
    if not envs.is_set("VLLM_TRITON_MLA_VERIFY"):
        raise ValueError(
            "TRITON_MLA with speculative decoding needs VLLM_TRITON_MLA_VERIFY "
            "set explicitly: unset, ragged verify batches run MLA prefill. "
            "Set VLLM_TRITON_MLA_VERIFY=g32, or =uniform to diagnose that path."
        )
    if envs.VLLM_TRITON_MLA_VERIFY == "uniform":
        logger.warning_once(
            "%s: VLLM_TRITON_MLA_VERIFY=uniform under speculative decoding; "
            "ragged verify batches run MLA prefill (diagnostic only)",
            TRITON_MLA_UNIFORM_VERIFY_MARKER,
        )


class TritonMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    # forward_mqa flattens a uniform multi-token block to one decode row per
    # query token, so spec-decode verification takes the decode path.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        _check_verify_variant(vllm_config)
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._reserve_attn_logits_workspace()

    def _reserve_attn_logits_workspace(self) -> None:
        """Pre-size the shared workspace for the decode split-KV attn logits.

        Reserving at the worst case (max_model_len -> max num_kv_splits,
        max_num_seqs decode tokens) before warmup/cudagraph capture means the
        per-call ``get_simultaneous`` in ``forward_mqa`` never has to grow the
        buffer at runtime (which would raise once the workspace is locked).
        """
        if not is_workspace_manager_initialized():
            return
        # forward_mqa flattens each request's block to query_len decode rows,
        # and query_len is bounded by the reorder threshold.
        B = (
            self.vllm_config.scheduler_config.max_num_seqs
            * self.reorder_batch_threshold
        )
        # DCP all-gathers the query heads before forward_mqa.
        q_num_heads = self.num_heads * self.dcp_world_size
        max_splits = _compute_num_kv_splits(
            self.model_config.max_model_len,
            current_platform.num_compute_units(),
        )
        lse_dim = self.mla_dims.kv_lora_rank + 1
        current_workspace_manager().get_simultaneous(
            ((B, q_num_heads, max_splits, lse_dim), torch.float32),
        )


def build_verify_rows(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_rows: int,
    num_heads: int,
    split_by_request: bool,
    grouped: bool,
    skip_single: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Causal verify rows for a ragged decode batch, without a host sync.

    Args:
        query_start_loc: ``[num_reqs + 1]`` int32 query offsets.
        seq_lens: ``[num_reqs]`` int32 final sequence lengths.
        num_rows: Query rows handed to the kernel; rows past the last query
            (padded uniform batches) get zero extent.
        num_heads: Query heads per row.
        split_by_request: Partition KV splits by the request's final length
            instead of each row's causal length.
        grouped: Also build g32 query tiles.
        skip_single: Leave single-token requests to the decode kernel: their
            rows get a zero split length and no query tile.

    Returns:
        ``(row_req, causal_lens, split_lens, query_tiles)``. ``query_tiles`` is
        ``None`` unless grouped; its length depends only on ``num_rows`` and
        the request count, and unused tiles have a zero query count.
    """
    num_reqs = seq_lens.shape[0]
    device = seq_lens.device
    query_ends = query_start_loc[1:]
    query_lens = query_ends - query_start_loc[:-1]
    if skip_single:
        query_lens = torch.where(query_lens > 1, query_lens, 0)
    row_ids = torch.arange(num_rows, dtype=torch.int32, device=device)
    row_req = torch.searchsorted(query_ends, row_ids, right=True, out_int32=True)
    valid = row_req < num_reqs
    row_req.clamp_(max=num_reqs - 1)
    req_seq_lens = seq_lens.index_select(0, row_req)
    causal_lens = torch.where(
        valid, req_seq_lens - query_ends.index_select(0, row_req) + row_ids + 1, 0
    )
    split_lens = (
        torch.where(valid, req_seq_lens, 0) if split_by_request else causal_lens
    )
    if skip_single:
        split_lens = torch.where(query_lens.index_select(0, row_req) > 0, split_lens, 0)
    if not grouped:
        return row_req, causal_lens, split_lens, None

    # sum(ceil(q_i * H / T)) <= ceil(rows * H / T) + num_reqs
    num_tiles = triton.cdiv(num_rows * num_heads, _VERIFY_TILE_ROWS) + num_reqs
    tiles_per_req = (
        query_lens * num_heads + _VERIFY_TILE_ROWS - 1
    ) // _VERIFY_TILE_ROWS
    tile_ends = tiles_per_req.cumsum(0, dtype=torch.int32)
    tile_ids = torch.arange(num_tiles, dtype=torch.int32, device=device)
    tile_req = torch.searchsorted(tile_ends, tile_ids, right=True, out_int32=True)
    tile_valid = tile_req < num_reqs
    tile_req.clamp_(max=num_reqs - 1)
    first_tile = (tile_ends - tiles_per_req).index_select(0, tile_req)
    query_tiles = torch.stack(
        [
            query_start_loc.index_select(0, tile_req),
            (tile_ids - first_tile) * _VERIFY_TILE_ROWS,
            torch.where(tile_valid, query_lens.index_select(0, tile_req), 0),
            torch.where(tile_valid, seq_lens.index_select(0, tile_req), 0),
        ],
        dim=1,
    )
    return row_req, causal_lens, split_lens, query_tiles


@dataclass
class TritonMLAVerifyDecodeMetadata(MLACommonDecodeMetadata):
    """Causal verify rows; see ``decode_attention_fwd_verify``.

    ``block_table`` and ``seq_lens`` stay per request. Row ``r`` of request
    ``i`` (query ``j`` of ``q_i``) reads block-table row ``row_req[r] = i`` and
    sees ``causal_lens[r] = s_i - q_i + j + 1`` KV entries. When
    ``single_rows`` is set, those rows (the single-token requests) run on the
    decode kernel with ``single_block_table`` and ``single_seq_lens``.
    """

    row_req: torch.Tensor
    causal_lens: torch.Tensor
    split_lens: torch.Tensor
    query_tiles: torch.Tensor | None
    num_query_tiles: int
    tile_heads: int
    single_rows: torch.Tensor | None = None
    single_block_table: torch.Tensor | None = None
    single_seq_lens: torch.Tensor | None = None


class TritonMLAVerifyMetadataBuilder(TritonMLAMetadataBuilder):
    """Route every short query length to decode as causal per-token rows.

    Used for ``VLLM_TRITON_MLA_VERIFY`` r16/r32/g32. A batch whose longest
    query is one token keeps the uniform builder's metadata. Row and tile
    tables are written into persistent buffers each step and the kernel grid
    depends only on the row and request counts, so a padded uniform batch may
    replay a FULL graph as on the uniform path (UNIFORM_BATCH). Ragged batches
    never match the uniform dispatch key and run under PIECEWISE graphs, so
    their single-token decode rows use per-step tensors.
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.VARLEN
    route_single: bool = False

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if self.dcp_world_size > 1:
            raise NotImplementedError("TritonMLA verify rows need DCP=1")
        self.grouped = envs.VLLM_TRITON_MLA_VERIFY == "g32"
        self.split_by_request = envs.VLLM_TRITON_MLA_VERIFY_SPLIT == "request"
        if self.grouped and not self.split_by_request:
            raise ValueError(
                "g32 shares KV blocks across a request's queries and needs "
                "VLLM_TRITON_MLA_VERIFY_SPLIT=request"
            )
        max_reqs = vllm_config.scheduler_config.max_num_seqs
        max_rows = max_reqs * self.reorder_batch_threshold
        self._row_req = torch.zeros(max_rows, dtype=torch.int32, device=device)
        self._causal_lens = torch.zeros_like(self._row_req)
        self._split_lens = torch.zeros_like(self._row_req)
        self._query_tiles: torch.Tensor | None = None
        if self.grouped:
            max_tiles = (
                triton.cdiv(max_rows * self.num_heads, _VERIFY_TILE_ROWS) + max_reqs
            )
            self._query_tiles = torch.zeros(
                max_tiles, 4, dtype=torch.int32, device=device
            )
        self.route_single = envs.VLLM_TRITON_MLA_VERIFY in ("r32", "g32") and (
            envs.VLLM_TRITON_MLA_VERIFY_ROUTE_Q1
        )
        logger.info_once(
            "TritonMLA verify: %s, KV split by %s, %d warps, q=1 rows on %s kernel",
            envs.VLLM_TRITON_MLA_VERIFY,
            "request" if self.split_by_request else "row",
            envs.VLLM_TRITON_MLA_VERIFY_WARPS,
            "decode" if self.route_single else "verify",
        )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        max_query_len: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> MLACommonDecodeMetadata:
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        if int(query_lens_cpu.max()) <= 1:
            return super()._build_decode(
                block_table_tensor,
                seq_lens_device,
                max_seq_len,
                query_start_loc_cpu,
                query_start_loc_device,
                num_decode_tokens,
                max_query_len,
                dcp_tot_seq_lens_device,
            )
        rows = num_decode_tokens
        if rows > self._row_req.shape[0]:
            raise ValueError(
                f"{rows} verify rows exceed the {self._row_req.shape[0]} reserved"
            )
        num_single = int((query_lens_cpu == 1).sum()) if self.route_single else 0
        single_rows = single_block_table = single_seq_lens = None
        if num_single:
            # Stable sort puts the single-token requests first, in batch order,
            # without a host sync.
            query_lens = query_start_loc_device[1:] - query_start_loc_device[:-1]
            single = torch.sort((query_lens != 1).int(), stable=True).indices
            single = single[:num_single]
            single_rows = query_start_loc_device.index_select(0, single).long()
            single_block_table = block_table_tensor.index_select(0, single)
            single_seq_lens = seq_lens_device.index_select(0, single)
        row_req, causal_lens, split_lens, tiles = build_verify_rows(
            query_start_loc_device,
            seq_lens_device,
            rows,
            self.num_heads,
            self.split_by_request,
            self.grouped,
            skip_single=num_single > 0,
        )
        self._row_req[:rows].copy_(row_req)
        self._causal_lens[:rows].copy_(causal_lens)
        self._split_lens[:rows].copy_(split_lens)
        num_tiles = 0
        if self._query_tiles is not None:
            assert tiles is not None
            num_tiles = tiles.shape[0]
            self._query_tiles[:num_tiles].copy_(tiles)

        return TritonMLAVerifyDecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            dcp_tot_seq_lens=None,
            row_req=self._row_req[:rows],
            causal_lens=self._causal_lens[:rows],
            split_lens=self._split_lens[:rows],
            query_tiles=self._query_tiles,
            num_query_tiles=num_tiles,
            tile_heads=self.num_heads,
            single_rows=single_rows,
            single_block_table=single_block_table,
            single_seq_lens=single_seq_lens,
        )


class TritonMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3)
        return (0, 1, 2)

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["TritonMLAImpl"]:
        return TritonMLAImpl

    @staticmethod
    def get_builder_cls() -> type[MLACommonMetadataBuilder[MLACommonMetadata]]:
        if envs.VLLM_TRITON_MLA_VERIFY == "uniform":
            return TritonMLAMetadataBuilder
        return TritonMLAVerifyMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonMLAImpl(MLACommonImpl[MLACommonMetadata]):
    can_return_lse_for_decode: bool = True
    _warm_splits: tuple[int, ...] = ()

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TritonMLAImpl"
            )

        if current_platform.is_cuda():
            cap = current_platform.get_device_capability()
            cap_str = cap.as_version_str() if cap is not None else "unknown"
            dev = current_platform.get_device_name()
            if self.kv_cache_dtype.startswith("fp8") and not (
                current_platform.has_device_capability(89)
            ):
                suggested = (
                    "float16" if (cap is None or cap.to_int() < 80) else "bfloat16"
                )
                raise ValueError(
                    f"FP8 KV cache is not supported by the Triton MLA backend "
                    f"on {dev} (compute capability {cap_str}); native FP8 "
                    f"(fp8e4nv) requires SM89+. Re-run with "
                    f"--kv-cache-dtype {suggested}."
                )
            if self.kv_cache_dtype == "bfloat16" and not (
                current_platform.has_device_capability(80)
            ):
                raise ValueError(
                    f"bfloat16 KV cache is not supported by the Triton MLA "
                    f"backend on {dev} (compute capability {cap_str}); "
                    f"bfloat16 requires SM80+. Re-run with "
                    f"--kv-cache-dtype float16."
                )

        # For FP8 KV cache, we dequantize to BF16 on load inside the
        # Triton kernel. Tell the common layer not to quantize queries
        # to FP8 — we handle FP8 KV cache with BF16 queries (Mode 1).
        if is_quantized_kv_cache(self.kv_cache_dtype):
            self.supports_quant_query_input = False

        self._sm_count = current_platform.num_compute_units()
        self._verify_block_h = (
            _VERIFY_TILE_ROWS if envs.VLLM_TRITON_MLA_VERIFY in ("r32", "g32") else 16
        )
        self._verify_num_warps = envs.VLLM_TRITON_MLA_VERIFY_WARPS
        if current_platform.is_cuda_alike() and not envs.VLLM_BATCH_INVARIANT:
            config = get_current_vllm_config_or_none()
            model_config = config.model_config if config else None
            max_len = model_config.max_model_len if model_config else 1 << 30
            max_splits = _compute_num_kv_splits(max_len, self._sm_count)
            self._warm_splits = tuple(
                sorted(
                    {
                        min(1 << k, max_splits)
                        for k in range(max_splits.bit_length() + 1)
                    }
                    | {max_splits}
                )
            )

    def _warm_split_variants(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_c_cache: torch.Tensor,
        page_size: int,
        block_table: torch.Tensor,
        layer: AttentionLayer,
    ) -> None:
        """JIT-compile the stage kernels for every eager ``num_kv_splits``.

        FULL graphs capture decode at the largest split count, while eager
        steps pick a power of two from their longest sequence, so each new
        count would otherwise compile on its first step. Rows of zero length
        launch the kernels with the serving strides without reading the cache.
        """
        todo = [s for s in self._warm_splits if s not in _WARMED_SPLITS]
        self._warm_splits = ()
        if not todo:
            return
        _WARMED_SPLITS.update(todo)
        heads = q.shape[1]
        q = q[:1]
        zeros = torch.zeros(1, dtype=torch.int32, device=q.device)
        tiles = torch.zeros(1, 4, dtype=torch.int32, device=q.device)
        grouped = envs.VLLM_TRITON_MLA_VERIFY == "g32"
        o = q.new_empty(1, heads, self.kv_lora_rank)
        lse = q.new_empty(1, heads)
        try:
            for splits in todo:
                logits = torch.empty(
                    1,
                    heads,
                    splits,
                    self.kv_lora_rank + 1,
                    dtype=torch.float32,
                    device=q.device,
                )
                decode_attention_fwd(
                    q,
                    kv_cache,
                    kv_c_cache,
                    o,
                    lse,
                    block_table[:1],
                    zeros,
                    logits,
                    splits,
                    self.scale,
                    page_size,
                    k_scale=layer._k_scale,
                    v_scale=layer._k_scale,
                    is_mla=True,
                )
                if envs.VLLM_TRITON_MLA_VERIFY == "uniform":
                    continue
                decode_attention_fwd_verify(
                    q,
                    kv_cache,
                    kv_c_cache,
                    o,
                    lse,
                    block_table[:1],
                    zeros,
                    zeros,
                    zeros,
                    logits,
                    splits,
                    self.scale,
                    page_size,
                    layer._k_scale,
                    query_tiles=tiles if grouped else None,
                    num_query_tiles=1 if grouped else 0,
                    block_h=self._verify_block_h,
                    num_warps=self._verify_num_warps,
                )
        except Exception as e:
            logger.warning("TritonMLA: KV split warmup failed: %s", e)
            return
        logger.info("TritonMLA: compiled decode/verify for KV splits %s", todo)

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]
        q_num_heads = q.shape[1]
        o = torch.zeros(
            B, q_num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )
        lse = torch.zeros(B, q_num_heads, dtype=q.dtype, device=q.device)

        # For batch invariance, use only 1 split to ensure deterministic reduction
        if envs.VLLM_BATCH_INVARIANT:
            num_kv_splits = 1
        else:
            num_kv_splits = _compute_num_kv_splits(
                attn_metadata.max_seq_len, self._sm_count
            )

        # NOTE: the +1 stores the LogSumExp (LSE) that the stage2 kernel uses to
        # merge partial attention outputs across splits. The scratch is served
        # from the shared workspace (reserved at max in the metadata builder), so
        # there is no per-call allocation on the decode hot path. Fall back to a
        # direct allocation when the workspace manager is not initialized (e.g.
        # unit tests without a GPUModelRunner).
        logits_shape = (B, q_num_heads, num_kv_splits, self.kv_lora_rank + 1)
        if is_workspace_manager_initialized():
            (attn_logits,) = current_workspace_manager().get_simultaneous(
                (logits_shape, torch.float32),
            )
        else:
            attn_logits = torch.empty(
                logits_shape, dtype=torch.float32, device=q.device
            )

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        decode = attn_metadata.decode
        if self._warm_splits and not torch.cuda.is_current_stream_capturing():
            self._warm_split_variants(
                q, kv_c_and_k_pe_cache, kv_c_cache, PAGE_SIZE, decode.block_table, layer
            )
        if isinstance(decode, TritonMLAVerifyDecodeMetadata):
            assert q_num_heads == decode.tile_heads, (
                f"verify tiles built for {decode.tile_heads} heads, q has {q_num_heads}"
            )
            decode_attention_fwd_verify(
                q,
                kv_c_and_k_pe_cache,
                kv_c_cache,
                o,
                lse,
                decode.block_table,
                decode.row_req,
                decode.causal_lens,
                decode.split_lens,
                attn_logits,
                num_kv_splits,
                self.scale,
                PAGE_SIZE,
                layer._k_scale,
                query_tiles=decode.query_tiles,
                num_query_tiles=decode.num_query_tiles,
                block_h=self._verify_block_h,
                num_warps=self._verify_num_warps,
            )
            single_rows = decode.single_rows
            if single_rows is not None:
                # Verify wrote placeholders for these rows and is done with the
                # logits scratch, so decode reuses it and overwrites the rows.
                assert decode.single_block_table is not None
                assert decode.single_seq_lens is not None
                n = single_rows.shape[0]
                o_single = o.new_empty(n, q_num_heads, self.kv_lora_rank)
                lse_single = lse.new_empty(n, q_num_heads)
                decode_attention_fwd(
                    q.index_select(0, single_rows),
                    kv_c_and_k_pe_cache,
                    kv_c_cache,
                    o_single,
                    lse_single,
                    decode.single_block_table,
                    decode.single_seq_lens,
                    attn_logits[:n],
                    num_kv_splits,
                    self.scale,
                    PAGE_SIZE,
                    k_scale=layer._k_scale,
                    v_scale=layer._k_scale,
                    is_mla=True,
                )
                o.index_copy_(0, single_rows, o_single)
                lse.index_copy_(0, single_rows, lse_single)
            return o, lse

        block_table = attn_metadata.decode.block_table
        seq_lens = attn_metadata.decode.seq_lens
        # decode_attention_fwd has no causal flag: it launches one program per
        # row of q and reads that row's KV extent from seq_lens, so intra-block
        # causality is expressed as per-row extents. Deriving query_len from the
        # tensors the kernel indexes keeps the three row counts in step.
        num_decodes = seq_lens.shape[0]
        query_len, remainder = divmod(B, num_decodes) if num_decodes else (1, 0)
        assert remainder == 0, (
            f"non-uniform decode block: {B} query rows over {num_decodes} requests"
        )
        if query_len > 1:
            block_table = block_table.repeat_interleave(query_len, dim=0)
            # Per-row extents are offsets off the global sequence length;
            # under DCP seq_lens is this rank's local slice instead.
            assert self.dcp_world_size == 1, (
                "causal multi-token decode is not supported with DCP"
            )
            # Row t attends the prefix plus block tokens 0..t. Clamp holds
            # padding rows at extent 0; the kernel skips them either way
            # (split_kv_end == split_kv_start).
            offsets = torch.arange(
                1 - query_len, 1, device=seq_lens.device, dtype=seq_lens.dtype
            )
            seq_lens = (seq_lens.unsqueeze(1) + offsets).flatten().clamp(min=0)

        # Run MQA — always pass layer scales. When KV cache is
        # BF16 the kernel's `if dtype.is_fp8()` check is a no-op.
        decode_attention_fwd(
            q,
            kv_c_and_k_pe_cache,
            kv_c_cache,
            o,
            lse,
            block_table,
            seq_lens,
            attn_logits,
            num_kv_splits,
            self.scale,
            PAGE_SIZE,
            k_scale=layer._k_scale,
            v_scale=layer._k_scale,
            is_mla=True,
        )

        return o, lse

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
from flashinfer.utils import (
    get_device_sm_count,
    get_trtllm_gen_multi_ctas_kv_counter_bytes,
)

from vllm.config import VllmConfig, get_current_vllm_config
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
from vllm.model_executor.warmup.cutedsl_warmup import (
    CuTeDSLCompileUnit,
    register_cutedsl_warmup_provider,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_cute_warmup import (
    cute_decode_page_table_width,
    cute_decode_warmup_calls,
)
from vllm.v1.attention.backends.utils import KVCacheLayoutType
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


def _trtllm_gen_mla_decode_supports_num_heads(num_heads: int) -> bool:
    """True if trtllm-gen's MLA decode kernel tiles this query head count exactly.

    The kernel groups Q heads into CTAs of ``min(num_heads, tileSizeQ)`` and,
    before FlashInfer 0.6.16, raises "The numHeadsQ/numHeadsKv is not supported"
    unless ``num_heads`` is divisible by that (flashinfer fmhaKernels.cuh).
    ``tileSizeQ`` is 8/16 for ``num_heads <= 8``/``<= 32`` (SwapsMmaAb) else 64;
    treat ``> 32`` as tile 64 (safe bound). E.g. 20/96 -> False, 48/64/128 -> True.
    FlashInfer 0.6.16 pads the other counts up to <= 64 heads to a power-of-two
    tile instead (getPaddedMlaTileSizeQ); they still go to CuTeDSL.
    """
    if num_heads <= 8:
        tile = 8
    elif num_heads <= 32:
        tile = 16
    else:
        tile = 64
    return num_heads % min(num_heads, tile) == 0


def _select_mla_decode_backend(num_heads: int) -> str | None:
    """cute-dsl for head counts trtllm-gen cannot tile, else None (=> auto)."""
    if not _trtllm_gen_mla_decode_supports_num_heads(num_heads):
        logger.warning_once(
            "trtllm-gen MLA decode does not tile num_heads=%d exactly; "
            "using the cute-dsl backend.",
            num_heads,
        )
        return "cute-dsl"
    return None


@dataclass(frozen=True)
class _CuteDecodeWarmupShape:
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    page_size: int
    max_q_len: int
    max_batch: int
    max_model_len: int


def _cute_decode_warmup_shape(
    vllm_config: VllmConfig, kv_cache_dtype: str
) -> _CuteDecodeWarmupShape | None:
    """What the CuTe decode route will see, or None without a model or page size.

    The page size is the kernel block size the runner picks for this backend,
    so call this once the block size is final: while the model is built it is
    still the 16-token default, which no FlashInfer MLA kernel block divides.
    An FP8 cache comes with an FP8 query (MLA quantizes it for the decode).
    """
    from vllm.v1.worker.utils import select_common_block_size

    block_size = vllm_config.cache_config.block_size
    if vllm_config.model_config is None or not block_size:
        return None
    try:
        page_size = select_common_block_size(block_size, [FlashInferMLABackend])
    except ValueError:
        return None
    if is_quantized_kv_cache(kv_cache_dtype):
        q_dtype = kv_dtype = current_platform.fp8_dtype()
    else:
        q_dtype = kv_dtype = vllm_config.model_config.dtype
    spec = vllm_config.speculative_config
    num_spec = (spec.num_speculative_tokens or 0) if spec is not None else 0
    return _CuteDecodeWarmupShape(
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        page_size=page_size,
        max_q_len=1 + num_spec,
        max_batch=vllm_config.scheduler_config.max_num_seqs,
        max_model_len=vllm_config.model_config.max_model_len,
    )


FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024
FLASHINFER_MLA_LSE_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024

_fi_workspace: torch.Tensor | None = None


def _get_workspace_buffer(return_lse: bool) -> torch.Tensor:
    global _fi_workspace

    buffer_size = (
        FLASHINFER_MLA_LSE_WORKSPACE_BUFFER_SIZE
        if return_lse
        else FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE
    )
    if _fi_workspace is None or _fi_workspace.numel() < buffer_size:
        # FlashInfer's CuteDSL MLA-decode tactic requires an int8 workspace;
        # the trtllm-gen path views it as uint8, so int8 is safe for all backends.
        _fi_workspace = torch.zeros(buffer_size, dtype=torch.int8, device="cuda")
    return _fi_workspace


_fi_multi_ctas_kv_counter: torch.Tensor | None = None


def _get_multi_ctas_kv_counter_buffer(
    min_bytes: int, device: torch.device
) -> torch.Tensor:
    """Persistent, zero-initialized trtllm-gen multi-CTA-KV counter buffer.

    trtllm-gen's multi-CTA-KV MLA decode kernel resets these semaphores to zero
    at the end of every launch, so the buffer only needs zeroing once. Without a
    caller-owned buffer the public entry point re-allocates and re-zeros it on
    every decode step. ``min_bytes`` is sized to the worst-case batch so the
    buffer is allocated once and never reallocated after CUDA-graph capture.
    """
    global _fi_multi_ctas_kv_counter
    if (
        _fi_multi_ctas_kv_counter is None
        or _fi_multi_ctas_kv_counter.numel() < min_bytes
    ):
        _fi_multi_ctas_kv_counter = torch.zeros(
            min_bytes, dtype=torch.uint8, device=device
        )
    return _fi_multi_ctas_kv_counter


@dataclass
class FlashInferMLADecodeMetadata(MLACommonDecodeMetadata):
    query_start_loc: torch.Tensor | None = None
    max_query_len: int = 1


@dataclass
class FlashInferMLAMetadata(MLACommonMetadata[FlashInferMLADecodeMetadata]):
    pass


class FlashInferMLAMetadataBuilder(MLACommonMetadataBuilder[FlashInferMLAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.VARLEN

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, FlashInferMLAMetadata
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
    ) -> FlashInferMLADecodeMetadata:
        # Promised bound, not measured: a capture dummy puts one token/req, so
        # measuring would bake an undersized graph. Ragged only when max_query_len>1.
        return FlashInferMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
            query_start_loc=query_start_loc_device,
            max_query_len=max_query_len,
        )


class FlashInferMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [32, 64]

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3)
        return (0, 1, 2)

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA"

    @staticmethod
    def get_impl_cls() -> type["FlashInferMLAImpl"]:
        return FlashInferMLAImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLAMetadataBuilder"]:
        return FlashInferMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 10

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # FlashInfer MLA kernel requires qk_nope_head_dim in [64, 128, 192]
        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            qk_nope_head_dim = getattr(hf_text_config, "qk_nope_head_dim", 1)
            if qk_nope_head_dim not in [64, 128, 192]:
                return (
                    "FlashInfer MLA kernel requires qk_nope_head_dim "
                    f"in [64, 128, 192], but got {qk_nope_head_dim}"
                )
        return None

    @classmethod
    def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":
        return "HND"


class FlashInferMLAImpl(MLACommonImpl[FlashInferMLAMetadata]):
    can_return_lse_for_decode: bool = True
    # trtllm-gen MLA decode emits LSE in log2 (per flashinfer's own
    # reference at flashinfer/trace/templates/attention.py:81:
    # `logsumexp / log(2.0)`). Override the AttentionImplBase default
    # so MLAAttention's DCP combine branches on the correct base
    # (IS_BASE_E=False uses tl.exp2/tl.log2 natively, avoiding an FP
    # multiply per decode step).
    lse_base_on_e: bool = False

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
                "FlashInferMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferMLAImpl"
            )

        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None
        # Worst-case decode batch for the persistent counter buffer, read here
        # because the config is only in scope during construction.
        vllm_config = get_current_vllm_config()
        sched = vllm_config.scheduler_config
        self._mla_counter_max_batch: int = (
            sched.max_num_batched_tokens or sched.max_num_seqs
        )
        self._mla_counter_bytes: int | None = None
        self._cute_warmup_config: VllmConfig | None = None
        self._cute_warmup: _CuteDecodeWarmupShape | None = None
        if not _trtllm_gen_mla_decode_supports_num_heads(
            num_heads
        ) and current_platform.is_device_capability_family(100):
            # The shape waits for the block size, which is set after the model.
            self._cute_warmup_config = vllm_config
            register_cutedsl_warmup_provider(self)

    def get_cutedsl_warmup_compile_units(self) -> tuple[CuTeDSLCompileUnit, ...]:
        """One unit per decode query length the CuTe route can be asked for.

        CuTeDSL compiles a kernel per maximum query length, reducer tiling and
        split-KV use, and the last two follow the batch and context length, so
        an eager decode or verify step could otherwise JIT while it runs. Each
        unit runs the length through the same public call as `forward_mqa` at
        the batches and contexts that reach every variant.
        """
        assert self._cute_warmup_config is not None
        shape = _cute_decode_warmup_shape(self._cute_warmup_config, self.kv_cache_dtype)
        if shape is None:
            logger.warning_once(
                "No CuTe MLA decode warmup for block size %s; decode steps "
                "will compile their kernels.",
                self._cute_warmup_config.cache_config.block_size,
            )
            return ()
        self._cute_warmup = shape
        return tuple(
            CuTeDSLCompileUnit(
                name="flashinfer_mla_cute_decode",
                key=("flashinfer_mla_cute_decode", self.num_heads, shape, q_len),
                compile=functools.partial(self._warm_cute_decode, q_len),
            )
            for q_len in range(1, shape.max_q_len + 1)
        )

    def _warm_cute_decode(self, q_len: int) -> None:
        try:
            self._run_cute_decode(q_len)
        except Exception:
            # Not fatal: the first step that needs the kernel compiles it.
            logger.warning(
                "CuTe MLA decode warmup failed at q_len=%d.", q_len, exc_info=True
            )

    def _run_cute_decode(self, q_len: int) -> None:
        shape = self._cute_warmup
        assert shape is not None
        device = torch.device("cuda", torch.accelerator.current_device_index())
        width = self.kv_lora_rank + self.qk_rope_head_dim
        kv_cache = torch.zeros(
            (1, 1, shape.page_size, width), dtype=shape.kv_dtype, device=device
        )
        pages = cute_decode_page_table_width(shape.max_model_len, shape.page_size)
        calls = cute_decode_warmup_calls(
            q_len,
            self.num_heads,
            get_device_sm_count(device),
            shape.max_batch,
            shape.max_model_len,
        )
        for batch, seq_len in calls:
            kwargs: dict[str, Any] = {}
            if q_len > 1:
                ends = [0, *range(q_len, q_len + batch)]
                kwargs["cum_seq_lens_q"] = torch.tensor(
                    ends, dtype=torch.int32, device=device
                )
                kwargs["max_q_len"] = q_len
                query_shape: tuple[int, ...] = (ends[-1], self.num_heads, width)
            else:
                query_shape = (batch, 1, self.num_heads, width)
            trtllm_batch_decode_with_kv_cache_mla(
                query=torch.zeros(query_shape, dtype=shape.q_dtype, device=device),
                kv_cache=kv_cache,
                workspace_buffer=_get_workspace_buffer(False),
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                block_tables=torch.zeros(
                    (batch, pages), dtype=torch.int32, device=device
                ),
                seq_lens=torch.full(
                    (batch,), seq_len, dtype=torch.int32, device=device
                ),
                max_seq_len=seq_len,
                bmm1_scale=self.scale,
                bmm2_scale=1.0,
                backend="cute-dsl",
                **kwargs,
            )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if isinstance(q, tuple):
            q_nope, q_pe = q
            q = torch.cat([q_nope, q_pe], dim=-1)

        cum_seq_lens_q: torch.Tensor | None = None
        max_q_len: int | None = None
        if attn_metadata.decode.max_query_len > 1 and self.dcp_world_size == 1:
            # Causal spec decode: keep q compact and let the kernel tile each
            # request's length (uniform 1+k, or ragged when draft lengths vary).
            # DCP keeps the uniform reshape below: it needs LSE, which the kernels
            # do not return on the ragged path (flashinfer #3238).
            cum_seq_lens_q = attn_metadata.decode.query_start_loc
            # The promised bound can exceed this step's flat q (a lone undrafted
            # suffix row beside a prefill chunk). FlashInfer >= 0.6.16 accepts
            # that, and CuTeDSL compiles one kernel per bound, so keep it fixed.
            max_q_len = attn_metadata.decode.max_query_len
        # trtllm API requires extra dimension q_len_per_request for MTP
        elif attn_metadata.num_decode_tokens % attn_metadata.num_decodes != 0:
            logger.warning_once(
                """FlashInferMLAImpl got a query of uneven length.
                This usually indicates an issue in batch reordering
                or incorrect setup in dummy_run."""
            )
            q = q.unsqueeze(1)
        else:
            q = q.view(attn_metadata.num_decodes, -1, q.shape[-2], q.shape[-1])

        if self.bmm1_scale is None:
            self.bmm1_scale = self.scale
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float

        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm2_scale *= layer._k_scale_float

        return_lse = self.need_to_return_lse_for_decode
        workspace_buffer = _get_workspace_buffer(return_lse)
        extra_kwargs: dict[str, Any] = {}
        if cum_seq_lens_q is not None:
            # Neither decode backend returns LSE on the ragged path
            # (flashinfer #3238); DCP, the only LSE consumer, took the uniform
            # branch above, so this only guards a future caller wiring the two.
            assert not return_lse, (
                "FlashInferMLA ragged decode cannot return LSE; DCP and adaptive "
                "variable-length decode are mutually exclusive."
            )
            extra_kwargs["cum_seq_lens_q"] = cum_seq_lens_q
            extra_kwargs["max_q_len"] = max_q_len
        # Parallel gathers can change the runtime Q heads from TP-local num_heads.
        runtime_num_heads = q.shape[-2]
        decode_backend = _select_mla_decode_backend(runtime_num_heads)
        if decode_backend:
            if return_lse:
                # lse_base_on_e above is trtllm-gen's base 2; CuTeDSL's is base e.
                raise NotImplementedError(
                    "FlashInferMLA DCP decode with the cute-dsl backend "
                    f"(num_heads={runtime_num_heads}) is not supported."
                )
            # CuTeDSL compiles one kernel per max_q_len, split-KV use and reducer
            # tiling, so a step outside a full cudagraph can JIT-compile.
            extra_kwargs["backend"] = decode_backend
        elif kv_c_and_k_pe_cache.shape[-2] in (32, 64):
            # Only where auto can dispatch to trtllm-gen (block size 32 or 64);
            # FlashInfer rejects the buffer when only CuTeDSL can run.
            if self._mla_counter_bytes is None:
                self._mla_counter_bytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(
                    self._mla_counter_max_batch,
                    runtime_num_heads,
                    get_device_sm_count(q.device),
                )
            extra_kwargs["multi_ctas_kv_counter_buffer"] = (
                _get_multi_ctas_kv_counter_buffer(self._mla_counter_bytes, q.device)
            )
        kernel_out = trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv_c_and_k_pe_cache.unsqueeze(1),
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=attn_metadata.decode.block_table,
            seq_lens=attn_metadata.decode.seq_lens,
            max_seq_len=attn_metadata.max_seq_len,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            return_lse=return_lse,
            **extra_kwargs,
        )
        if return_lse:
            o, lse = kernel_out
        else:
            o, lse = kernel_out, None

        # Flatten the output for consistent shape
        o = o.view(-1, o.shape[-2], o.shape[-1])

        return o, lse

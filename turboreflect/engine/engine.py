# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""Engine construction, and the KV pool an engine reports right after startup."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from turboreflect.config import (
    ADAPTIVE_DRAFT_SCHEDULERS,
    MULTIMODAL_FIELDS,
    UNQUANTIZED_CACHE_DTYPES,
    RunConfig,
)
from turboreflect.engine.kv_layout import (
    bytes_per_token,
    geometry_from_model_config,
)
from turboreflect.locality_pd.config import (
    P_SUFFIX_LAYOUT_ONLY,
    ROLE_P,
    PDSettings,
)
from turboreflect.locality_pd.message import EngineFacts
from turboreflect.metrics.trace import TRACE_DIR_ENV
from vllm.config.kv_transfer import KVTransferConfig
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.engine.async_llm import AsyncLLM

_GIB = 1024**3
# The engine core's `get_kv_pool_layout`: its pool's blocks per sequence.
KV_POOL_LAYOUT_UTILITY = "get_kv_pool_layout"

# Pinned by every P/D engine: two that disagree fail with a misleading
# "Heterogeneous TP expects same kv_cache_layout".
_KV_CACHE_LAYOUT_ENV = "VLLM_KV_CACHE_LAYOUT"

_MODEL_RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"
# FlashInfer's sampler stays off unless set: its warm-up JIT-builds kernels, which
# fails without nvcc.
_FLASHINFER_SAMPLER_ENV = "VLLM_USE_FLASHINFER_SAMPLER"

# The engine trace's scheduler for each scheduler a run may name.
_TRACED_SCHEDULERS = {
    None: "turboreflect.metrics.trace.SpecTraceScheduler",
    "turboreflect.context_reuse.scheduler.ReserveOnArrivalScheduler": (
        "turboreflect.metrics.trace.ReserveOnArrivalTraceScheduler"
    ),
    "turboreflect.context_reuse.scheduler.ContextReuseScheduler": (
        "turboreflect.metrics.trace.ContextReuseTraceScheduler"
    ),
}


class EngineError(Exception):
    """The engine cannot run this configuration."""


def _select_model_runner(cfg: RunConfig) -> None:
    """Use V1 when suffix decoding is enabled; vLLM V2 rejects it."""
    if cfg.suffix_decoding:
        os.environ[_MODEL_RUNNER_ENV] = "0"


@dataclass(frozen=True)
class KvPoolReport:
    """The profiled KV pool, in tokens and bytes.

    `capacity_tokens` counts one token group's blocks; each sequence also holds
    `fixed_blocks_per_sequence` state blocks over `state_groups` Mamba groups.
    """

    block_size: int
    num_gpu_blocks: int
    capacity_tokens: int
    bytes_per_token: int
    effective_max_model_len: int
    cache_dtype: str
    model_dtype: str
    token_groups: int = 1
    state_groups: int = 0
    fixed_blocks_per_sequence: int = 0
    mamba_cache_mode: str = "none"

    @property
    def bytes_per_full_sequence(self) -> int:
        return self.bytes_per_token * self.effective_max_model_len

    @property
    def max_full_sequences(self) -> int:
        return self.sequences_fitting(self.effective_max_model_len)

    def sequences_fitting(self, tokens: int, reuses_context: bool = False) -> int:
        """Sequences of `tokens` tokens the pool holds at once.

        `reuses_context` adds the two boundary state blocks per Mamba group kept for
        the next stage.
        """
        per = -(-tokens // self.block_size)
        fixed = self.fixed_blocks_per_sequence
        if reuses_context:
            fixed += 2 * self.state_groups
        return (self.capacity_tokens // self.block_size) // max(
            1, per + -(-fixed // self.token_groups)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_size": self.block_size,
            "num_gpu_blocks": self.num_gpu_blocks,
            "capacity_tokens": self.capacity_tokens,
            "kv_bytes_per_token": self.bytes_per_token,
            "kv_gib_per_full_sequence": self.bytes_per_full_sequence / _GIB,
            "kv_pool_gib": self.capacity_tokens * self.bytes_per_token / _GIB,
            "max_concurrent_full_sequences": self.max_full_sequences,
            "token_groups": self.token_groups,
            "state_groups": self.state_groups,
            "fixed_blocks_per_sequence": self.fixed_blocks_per_sequence,
            "mamba_cache_mode": self.mamba_cache_mode,
            "max_model_len": self.effective_max_model_len,
            "kv_cache_dtype": self.cache_dtype,
            "model_dtype": self.model_dtype,
        }

    def lines(self) -> list[str]:
        d = self.as_dict()
        return [
            (
                f"KV pool     : {self.num_gpu_blocks} blocks x {self.block_size} "
                f"= {self.capacity_tokens} tokens ({d['kv_pool_gib']:.1f} GiB)"
            ),
            f"KV dtype    : {self.cache_dtype} (model {self.model_dtype})",
            f"KV per token: {self.bytes_per_token / 1024:.0f} KiB",
            (
                f"KV per {self.effective_max_model_len}-token sequence: "
                f"{d['kv_gib_per_full_sequence']:.1f} GiB"
            ),
            f"Fits        : {self.max_full_sequences} concurrent full-length sequences",
        ]


def _kv_bytes_per_token(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    cache_dtype: str,
) -> int:
    """Bytes of KV per token over this GPU's layers, in the layout the kernels write.

    `engine.kv_layout` owns the arithmetic; this adds the engine's geometry and dtype.
    """
    geo = geometry_from_model_config(model_config, parallel_config)
    native_itemsize = model_config.dtype.itemsize
    if cache_dtype in UNQUANTIZED_CACHE_DTYPES:
        native_itemsize = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype].itemsize
        cache_dtype = "auto"
    return bytes_per_token(
        geo,
        cache_dtype,
        native_itemsize=native_itemsize,
    )


def gpu_memory_note() -> str:
    """Free and total device memory, just before the engine starts.

    vLLM budgets a fraction of total memory, so another process's share shrinks the KV
    pool; this log line shows that cause.
    """
    import torch

    free, total = torch.accelerator.get_memory_info()
    return f"GPU memory  : {free / _GIB:.1f} GiB free of {total / _GIB:.1f} GiB"


def _attention_args(cfg: RunConfig) -> dict[str, Any]:
    """The run's attention pins; vLLM reads no environment variable for the backend."""
    attention: dict[str, Any] = {}
    if cfg.attention_backend is not None:
        attention["backend"] = cfg.attention_backend
    if cfg.flash_attn_version is not None:
        attention["flash_attn_version"] = cfg.flash_attn_version
    return {"attention_config": attention} if attention else {}


def _suffix_config(cfg: RunConfig, role: str | None = None) -> dict[str, Any] | None:
    """The engine's `speculative_config`, or None when it drafts nothing.

    A P engine drafts only under `RunConfig.pd_p_speculates`. Otherwise a P of a model
    with recurrent state still takes D's draft count, which sizes its Mamba states for
    NIXL's slot-for-slot transfer, but proposes nothing (`P_SUFFIX_LAYOUT_ONLY`).
    """
    layout_only = role == ROLE_P and not cfg.pd_p_speculates()
    if not cfg.suffix_decoding or (layout_only and not cfg.has_recurrent_state()):
        return None
    config = {
        "method": "suffix",
        "num_speculative_tokens": cfg.suffix_num_speculative_tokens,
        "suffix_decoding_max_tree_depth": cfg.suffix_max_tree_depth,
        "suffix_decoding_max_cached_requests": cfg.suffix_max_cached_requests,
    }
    if layout_only:
        config.update(P_SUFFIX_LAYOUT_ONLY)
    return config


def _parallel_args(cfg: RunConfig) -> dict[str, Any]:
    if cfg.tensor_parallel_size is None:
        return {}
    return {"tensor_parallel_size": cfg.tensor_parallel_size}


def _memory_args(cfg: RunConfig) -> dict[str, Any]:
    if cfg.gpu_memory_utilization is None:
        return {}
    return {"gpu_memory_utilization": cfg.gpu_memory_utilization}


def _model_args(cfg: RunConfig) -> dict[str, Any]:
    model_args: dict[str, Any] = {}
    if cfg.trust_remote_code is not None:
        model_args["trust_remote_code"] = cfg.trust_remote_code
    if cfg.enforce_eager is not None:
        model_args["enforce_eager"] = cfg.enforce_eager
    if cfg.language_model_only is not None:
        model_args["language_model_only"] = cfg.language_model_only
    for name in MULTIMODAL_FIELDS:
        if getattr(cfg, name) is not None:
            model_args[name] = getattr(cfg, name)
    return model_args


def _kernel_args(cfg: RunConfig) -> dict[str, Any]:
    return {
        name: getattr(cfg, name)
        for name in ("moe_backend", "gdn_prefill_backend")
        if getattr(cfg, name) is not None
    }


def _scheduler_args(cfg: RunConfig) -> dict[str, Any]:
    """vLLM's admission gate (`RunConfig.scheduler_reserve_full_isl`), when stated."""
    if cfg.scheduler_reserve_full_isl is None:
        return {}
    return {"scheduler_reserve_full_isl": cfg.scheduler_reserve_full_isl}


def _compilation_args(cfg: RunConfig) -> dict[str, Any]:
    """The cudagraph mode, named only when the config names it.

    `compilation_config` feeds `VllmConfig.compute_hash()`, the torch.compile cache key,
    so passing vLLM's default would still pay a cold compile.
    """
    if cfg.cudagraph_mode is None:
        return {}
    return {"compilation_config": {"cudagraph_mode": cfg.cudagraph_mode}}


def _finalize_engine_args(
    cfg: RunConfig,
    engine_args: AsyncEngineArgs,
    scheduler_cls: str | None,
    name: str | None = None,
    worker_scheduler_cls: str | None = None,
) -> AsyncEngineArgs:
    """Finish a P/D engine's arguments as `build_engine_args` finishes one engine's.

    Args:
        engine_args: Completed in place and returned.
        scheduler_cls: The scheduler the engine trace wraps.
        name: Named in errors, so a refusal says which worker raised it.
        worker_scheduler_cls: The traced scheduler a P/D method runs in place of
            `scheduler_cls`, which adaptive draft length is then mixed into.
    """
    side = "" if name is None else f"{name}: "
    os.environ.setdefault(_FLASHINFER_SAMPLER_ENV, "0")
    if cfg.async_scheduling is not False:
        raise EngineError(
            f"{side}the engine trace requires async_scheduling=false; this run "
            f"has {cfg.async_scheduling}"
        )
    engine_args.scheduler_cls = _TRACED_SCHEDULERS[scheduler_cls]
    if worker_scheduler_cls is not None:
        engine_args.scheduler_cls = scheduler_cls = worker_scheduler_cls
    if cfg.adaptive_draft is not None and engine_args.speculative_config:
        _apply_adaptive_draft(cfg, engine_args, scheduler_cls, side)
    return engine_args


def _apply_adaptive_draft(
    cfg: RunConfig, engine_args: AsyncEngineArgs, scheduler_cls: str | None, side: str
) -> None:
    """Swap in the adaptive-draft-length scheduler, set up via `additional_config`."""
    traced = ADAPTIVE_DRAFT_SCHEDULERS.get(scheduler_cls, scheduler_cls)
    if traced not in ADAPTIVE_DRAFT_SCHEDULERS.values():
        raise EngineError(
            f"{side}adaptive draft length is not mixed into {scheduler_cls}; expected "
            f"one of {list(ADAPTIVE_DRAFT_SCHEDULERS)}"
        )
    engine_args.scheduler_cls = traced
    engine_args.additional_config = {
        **(engine_args.additional_config or {}),
        "adaptive_draft": dataclasses.asdict(cfg.adaptive_draft),
    }


def _start_engine(
    engine_args: AsyncEngineArgs, trace_dir: Path, stat_loggers: list[Any] | None
) -> AsyncLLM:
    """Start an engine; its core process inherits `trace_dir` via the environment."""
    os.environ[TRACE_DIR_ENV] = str(trace_dir)
    return AsyncLLM.from_engine_args(
        engine_args,
        usage_context=UsageContext.ENGINE_CONTEXT,
        stat_loggers=stat_loggers,
    )


def build_engine_args(cfg: RunConfig) -> AsyncEngineArgs:
    """Build one model instance's arguments, ready for `AsyncLLM`.

    Raises:
        EngineError: If scheduling may be asynchronous, which the engine trace cannot
            follow, or if adaptive draft length has no form of the scheduler.
    """
    _select_model_runner(cfg)
    # Named only when the config names it (`_compilation_args`).
    compilation = (
        {"compilation_config": {"cudagraph_mode": cfg.cudagraph_mode}}
        if cfg.cudagraph_mode is not None
        else {}
    )
    engine_args = AsyncEngineArgs(
        model=cfg.model,
        max_model_len=cfg.max_model_len,
        enable_prefix_caching=cfg.enable_prefix_caching,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        max_num_seqs=cfg.max_num_seqs,
        scheduling_policy=cfg.scheduling_policy,
        scheduler_cls=cfg.scheduler_cls,
        async_scheduling=cfg.async_scheduling,
        hf_overrides=cfg.hf_overrides(),
        speculative_config=_suffix_config(cfg),
        # Per-stage numbers come off RequestStateStats, populated only with stats on.
        disable_log_stats=False,
        seed=cfg.seed,
        **_memory_args(cfg),
        **_model_args(cfg),
        **_parallel_args(cfg),
        kv_cache_dtype=cfg.kv_cache_dtype,
        **_attention_args(cfg),
        **_kernel_args(cfg),
        **_scheduler_args(cfg),
        **compilation,
    )
    # The engine trace (`turboreflect.metrics.trace`) is always on.
    if cfg.async_scheduling is not False:
        raise EngineError(
            "the engine trace requires async_scheduling=false; this run has "
            f"{cfg.async_scheduling}"
        )
    engine_args.scheduler_cls = _TRACED_SCHEDULERS[cfg.scheduler_cls]
    if cfg.adaptive_draft is not None and engine_args.speculative_config:
        _apply_adaptive_draft(cfg, engine_args, cfg.scheduler_cls, "")
    return engine_args


def build_engine(
    cfg: RunConfig, trace_dir: Path, stat_loggers: list[Any] | None = None
) -> AsyncLLM:
    """Bring up an AsyncLLM for `cfg`, or fail before submission."""
    os.environ.setdefault(_FLASHINFER_SAMPLER_ENV, "0")
    os.environ[TRACE_DIR_ENV] = str(trace_dir)
    engine_args = build_engine_args(cfg)
    return AsyncLLM.from_engine_args(
        engine_args,
        usage_context=UsageContext.ENGINE_CONTEXT,
        stat_loggers=stat_loggers,
    )


async def kv_pool_report(llm: AsyncLLM) -> KvPoolReport:
    """Read back the profiled KV pool and how the engine core spends its blocks."""
    vllm_config = llm.vllm_config
    cache_config = vllm_config.cache_config
    num_gpu_blocks = cache_config.num_gpu_blocks
    layout = await llm.engine_core.call_utility_async(KV_POOL_LAYOUT_UTILITY)
    return KvPoolReport(
        block_size=cache_config.block_size,
        num_gpu_blocks=num_gpu_blocks,
        capacity_tokens=num_gpu_blocks
        // layout["token_groups"]
        * cache_config.block_size,
        token_groups=layout["token_groups"],
        state_groups=layout["state_groups"],
        fixed_blocks_per_sequence=layout["fixed_blocks_per_sequence"],
        mamba_cache_mode=str(getattr(cache_config, "mamba_cache_mode", "none")),
        bytes_per_token=_kv_bytes_per_token(
            vllm_config.model_config,
            vllm_config.parallel_config,
            cache_config.cache_dtype,
        ),
        effective_max_model_len=vllm_config.model_config.max_model_len,
        cache_dtype=str(cache_config.cache_dtype),
        model_dtype=str(vllm_config.model_config.dtype).removeprefix("torch."),
    )


# vLLM's registered name for the pull-mode connector.
KV_CONNECTOR = "NixlConnector"

# Stated: under `recompute` D recomputes blocks it could not load, hiding a failed
# transfer inside D's timing.
KV_LOAD_FAILURE_POLICY = "fail"

# P only pulls D's held history and declines a pull once D's hold is about to lapse;
# failing that request would cancel the pass, so P recomputes and reports only what
# loaded as external.
PULL_LOAD_FAILURE_POLICY = "recompute"

# `kv_connector_extra_config` readers supply their own defaults and log no miss, so a
# misspelt key silently changes the lease or stops pulls.
KV_LEASE_DURATION_KEY = "kv_lease_duration"
BIDIRECTIONAL_KEY = "bidirectional_kv_xfer"
DECODER_TTL_KEY = "decoder_kv_blocks_ttl"
RECOMPUTE_THRESHOLD_KEY = "kv_recompute_threshold"


def load_failure_policy(role: str) -> str:
    """The `kv_load_failure_policy` an engine of `role` is built with."""
    if role == ROLE_P:
        return PULL_LOAD_FAILURE_POLICY
    return KV_LOAD_FAILURE_POLICY


def lease_extra_config(settings: PDSettings) -> dict[str, int]:
    """`kv_connector_extra_config` for either side, holding just the lease."""
    return {KV_LEASE_DURATION_KEY: int(settings.kv_lease_duration_s)}


def pull_extra_config(settings: PDSettings) -> dict[str, Any]:
    """The `kv_connector_extra_config` keys of a pull from D.

    Every engine gets them: D holds and exports blocks, P pulls them, and a side
    without them leaves the other waiting.
    """
    return {
        BIDIRECTIONAL_KEY: True,
        DECODER_TTL_KEY: settings.decoder_kv_blocks_ttl_s,
        RECOMPUTE_THRESHOLD_KEY: settings.kv_recompute_threshold,
    }


def pd_engine_args(settings: PDSettings, name: str) -> dict[str, Any]:
    """The `AsyncEngineArgs` fields one P/D engine does not share with the rest."""
    return {
        "kv_transfer_config": KVTransferConfig(
            kv_connector=KV_CONNECTOR,
            kv_role=settings.kv_role(name),
            kv_buffer_device="cuda",
            kv_load_failure_policy=load_failure_policy(settings.role(name)),
            kv_connector_extra_config={
                **lease_extra_config(settings),
                **pull_extra_config(settings),
            },
        ),
        "enable_prefix_caching": settings.prefix_caching(name),
        "max_num_seqs": settings.engine(name).max_num_seqs,
        "scheduling_policy": settings.scheduling_policy(name),
    }


def build_pd_engine_args(
    cfg: RunConfig,
    settings: PDSettings,
    name: str,
) -> AsyncEngineArgs:
    """Build one P/D engine's arguments; raises as `build_engine_args` does."""
    _select_model_runner(cfg)
    os.environ[_KV_CACHE_LAYOUT_ENV] = settings.kv_cache_layout
    role = settings.role(name)

    engine_args = AsyncEngineArgs(
        model=cfg.model,
        max_model_len=cfg.max_model_len,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        async_scheduling=cfg.async_scheduling,
        hf_overrides=cfg.hf_overrides(),
        speculative_config=_suffix_config(cfg, role),
        disable_log_stats=False,
        seed=cfg.seed,
        **_memory_args(cfg),
        **_model_args(cfg),
        **_parallel_args(cfg),
        kv_cache_dtype=cfg.kv_cache_dtype,
        **_attention_args(cfg),
        **_kernel_args(cfg),
        **_scheduler_args(cfg),
        **_compilation_args(cfg),
        **pd_engine_args(settings, name),
    )
    engine_args = _finalize_engine_args(
        cfg, engine_args, None, name, cfg.pd_method.scheduler_cls()
    )
    engine_args.additional_config = {
        **(engine_args.additional_config or {}),
        "locality_pd": cfg.pd_method.engine_extra_config(settings, name),
    }
    return engine_args


def build_pd_engine(
    cfg: RunConfig,
    settings: PDSettings,
    name: str,
    trace_dir: Path,
    stat_loggers: list[Any] | None = None,
) -> AsyncLLM:
    """Bring up one engine of a P/D run; raises as `build_engine_args` does.

    Args:
        trace_dir: This engine's own; trace file names are fixed, so two engines
            sharing one would interleave rows.
        stat_loggers: This engine's own `StatLoggerFactory` list, as `StepLogger` is
            stateful.
    """
    engine_args = build_pd_engine_args(cfg, settings, name)
    return _start_engine(engine_args, trace_dir, stat_loggers)


def collect_engine_facts(llm: AsyncLLM) -> EngineFacts:
    """What the driver reads of the engine `build_pd_engine` returned."""
    return EngineFacts(block_size=llm.vllm_config.cache_config.block_size)

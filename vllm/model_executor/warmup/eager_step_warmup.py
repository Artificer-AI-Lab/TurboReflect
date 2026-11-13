# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up the kernel variants only steps run outside a CUDA graph reach.

Capture compiles what a graph replays; an eager step -- one larger than every
captured size, or a mixed step outside a piecewise graph -- can still compile a
Triton variant mid-serving, because Triton specializes integer arguments on 1
and on divisibility by 16 and some kernels take strides as constexprs.
"""

import functools
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import CUDAGraphMode
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

_DIVISIBILITY = 16
# Batch sizes reaching the 1, plain and multiple-of-16 specializations.
_BATCH_SIZES = (1, 2, _DIVISIBILITY)


def eager_from(vllm_config: Any) -> int:
    """The smallest step that can run outside a CUDA graph.

    Only a piecewise graph takes a mixed step, so without one every size can.
    """
    compilation = vllm_config.compilation_config
    mode = compilation.cudagraph_mode
    piecewise = mode is not None and mode.name in ("PIECEWISE", "FULL_AND_PIECEWISE")
    if vllm_config.model_config.enforce_eager or not piecewise:
        return 1
    return (compilation.max_cudagraph_capture_size or 0) + 1


def moe_warmup_token_counts(
    config_of: Callable[[int], dict[str, Any]],
    top_k: int,
    num_experts: int,
    first: int,
    last: int,
) -> list[int]:
    """Step sizes in `[first, last]` reaching every `fused_moe_kernel` variant.

    Within a range of M where the tile config and the kernel's M thresholds
    are fixed, the variant depends only on how its integer arguments divide,
    and those are affine in M, so the top 16 sizes of the range show all of
    them.
    """

    def key(tokens: int) -> tuple:
        config = config_of(tokens)
        slots = tokens * top_k
        return (
            tuple(sorted(config.items())),
            tokens < config["BLOCK_SIZE_M"],
            slots < config["BLOCK_SIZE_M"],
            slots * 4 <= num_experts,
            slots < num_experts,
        )

    ranges: list[list[Any]] = []
    for tokens in range(1, last + 1):
        current = key(tokens)
        if ranges and ranges[-1][2] == current:
            ranges[-1][1] = tokens
        else:
            ranges.append([tokens, tokens, current])
    counts: list[int] = []
    for start, end, _ in ranges:
        if end >= first:
            counts.extend(range(max(start, end - _DIVISIBILITY + 1, first), end + 1))
    return counts


def _triton_moe_config_fns(
    model: torch.nn.Module, dtype: torch.dtype
) -> list[tuple[Callable[[int], dict[str, Any]], int, int]]:
    """`(config of M, top_k, experts)` per distinct Triton-expert MoE shape."""
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
        TritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        try_get_optimal_moe_config,
    )

    problems = {}
    for module in model.modules():
        kernel = getattr(getattr(module, "quant_method", None), "moe_kernel", None)
        experts = getattr(getattr(kernel, "impl", None), "fused_experts", None)
        if not isinstance(experts, TritonExperts):
            continue
        block_shape = experts.block_shape
        key = (
            tuple(module.w13_weight.size()),
            tuple(module.w2_weight.size()),
            module.top_k,
            module.global_num_experts,
            experts.quant_config.config_name(dtype),
            None if block_shape is None else tuple(block_shape),
        )
        problems[key] = block_shape
    return [
        (
            functools.partial(
                try_get_optimal_moe_config, w1, w2, top_k, name, block_shape=block
            ),
            top_k,
            num_experts,
        )
        for (w1, w2, top_k, num_experts, name, _), block in problems.items()
    ]


@torch.inference_mode()
def eager_moe_warmup(runner: "GPUModelRunner") -> None:
    """Run every eager step size that reaches a new `fused_moe_kernel` variant.

    A chunked-prefill pass compiles them at step sizes no capture visited.
    """
    first = eager_from(runner.vllm_config)
    scheduler = runner.scheduler_config
    last = min(
        scheduler.max_num_scheduled_tokens or scheduler.max_num_batched_tokens,
        runner.max_num_tokens,
    )
    counts: set[int] = set()
    for config_of, top_k, num_experts in _triton_moe_config_fns(
        runner.get_model(), runner.model_config.dtype
    ):
        counts.update(
            moe_warmup_token_counts(config_of, top_k, num_experts, first, last)
        )
    if not counts:
        return
    started = time.perf_counter()
    for num_tokens in sorted(counts):
        runner._dummy_run(
            num_tokens, cudagraph_runtime_mode=CUDAGraphMode.NONE, skip_eplb=True
        )
    logger.info(
        "Warmed up fused MoE on %d eager step sizes in %.1fs.",
        len(counts),
        time.perf_counter() - started,
    )


def _gdn_builder(runner: "GPUModelRunner") -> Any:
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

    return next(
        (
            builder
            for groups in runner.attn_groups
            for group in groups
            for builder in group.metadata_builders
            if isinstance(builder, GDNAttentionMetadataBuilder)
        ),
        None,
    )


def uncompacted_decode_sizes(
    builder: Any, query_len: int, max_reqs: int, max_tokens: int
) -> list[int]:
    """Uniform decode batch sizes whose state indices GDN leaves uncompacted.

    A full-graph builder copies decode batches of up to
    `decode_cudagraph_max_bs` tokens into its capture buffers; larger batches,
    and every batch without full graphs, run eager on the gathered table.
    """
    if builder.use_full_cuda_graph:
        first = builder.decode_cudagraph_max_bs // query_len + 1
        candidates = {first, -(-first // _DIVISIBILITY) * _DIVISIBILITY}
    else:
        candidates = set(_BATCH_SIZES)
    return sorted(
        n for n in candidates if n <= max_reqs and n * query_len <= max_tokens
    )


def _warm_mamba_align_copies(runner: "GPUModelRunner", sizes: list[int]) -> None:
    """Compile the align-mode state copies at every batch specialization.

    The inputs make both kernels return before copying anything; the bound
    layout is dropped afterwards so the first real step binds its own.
    """
    if runner.cache_config.mamba_cache_mode != "align":
        return
    ctx = runner._get_mamba_bufs().postprocess_align
    if ctx is None:
        return
    ctx.initialize_from_forward_context(
        runner.kv_cache_config,
        runner.compilation_config.static_forward_context,
        runner.model.get_mamba_state_copy_func(),
        [
            runner.input_batch.block_table[gid].get_device_tensor(runner.max_num_reqs)
            for gid in ctx.mamba_group_ids
        ],
    )
    try:
        for n in sizes:
            # A fresh source column skips the precopy; with one block's worth
            # computed, the step's state stays in its block.
            fresh, zero, one = (
                torch.full((n,), value, dtype=torch.int32, device=runner.device)
                for value in (-1, 0, 1)
            )
            block = torch.full_like(zero, ctx.block_size)
            ctx.run_fused_precopy(n, zero, fresh, zero, None)
            ctx.run_fused_postprocess(n, one, zero, one, block, zero)
    finally:
        ctx.is_initialized = False


@torch.inference_mode()
def eager_decode_warmup(runner: "GPUModelRunner") -> None:
    """Run the eager GDN steps through the runner's own metadata builders.

    In align mode GDN gathers each request's `1 + num_spec` state blocks into
    one table, and plain (non-speculative) rows index its first column, whose
    row stride is a constexpr of the conv and recurrent kernels. Only a step of
    plain rows alone keeps that strided column; beside a speculative row they
    get a stride-1 copy. A speculative engine counts every decode, with drafts
    or without, as a speculative row, so it serves both kinds of mixed step.
    """
    builder = _gdn_builder(runner)
    if builder is None:
        return
    num_spec = builder.num_spec
    # The draft count of each decode row picks its path: -1 is plain.
    mixed_drafts = (-1, 0) if num_spec else (-1,)
    uniform = ((1, 0), (1 + num_spec, num_spec)) if num_spec else ((1, -1),)
    steps = [
        (2 * n + 1, n, draft, {"create_mixed_batch": True})
        for draft in mixed_drafts
        for n in _BATCH_SIZES
        if n < runner.max_num_reqs
    ] + [
        (
            n * query_len,
            n,
            draft,
            {"uniform_decode": True, "uniform_query_len": query_len},
        )
        for query_len, draft in uniform
        for n in uncompacted_decode_sizes(
            builder, query_len, runner.max_num_reqs, runner.max_num_tokens
        )
    ]
    drafts = runner.num_decode_draft_tokens
    saved = drafts.np.copy()
    try:
        for num_tokens, num_decodes, draft, layout in steps:
            drafts.np[:] = -1
            drafts.np[:num_decodes] = draft
            runner._dummy_run(
                num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                force_attention=True,
                skip_eplb=True,
                **layout,
            )
    finally:
        drafts.np[:] = saved
    if num_spec:
        _warm_mamba_align_copies(
            runner, [n for n in _BATCH_SIZES if n <= runner.max_num_reqs]
        )

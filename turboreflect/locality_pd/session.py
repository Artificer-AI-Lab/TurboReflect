# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""The run loop for a split run, beside `engine.runner`'s single-engine loop.

This process must not import vLLM: spawn re-runs the entry module in every worker,
which would then attach to the parent's devices and open its NIXL side channel on
another engine's port. Rules shared with `program.pipeline` are imported inside the
function that applies them, which runs once every engine is serving.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from turboreflect.config import (
    PIPELINE_MODE_ROUNDS,
    RunConfig,
    strips_stray_reasoning,
)
from turboreflect.context_reuse.admission import StageSubmit
from turboreflect.engine.runner import (
    DriverError,
    merge_section,
    program_jobs,
    warm_up_concurrent,
    warmup_concurrency,
    warmup_program_id,
)
from turboreflect.locality_pd import message
from turboreflect.locality_pd.admission import (
    StageBound,
    StageShape,
    peak_blocks,
    stage_bounds,
)
from turboreflect.locality_pd.cache_hits import Ledger, lcp
from turboreflect.locality_pd.config import (
    ROLE_D,
    ROLE_P,
    PDSettings,
)
from turboreflect.locality_pd.dispatch import (
    GENERATION,
    LATER_STAGE,
    LOCAL,
    REMOTE_HOME,
    WorkItem,
)
from turboreflect.locality_pd.driver import PDDriver
from turboreflect.locality_pd.policy import ANY_WORKER, PrefillTask
from turboreflect.locality_pd.runtime import LocalityPDSettings, PolicyRuntime
from turboreflect.locality_pd.worker import trace_dir
from turboreflect.metrics.metrics import StageMetrics
from turboreflect.metrics.report import write_metrics_safely
from turboreflect.metrics.results import (
    PassResult,
    ProgramResult,
    ResultWriter,
    final_artifact,
)
from turboreflect.program.media import (
    ClipLayout,
    Expansion,
    MediaError,
    bind,
    clip_ref,
    layouts_needed,
)
from turboreflect.program.prompt import (
    JOIN_REQUOTE,
    JOIN_TOKENS,
    Rendered,
    join_record,
    render_successor,
)
from turboreflect.program.rounds import (
    PlannedStage,
    StagePlan,
    next_inputs,
    record_rounds,
    stage_plan,
)
from turboreflect.program.stages import merged_parts
from turboreflect.program.task import Task, load_task

_BAR = "=" * 78

# Warmup transfers per pair and direction; the first pays the NIXL handshake and
# the first transfer plan.
PAIR_ROUNDS = 2
# D's tokens for each: a first token, and ids a later stage renders on.
PAIR_GEN_TOKENS = 2


def split_sampling(
    cfg: RunConfig, decode_path: str, program_index: int = 0
) -> dict[str, Any]:
    """The decode engine's sampling, as the plain dict the driver sends.

    Draws as `program.pipeline.sampling_params` does for the same path. Only the
    streaming differs: a split run times D's first token, so both paths stream as
    `delta` and the worker joins the deltas into the completion.
    """
    draw = dict(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        seed=cfg.request_seed(program_index),
    )
    if decode_path == "quality":
        return dict(
            max_tokens=cfg.max_tokens_quality,
            output_kind=message.DELTA,
            detokenize=True,
            **draw,
        )
    if decode_path == "efficiency":
        return dict(
            max_tokens=cfg.gen_tokens,
            ignore_eos=True,
            output_kind=message.DELTA,
            detokenize=True,
            **draw,
        )
    raise DriverError(f"unknown decode path {decode_path!r}")


def clip_layout(engines: Any, task: Task, cfg: RunConfig) -> ClipLayout | None:
    """Clip layout for `task` when prompts are placed by position, else None."""
    if not task.videos or not layouts_needed(cfg):
        return None
    return engines.clip_layouts.get(tuple(task.videos))


def engine_view(rendered: Rendered, expansion: Expansion | None) -> Rendered:
    """`rendered` with each clip expanded as at the generation stage."""
    if expansion is None:
        return rendered
    return dataclasses.replace(
        rendered, token_ids=expansion.engine_ids(rendered.token_ids)
    )


def check_pull_prompt(
    stage: str,
    rendered: Rendered,
    history: Sequence[int],
    params: Any,
    *,
    text_join_pull: bool = False,
) -> str | None:
    """Why P cannot pull D's blocks under this prompt, or None when it can.

    vLLM checks no hash on a pull from D: P loads D's KV for the first
    `remote_num_tokens` positions of whatever prompt it holds, so the prompt must
    open with those positions of `history`. A token-id join does; a requote, or a
    text join under `pd.text_join_pull`, does once cut by `truncated_pull`. P
    prefills a refused prompt whole.

    Args:
        history: The previous stage's prompt and the ids D generated for it.
        params: What P would be handed of what that stage's D returned.
    """
    if not (rendered.join == JOIN_TOKENS or _cut_pull(rendered, text_join_pull)):
        return (
            f"{stage}: joined to the stage before as {rendered.join!r}, not "
            "token ids, and a pull from D checks no hash"
        )
    offered = params.get("remote_num_tokens") or 0
    if list(rendered.token_ids[:offered]) != list(history[:offered]):
        return (
            f"{stage}: D offers KV for {offered} positions, and the prompt does "
            f"not open with the {offered} tokens D computed it for"
        )
    return None


def truncated_pull(
    params: Mapping[str, Any], shared: int, block_size: int
) -> dict[str, Any]:
    """D's params cut to the whole blocks a prompt shares with D's history.

    Cutting the count alone would load KV computed for other tokens, as P pairs its
    pull blocks with the end of each remote group; so each group is cut too, which
    holds for groups that open at the history's first block (full attention; a
    sliding-window group is exported clipped). A hybrid D's exact recurrent states
    (`remote_mamba_states`) stay whole; P loads the latest within the cut
    (`hybrid_pull_end`).

    Returns:
        A copy offering `c = floor(min(shared, offered) / block_size) * block_size`
        tokens and the first `c / block_size` blocks of each group.
    """
    cut = min(shared, params.get("remote_num_tokens") or 0)
    blocks = cut // block_size
    return {
        **params,
        "remote_num_tokens": blocks * block_size,
        "remote_block_ids": [group[:blocks] for group in params["remote_block_ids"]],
    }


def offered_tokens(
    params: Mapping[str, Any] | None,
    rendered: Rendered,
    shared: int,
    block_size: int,
    text_join_pull: bool,
) -> int:
    """What a P could pull of the stage before's hold on D for this prompt.

    0 with no hold or for a text join without `text_join_pull`; else D's offer, cut
    by `truncated_pull` unless joined by token ids, and ending at a hybrid D's
    latest exact recurrent state within it (`offered_end`).
    """
    if not params:
        return 0
    prompt = len(rendered.token_ids)
    if rendered.join == JOIN_TOKENS:
        return offered_end(params, prompt)
    if not _cut_pull(rendered, text_join_pull):
        return 0
    return offered_end(truncated_pull(params, shared, block_size), prompt)


def hybrid_pull_end(params: Mapping[str, Any], prompt_tokens: int, local: int) -> int:
    """Where a hybrid P's pull of a hybrid D's hold ends, given P's local hit.

    A copy of vLLM's `select_mamba_pull_state`, as the driver imports no vLLM. P
    drops the token NIXL leaves the decoder (`_truncate_mamba_request_for_prefill`)
    and keeps one more to compute, so the pull ends at the latest of D's exact
    recurrent states (`remote_mamba_states`) in `(local, min(offer, prompt - 2)]`,
    or at `local` when there is none or the hit is not block-aligned.
    """
    block = params.get("remote_block_size") or 0
    if not block or local % block:
        return local
    limit = min(params.get("remote_num_tokens") or 0, prompt_tokens - 2)
    return max(
        (
            entry[0]
            for entry in params.get("remote_mamba_states") or ()
            if local < entry[0] <= limit
        ),
        default=local,
    )


def offered_end(params: Mapping[str, Any], prompt_tokens: int) -> int:
    """The last position a P with no local hit could pull of `params`."""
    if "remote_mamba_states" not in params:
        return params.get("remote_num_tokens") or 0
    return hybrid_pull_end(params, prompt_tokens, 0)


def pull_recomputed_tokens(out: Any, params: Mapping[str, Any], threshold: int) -> int:
    """The tokens of P's pull from D that P computed again.

    The pull is `min(remote_num_tokens, prompt) - local`, 0 below the recompute
    threshold, or for a hybrid D ends at `hybrid_pull_end`. P counts what it pulled
    as external, so a declined or failed pull leaves `pull - external` above 0.
    """
    external = out.num_external_cached_tokens
    local = out.num_cached_tokens - external
    if "remote_mamba_states" in params:
        pull = hybrid_pull_end(params, out.prompt_len, local) - local
    else:
        pull = min(params.get("remote_num_tokens") or 0, out.prompt_len) - local
    if pull <= 0 or (threshold > 0 and pull < threshold):
        pull = 0
    return pull - external


def decode_state(engines: Any) -> tuple[int, int]:
    """The Ds' largest per-sequence state in ledger blocks: what a sequence holds at
    any length, and what a hold pins per block boundary (one per Mamba group).
    """
    pools = engines.report()
    found = [
        (
            -(-pool.get("fixed_blocks_per_sequence", 0) // groups),
            -(-pool.get("state_groups", 0) // groups),
        )
        for side in pools.values()
        if side.get("role") == ROLE_D
        for pool in [side.get("kv_pool") or {}]
        for groups in [pool.get("token_groups", 1)]
    ]
    return max(found, default=(0, 0))


async def run_split_program(
    engines: Any,
    tokenizer: Any,
    task: Task,
    cfg: RunConfig,
    decode_path: str,
    unit: int,
    program_id: str | None = None,
    run_id: str = "",
    run_epoch: float | None = None,
    concurrency: int = 1,
    program_index: int = 0,
    controller: Any = None,
    method_router: PolicyRuntime | None = None,
) -> ProgramResult:
    """Run every stage of one program (`stage_plan`), each decoded on its home.

    Stages, order, records and prompt ids match a single-engine program: each prompt
    is the last prompt's ids, D's generated ids and the template's next turn
    (`render_successor`). The dispatcher places each stage, priced by every engine's
    prefix-cache prediction: the generation stage prefills on a P and decodes on the
    program's home; a later stage runs on that D alone, or on a P that first pulls
    the history from it (`check_pull_prompt`). The previous stage's hold on D is let
    go at this stage's first token.

    Args:
        unit: Tokens per cache key, the same on every engine.
        program_id: Overrides the default `{sample}-{path}`.
        run_id: Salt namespace.
        run_epoch: `perf_counter` origin shared by the whole run.
        program_index: The program's place in the job list; picks its seed under
            `seed_per_program`.
        controller: Told when each stage goes out and comes back, as in
            `program.pipeline.run_program`; its answer is the stage priority sent to
            both engines. None uses `RunConfig.stage_priority_value`.
        method_router: The pass's `PolicyRuntime`, or None to make one.
    """
    # Imports vLLM, which this process may hold only once every worker exists.
    from turboreflect.program.pipeline import (
        _split_stray_reasoning,
    )

    program_id = program_id or f"{task.id}-{decode_path}"
    settings = cfg.pd.settings()
    if method_router is None:
        method_router = cfg.pd_method.make_router()
    sampling = split_sampling(cfg, decode_path, program_index)
    ledger = Ledger.for_run(cfg)
    epoch = time.perf_counter() if run_epoch is None else run_epoch

    salt = _program_salt(cfg, run_id, program_id)

    result = ProgramResult(
        program_id=program_id,
        sample_id=task.id,
        decode_path=decode_path,
        concurrency=concurrency,
        seed=sampling["seed"],
    )
    # One placeholder per clip, which the engines expand (`clip_ref` names them).
    task = bind(task, cfg)

    draft: str | None = None
    critique: str | None = None
    program_start = time.perf_counter()
    result.t_start_s = program_start - epoch
    # The last prompt and D's ids for it: the next prompt carries them as sampled.
    previous: Rendered | None = None
    previous_output_ids: list[int] | None = None
    # What the last stage's D computed KV for (engine ids), and its pull params.
    history: list[int] = []
    previous_params: Any = None
    last_output_at: float | None = None
    loop = asyncio.get_running_loop()
    # The program's home: the only prefix cache holding its generations.
    d = None
    # Unreleased holds by D's request id, let go at once if the program fails.
    holding: dict[str, str] = {}
    clips = clip_ref(task)
    media = {} if clips is None else {"media": clips}
    # The generation stage's clip expansion: from the startup layout where prompts
    # are placed by position, else from the first stage's served ids.
    expansion: Expansion | None = None
    layout = clip_layout(engines, task, cfg)

    program_plan = stage_plan(cfg, decode_path=decode_path)
    stages = program_plan.stages
    # Per-stage KV bounds, reserved before the generation stage is sent and
    # lowered as stages end.
    bounds: list[StageBound] = []
    shapes: list[StageShape] = []
    lookahead = 1 + (cfg.suffix_num_speculative_tokens if cfg.suffix_decoding else 0)
    state = decode_state(engines)
    permit_s = 0.0

    def max_tokens_of(planned: PlannedStage) -> int:
        return (
            sampling["max_tokens"] if planned.max_tokens is None else planned.max_tokens
        )

    def declared_rest(index: int) -> tuple[tuple[int, int, int], ...]:
        return _planned_rest(bounds, shapes, [max_tokens_of(s) for s in stages], index)

    try:
        for index, planned in enumerate(stages):
            stage = planned.stage
            if index == 0:
                permit_s = await engines.render_permit(program_id)
            # Off the event loop: a long encode would stall every other program.
            render_started = time.perf_counter()
            rendered: Rendered = await loop.run_in_executor(
                None,
                functools.partial(
                    render_successor,
                    tokenizer,
                    task,
                    stage,
                    cfg,
                    previous=previous,
                    generated_ids=previous_output_ids,
                    draft=draft,
                    critique=critique,
                    nest=planned.nest,
                ),
            )
            render_s = time.perf_counter() - render_started
            previous = rendered
            # Every cache prediction, price, cut and record is made on these (`view`).
            expansion, prompt_ids = _engine_ids(stage, rendered, expansion, layout)
            view = dataclasses.replace(rendered, token_ids=prompt_ids)
            request_id = f"{program_id}-{stage}"
            # The engine holding the previous stage's KV, for a stage that moves.
            holder = d
            first_at: list[float] = []
            history_tokens = lcp(history, prompt_ids)
            prefill_task = PrefillTask(history_tokens, len(prompt_ids) - history_tokens)
            predictions = await loop.run_in_executor(
                None,
                functools.partial(
                    ledger.predict_each,
                    settings.names(),
                    prompt_ids,
                    unit,
                ),
            )
            asked = time.perf_counter()
            if index == 0:
                shapes = await _stage_shapes(
                    method_router,
                    tokenizer,
                    task,
                    cfg,
                    rendered,
                    stages,
                    unit,
                )
                maxes = [max_tokens_of(s) for s in stages]
                bounds = stage_bounds(len(prompt_ids), maxes, shapes)
            ttl = settings.decoder_kv_blocks_ttl_s
            held_since = (
                last_output_at
                if (previous_params or {}).get("remote_request_id")
                else None
            )
            item = WorkItem(
                request_id,
                program_id,
                LATER_STAGE if index else GENERATION,
                len(prompt_ids),
                max_tokens_of(planned),
                hits=predictions,
                home=d,
                offered_tokens=0
                if held_since is None
                else offered_tokens(
                    previous_params,
                    view,
                    history_tokens,
                    unit,
                    settings.text_join_pull,
                ),
                hold_remaining_s=None
                if held_since is None or ttl is None
                else ttl - (asked - held_since),
                peak_blocks=0 if index else peak_blocks(bounds, unit, lookahead, state),
                rest=declared_rest(index),
                media=clips is not None,
            )
            placed = await engines.dispatch(item)
            d = placed.home
            move = placed.route == REMOTE_HOME
            decision = {
                **placed.record,
                "route": placed.route,
                "worker": placed.engine,
                "reason": placed.record["phase"],
            }
            local = decision["route"] == LOCAL
            pull = index > 0 and not local
            engine = d if local else decision["worker"]

            joined = join_record(view, history if index > 0 else None)
            pull_params = previous_params
            pull_record: dict[str, Any] = {}
            if pull:
                pull_params, pull_record, pull = _plan_pull(
                    stage,
                    view,
                    history,
                    previous_params,
                    joined,
                    unit,
                    settings.text_join_pull,
                )

            stage_sampling = _stage_sampling(sampling, planned, prefill_task)
            priority = await _stage_priority(controller, cfg, index, program_index)
            started = time.perf_counter()
            stage_start = started - epoch
            host_gap_s = None if last_output_at is None else started - last_output_at
            previous_end = last_output_at
            # The last stage holds nothing: an unread hold pins D's blocks for the TTL.
            later = index < len(stages) - 1
            # The previous hold, let go at this stage's first token on D, when D has
            # scheduled on it and any pull is over; a pulled hold's release is ignored.
            held = (previous_params or {}).get("remote_request_id")
            hooks: dict[str, Any] = {}
            hooks["on_first_token"] = functools.partial(
                _first_token,
                first_at,
            )
            if move:
                # Moved to this P: it pulls from the home, decodes and holds.
                split = await engines.generate_local(
                    request_id,
                    rendered.token_ids,
                    salt,
                    stage_sampling,
                    d=d,
                    priority=priority,
                    export_for_pull=later,
                    held=held,
                    held_by=holder,
                    pull_params=pull_params if pull else None,
                    **hooks,
                    **media,
                )
            elif local:
                split = await engines.generate_local(
                    request_id,
                    rendered.token_ids,
                    salt,
                    stage_sampling,
                    d=d,
                    priority=priority,
                    held=held,
                    export_for_pull=later,
                    **hooks,
                    **media,
                )
            else:
                split = await engines.generate_stage(
                    request_id,
                    rendered.token_ids,
                    salt,
                    stage_sampling,
                    p=engine,
                    d=d,
                    prefill_params=pull_params if pull else None,
                    priority=priority,
                    export_for_pull=later,
                    held=held,
                    **hooks,
                    **media,
                )
            last_output_at = time.perf_counter()
            wall = last_output_at - started
            stage_end = last_output_at - epoch
            holding.pop((previous_params or {}).get("remote_request_id"), None)
            exported = (split.d_output.kv_transfer_params or {}).get(
                "remote_request_id"
            )
            if later and exported:
                holding[exported] = d

            waits: dict[str, Any] = {
                "admission_wait_s": decision["admission_wait_s"]
                + (permit_s if index == 0 else 0.0),
                "gate_wait_s": decision["gate_wait_s"],
            }
            if first_at and previous_end is not None:
                # Previous stage's last token to this one's first, driver clock.
                decision["handoff_s"] = first_at[0] - previous_end
            metrics = StageMetrics.from_output(
                stage,
                split.d_output,
                wall,
                split.handoff(),
                host_gap_s=host_gap_s,
                render_s=render_s,
                **waits,
            )
            if clips is not None:
                expansion, prompt_ids = _served_ids(split, rendered, expansion)
            if pull:
                pull_record["pull_recomputed_tokens"] = pull_recomputed_tokens(
                    split.d_output if move else split.p_output,
                    pull_params,
                    settings.kv_recompute_threshold or 0,
                )
            result.stages[stage] = metrics
            result.spans[stage] = (stage_start, stage_end)
            result.priorities[stage] = priority
            result.prompt_shapes[stage] = rendered.shape()
            result.joins[stage] = {**joined, "prefill_engine": engine}
            if pull_record:
                result.pulls[stage] = pull_record
            result.method_decisions[stage] = {
                **decision,
                "name": method_router.name,
                "history_tokens": prefill_task.history_tokens,
                "incremental_tokens": prefill_task.incremental_tokens,
                "affinity": {
                    "key": None,
                    "d": d,
                    "p": None if local else engine,
                },
            }

            completion = split.d_output.outputs[0]
            _record_output(
                result,
                planned.output_name,
                completion,
                cfg,
                decode_path,
                _split_stray_reasoning,
            )
            # Filed after this stage's prediction: P holds the prompt, the decoder
            # (if it caches) the generation too.
            ledger.record(
                prompt_ids,
                completion.token_ids,
                (d,) if local or move else (engine, d),
                decoded=d,
            )
            history = [*prompt_ids, *completion.token_ids]
            previous_params = split.d_output.kv_transfer_params
            if index + 1 < len(stages):
                engines.resize(
                    program_id,
                    peak_blocks(
                        bounds[index + 1 :],
                        unit,
                        lookahead,
                        state,
                        held_output=bounds[index].max_tokens,
                    ),
                )
            # Stripped reasoning breaks the ids, so the next stage renders text.
            previous_output_ids, draft, critique = _stage_inputs(
                cfg,
                decode_path,
                stage,
                completion.text,
                list(completion.token_ids),
                draft,
                critique,
                _split_stray_reasoning,
            )
    finally:
        for exported, holder in holding.items():
            engines.release_hold(holder, exported)
        engines.leave(program_id)

    _finish_program(result, cfg, program_plan, program_start)
    return result


def _program_salt(cfg: RunConfig, run_id: str, program_id: str) -> str | None:
    """The program's cache salt, or None with program isolation off."""
    if not cfg.isolate_programs:
        return None
    return cfg.cache_salt(run_id, program_id)


def _engine_ids(
    stage: str,
    rendered: Rendered,
    expansion: Expansion | None,
    layout: ClipLayout | None,
) -> tuple[Expansion | None, list[int]]:
    """The expansion and the ids the engines run, each clip placeholder expanded.

    A stage with no `expansion` yet takes it from `layout`, if the run has one.
    """
    try:
        if expansion is None and layout is not None:
            expansion = layout.expansion(rendered.token_ids)
        prompt_ids = (
            rendered.token_ids
            if expansion is None
            else expansion.engine_ids(rendered.token_ids)
        )
    except MediaError as exc:
        raise DriverError(f"{stage}: {exc}") from exc
    return expansion, prompt_ids


def _plan_pull(
    stage: str,
    view: Rendered,
    history: Sequence[int],
    previous_params: Any,
    joined: Mapping[str, Any],
    unit: int,
    text_join_pull: bool,
) -> tuple[Any, dict[str, Any], bool]:
    """What P is handed of D's hold for this prompt, and whether it pulls at all.

    A cut pull (`_cut_pull`) takes the blocks the prompt shares with D's history in
    engine ids. Its KV may end inside a clip: P's request carries the clips, and
    vLLM encodes a clip whose tail it computes.

    Returns:
        The params, the stage's pull record, and whether the prompt may pull them.
    """
    pull_params = previous_params
    if previous_params and _cut_pull(view, text_join_pull):
        pull_params = truncated_pull(previous_params, joined["join_lcp_tokens"], unit)
    refused = check_pull_prompt(
        stage,
        view,
        history,
        pull_params,
        text_join_pull=text_join_pull,
    )
    offered = offered_end(pull_params, len(view.token_ids))
    pull_record = {
        "pull_offered_tokens": offered,
        "pull_skipped": refused is not None,
    }
    return pull_params, pull_record, refused is None


def _stage_sampling(
    sampling: Mapping[str, Any], planned: PlannedStage, task: PrefillTask
) -> dict[str, Any]:
    """The program's sampling for one stage, tagged with its matched prefix."""
    stage_sampling = dict(sampling)
    if planned.max_tokens is not None:
        stage_sampling = {**stage_sampling, "max_tokens": planned.max_tokens}
    return {
        **stage_sampling,
        "extra_args": {
            **(stage_sampling.get("extra_args") or {}),
            "locality_pd_prefill": {"history_tokens": task.history_tokens},
        },
    }


async def _stage_priority(
    controller: Any,
    cfg: RunConfig,
    index: int,
    program_index: int,
) -> int:
    """The stage priority a single-engine program would get, sent to P and D."""
    if controller is None:
        return cfg.stage_priority_value(
            successor=index > 0, program_index=program_index
        )
    return await controller.submit(
        StageSubmit(has_predecessor=index > 0, program_index=program_index)
    )


def _served_ids(
    split: Any, rendered: Rendered, expansion: Expansion | None
) -> tuple[Expansion, list[int]]:
    """The clip expansion and prompt ids as the engines ran them."""
    served = list(split.d_output.prompt_token_ids)
    if expansion is None:
        expansion = Expansion.from_generation(rendered.token_ids, served)
    return expansion, served


def _record_output(
    result: ProgramResult,
    name: str,
    completion: Any,
    cfg: RunConfig,
    decode_path: str,
    split_reasoning: Any,
) -> None:
    """File a stage's completion under `name`, stray reasoning apart."""
    text = completion.text
    if strips_stray_reasoning(cfg.enable_thinking, decode_path):
        text, reasoning = split_reasoning(text)
        if reasoning:
            result.reasoning[name] = reasoning
    result.outputs[name] = text
    result.output_token_ids[name] = list(completion.token_ids)


def _finish_program(
    result: ProgramResult,
    cfg: RunConfig,
    program_plan: StagePlan,
    program_start: float,
) -> None:
    """Set a finished program's artifact and its end-to-end time."""
    if cfg.pipeline_mode == PIPELINE_MODE_ROUNDS:
        record_rounds(result, program_plan)
    else:
        result.artifact = final_artifact(tuple(result.stages), result.outputs)
    if cfg.two_stage:
        result.artifact = merged_parts(result.artifact)[1]
    result.e2e_seconds = time.perf_counter() - program_start
    result.t_end_s = result.t_start_s + result.e2e_seconds


async def _stage_shapes(
    runtime: PolicyRuntime,
    tokenizer: Any,
    task: Task,
    cfg: RunConfig,
    generation: Rendered,
    declared: Sequence[PlannedStage],
    unit: int,
) -> list[StageShape]:
    """Each declared later stage's shape (`StageShape`), measured once a pass.

    Rendered on empty outputs, a stage's length past the generation prompt is the
    template alone, the same for every sample. One block of slack per stage covers
    a token join's boundary token.
    """
    key = tuple(planned.stage for planned in declared)
    measured = runtime.shapes.get(key)
    if measured is None:
        measured = runtime.shapes[key] = asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                _measure_shapes, tokenizer, task, cfg, generation, declared, unit
            ),
        )
    return await asyncio.shield(measured)


def _measure_shapes(
    tokenizer: Any,
    task: Task,
    cfg: RunConfig,
    generation: Rendered,
    declared: Sequence[PlannedStage],
    unit: int,
) -> list[StageShape]:
    shapes = []
    previous, draft, critique = generation, None, None
    for before, planned in zip(declared, declared[1:]):
        draft, critique = next_inputs(before.stage, "", draft, critique)
        rendered = render_successor(
            tokenizer,
            task,
            planned.stage,
            cfg,
            previous=previous,
            generated_ids=[],
            draft=draft,
            critique=critique,
            nest=planned.nest,
        )
        shared = lcp(previous.token_ids, rendered.token_ids)
        shapes.append(
            StageShape(
                len(rendered.token_ids) - len(generation.token_ids) + unit,
                len(previous.token_ids) - shared,
                shared == len(previous.token_ids),
            )
        )
        previous = rendered
    return shapes


def _planned_rest(
    bounds: Sequence[StageBound],
    shapes: Sequence[StageShape],
    max_tokens: Sequence[int],
    index: int,
) -> tuple[tuple[int, int, int], ...]:
    """The declared stages after `index`, as (matched prefix, append, output bound)."""
    rest = []
    for k in range(index + 1, len(bounds)):
        shape = shapes[k - 1]
        matched = bounds[k - 1].prompt_tokens - shape.released_tokens
        if shape.matches_output:
            matched += max_tokens[k - 1]
        rest.append((matched, max(0, bounds[k].prompt_tokens - matched), max_tokens[k]))
    return tuple(rest)


def _open_dispatcher(
    engines: Any, cfg: RunConfig, locality_pd: LocalityPDSettings
) -> None:
    """Open the ledger on each home and the dispatcher on the deployment calibration."""
    settings = cfg.pd.settings()
    homes = [*settings.names(ROLE_D)]
    if locality_pd.drain_on_p:
        homes += settings.names(ROLE_P)
    engines.open_ledger(homes, watermark=0.0)
    prior = locality_pd.prior
    engines.open_dispatcher(
        prior.step,
        lambda tokens: prior.model.transfer(int(tokens), ANY_WORKER, ANY_WORKER),
        tail_programs_per_engine=locality_pd.tail_programs_per_engine,
        drain_on_p=locality_pd.drain_on_p,
        role_flex=locality_pd.role_flex,
        tpot_delta=locality_pd.tpot_delta,
    )


def _stage_inputs(
    cfg: RunConfig,
    decode_path: str,
    stage: str,
    text: str,
    ids: list[int] | None,
    draft: str | None,
    critique: str | None,
    split_reasoning: Any,
) -> tuple[list[int] | None, str | None, str | None]:
    """The ids, draft and critique the stage after `stage` renders on."""
    if strips_stray_reasoning(cfg.enable_thinking, decode_path):
        text, reasoning = split_reasoning(text)
        if reasoning:
            ids = None
    return (ids, *next_inputs(stage, text, draft, critique))


def _cut_pull(rendered: Rendered, text_join_pull: bool) -> bool:
    """Whether a pull is cut to what the prompt shares with D's history."""
    if rendered.join == JOIN_REQUOTE:
        return True
    return text_join_pull and rendered.join != JOIN_TOKENS


def _first_token(stamps: list[float]) -> None:
    stamps.append(time.perf_counter())


def _report(result: ProgramResult) -> None:
    print(f"{result.program_id}   e2e {result.e2e_seconds:.1f}s")
    for stage, metrics in result.stages.items():
        print("  " + metrics.line())


async def warm_up_pairs(
    cfg: RunConfig,
    settings: PDSettings,
    engines: Any,
    tokenizer: Any,
    task: Task,
    run_id: str,
) -> None:
    """Hand KV across every P/D pair the pass can use, before it starts.

    NIXL handshakes once per pair, at its first transfer, and a program reaches only
    the pairs the dispatcher gives it. So each P sends the first stage's prompt to
    each D for `PAIR_ROUNDS` rounds, with a salt per P and round so D pulls it whole;
    each P then pulls that history back from each D, and last each D prefills a
    later stage alone. Every step is bounded by `pd.request_deadline_s`, as P's
    prefill has no watchdog.
    """
    deadline_s = settings.request_deadline_s
    sampling = {
        **split_sampling(cfg, cfg.decode_paths[0]),
        "max_tokens": PAIR_GEN_TOKENS,
        "ignore_eos": True,
    }
    loop = asyncio.get_running_loop()
    task = bind(task, cfg)
    first = await loop.run_in_executor(
        None,
        functools.partial(render_successor, tokenizer, task, cfg.stages[0], cfg),
    )
    later = cfg.stages[1] if len(cfg.stages) > 1 else None
    clips = clip_ref(task)
    media = {} if clips is None else {"media": clips}
    # A pull is cut on the engine's ids, as in `run_split_program`.
    layout = clip_layout(engines, task, cfg)
    try:
        expansion = None if layout is None else layout.expansion(first.token_ids)
        first_ids = engine_view(first, expansion).token_ids
    except MediaError as exc:
        raise DriverError(f"warmup: {exc}") from exc

    async def bounded(what: str, step: Any) -> Any:
        try:
            return await asyncio.wait_for(step, deadline_s)
        except asyncio.TimeoutError:
            raise DriverError(
                f"warmup {what}: no answer within pd.request_deadline_s "
                f"({deadline_s:.1f}s)"
            ) from None

    async def render_later(out: Any) -> Rendered:
        completion = out.outputs[0]
        return await loop.run_in_executor(
            None,
            functools.partial(
                render_successor,
                tokenizer,
                task,
                later,
                cfg,
                previous=first,
                generated_ids=list(completion.token_ids),
                draft=completion.text,
            ),
        )

    async def pull_back(p: str, d: str, round_: int, out: Any) -> None:
        """P pulls from `d` the history it just sent there."""
        what = f"{p}<-{d}"
        rendered = await render_later(out)
        completion = out.outputs[0]
        params = out.kv_transfer_params
        history = [*first_ids, *completion.token_ids]
        view = engine_view(rendered, expansion)
        if params and _cut_pull(rendered, settings.text_join_pull):
            block = engines.facts[settings.names()[0]].block_size
            shared = join_record(view, history)["join_lcp_tokens"]
            params = truncated_pull(params, shared, block)
        await bounded(
            what,
            engines.generate_stage(
                f"warmup-pull-{p}-{d}-{round_}-{later}",
                rendered.token_ids,
                f"{run_id}:warmup-pull-{p}-{d}-{round_}",
                sampling,
                p=p,
                d=d,
                prefill_params=params,
                # Unread, so a hold would make the drain wait out D's TTL.
                export_for_pull=False,
                **media,
            ),
        )

    async def prefill_alone(d: str, salt: str, out: Any) -> None:
        rendered = await render_later(out)
        await bounded(
            f"{d} alone",
            engines.generate_local(
                f"warmup-local-{d}-{later}",
                rendered.token_ids,
                salt,
                sampling,
                d=d,
                **media,
            ),
        )

    print(
        f"warmup: every P->D pair of {list(settings.names(ROLE_P))} x "
        f"{list(settings.names(ROLE_D))}, {PAIR_ROUNDS} rounds"
    )
    # Each D's newest transfer: its salt and what it answered.
    newest: dict[str, tuple[str, Any]] = {}
    for round_ in range(PAIR_ROUNDS):
        for p in settings.names(ROLE_P):
            salt = f"{run_id}:warmup-pair-{p}-{round_}"
            for d in settings.names(ROLE_D):
                what = f"{p}->{d}"
                split = await bounded(
                    what,
                    engines.generate_stage(
                        f"warmup-pair-{p}-{d}-{round_}-{cfg.stages[0]}",
                        first.token_ids,
                        salt,
                        sampling,
                        p=p,
                        d=d,
                        export_for_pull=later is not None,
                        **media,
                    ),
                )
                newest[d] = (salt, split.d_output)
                if later:
                    await pull_back(p, d, round_, split.d_output)
    if later:
        for d, (salt, out) in newest.items():
            await prefill_alone(d, salt, out)


async def warm_up_split(
    cfg: RunConfig,
    engines: Any,
    tokenizer: Any,
    unit: int,
    run_id: str,
    concurrency: int = 0,
    program: Any = run_split_program,
) -> None:
    """Pay every engine's lazy initialisation (JIT, attention autotune) before the pass.

    After `warm_up_pairs`, one sample runs down every decode path at full length,
    then `engine.runner.warm_up_concurrent` compiles the shapes only a concurrent
    pass makes. Both run the split program: warming each engine alone would prefill
    on D and decode on P, which no request does, and leave the transfer cold.

    Args:
        concurrency: Programs the concurrent phase keeps in flight; 0 or 1 skips it.
    """
    # The first sample with clips, so the vision encoder's first pass is the warmup's.
    tasks = [load_task(cfg.sample_path(sample)) for sample in cfg.samples]
    task = next((t for t in tasks if t.videos), tasks[0])
    print(_BAR)
    print(
        f"warmup: {task.id} down {len(cfg.decode_paths)} path(s), "
        "discarded and not summarised"
    )
    settings = cfg.pd.settings()
    method_router = cfg.pd_method.make_router()
    program = functools.partial(program, method_router=method_router)
    # The warmup's own dispatcher, for render permits; the pass opens its own.
    _open_dispatcher(engines, cfg, method_router.settings)
    engines.open_window("warmup")
    await warm_up_pairs(cfg, settings, engines, tokenizer, task, run_id)

    for decode_path in cfg.decode_paths:
        result = await program(
            engines,
            tokenizer,
            task,
            cfg,
            decode_path,
            unit,
            program_id=warmup_program_id(task.id, decode_path),
            run_id=run_id,
            concurrency=1,
        )
        _report(result)

    # The driver doubles as step logger: it opens a window in every worker.
    await warm_up_concurrent(
        engines,
        tokenizer,
        cfg,
        engines,
        run_id,
        concurrency,
        program=functools.partial(program, unit=unit),
    )
    await engines.drain()
    # Engine totals cover the pass only; reset in the workers, where loggers live.
    engines.reset_steps()


def _load_tokenizer(cfg: RunConfig) -> Any:
    """The model's tokenizer, loaded once every worker is spawned, since
    `transformers` may import torch.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg.model)


def _describe(report: Mapping[str, Any], settings: PDSettings) -> None:
    """Print each engine's role, card and KV pool."""
    print(_BAR)
    for name in settings.names():
        pool = report[name]["kv_pool"]
        print(
            f"{name} engine  : {settings.kv_role(name)} on card "
            f"{report[name]['visible_devices']!r}, side channel port "
            f"{settings.port(name)}, layout {settings.kv_cache_layout}, prefix "
            f"caching {'on' if settings.prefix_caching(name) else 'off'}"
        )
        print(
            f"{name} kv pool : {pool['capacity_tokens']} tokens "
            f"({pool['kv_pool_gib']:.1f} GiB), fits "
            f"{pool['max_concurrent_full_sequences']} full-length sequence(s)"
        )
    print(_BAR)


def _write_summary(
    writer: ResultWriter,
    batch: PassResult,
    totals: dict[str, Any],
    kv_pool: dict[str, Any],
    report: dict[str, Any],
    steps_paths: dict[str, str],
) -> Path:
    """Write the run's summary, with its P/D section and where each trace went."""
    summary = merge_section(
        writer.write_summary(batch, totals, kv_pool),
        "pd",
        # As each engine reported itself, not as the config asked.
        {"enabled": True, "engines": report},
    )
    return merge_section(
        summary,
        "engine_trace",
        {
            "dirs": {
                name: str(trace_dir(path, name).relative_to(writer.dir))
                for name, path in steps_paths.items()
            }
        },
    )


async def execute_pd(cfg: RunConfig) -> int:
    """Start every engine and run the pass across them; return exit code 0.

    Raises:
        DriverError: If the pass could not run.
        PDDriverError: If the engines could not be brought up.
    """
    settings = cfg.pd.settings()

    # Before the workers: each writes its engine's trace inside the run directory.
    writer = ResultWriter(cfg)
    steps_paths = {
        name: str(writer.dir / f"engine_steps_{name}.jsonl")
        for name in settings.names()
    }
    engines = PDDriver(cfg, settings, steps_paths=steps_paths)
    async with engines:
        report = engines.report()
        _describe(report, settings)
        tokenizer = _load_tokenizer(cfg)
        # The engines share one block size; no `CacheConfig` is reachable here.
        unit = engines.facts[settings.names()[0]].block_size
        kv_pool = {name: report[name]["kv_pool"] for name in settings.names()}

        writer.write_run_metadata(kv_pool=kv_pool, pd=dict(report))
        print(f"results -> {writer.dir}")

        await warm_up_split(
            cfg,
            engines,
            tokenizer,
            unit,
            writer.run_id,
            concurrency=warmup_concurrency(cfg),
        )

        # The programs, their ids and their salts are `turboreflect.engine.runner`'s.
        jobs = program_jobs(
            cfg, [load_task(cfg.sample_path(sample)) for sample in cfg.samples]
        )
        n = cfg.concurrency
        engines.open_window(f"n{n}")
        batch = PassResult(concurrency=n)
        semaphore = asyncio.Semaphore(n)
        method_router = cfg.pd_method.make_router()
        _open_dispatcher(engines, cfg, method_router.settings)
        epoch = time.perf_counter()

        failed = asyncio.Event()

        async def one(index, job):
            task, decode_path = job
            async with semaphore:
                if failed.is_set():
                    raise asyncio.CancelledError
                try:
                    result = await run_split_program(
                        engines,
                        tokenizer,
                        task,
                        cfg,
                        decode_path,
                        unit,
                        program_id=cfg.program_id(task.id, decode_path, n),
                        run_id=writer.run_id,
                        run_epoch=epoch,
                        concurrency=n,
                        program_index=index,
                        method_router=method_router,
                    )
                    # Off the loop: a stalled file system would hold every stage.
                    path = await asyncio.to_thread(writer.write_program, result)
                    print(f"  wrote {path.name}   e2e {result.e2e_seconds:.1f}s")
                    return result
                except BaseException:
                    # Stop queued programs before releasing this execution slot.
                    failed.set()
                    raise

        print(_BAR)
        print(f"pass n={n}: {len(jobs)} programs")
        tasks = [asyncio.create_task(one(i, j)) for i, j in enumerate(jobs)]
        try:
            batch.programs = list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        batch.makespan_s = time.perf_counter() - epoch
        batch.engine = await engines.step_totals("window")

        for result in batch.programs:
            print(_BAR)
            _report(result)
        print(_BAR)
        print(
            f"pass n={n}: makespan {batch.makespan_s:.1f}s, sum of e2e "
            f"{sum(p.e2e_seconds for p in batch.programs):.1f}s"
        )
        decided = [
            decision
            for result in batch.programs
            for decision in result.method_decisions.values()
        ]
        if decided:
            print(
                f"  {decided[0]['name']}: route "
                f"{dict(Counter(d['route'] for d in decided))}, reason "
                f"{dict(Counter(d['reason'] for d in decided))}"
            )
        for name in settings.names():
            side = batch.engine[name]
            print(
                f"  {name}: {side['steps']} steps, peak kv "
                f"{side['peak_kv_cache_usage']:.2f}"
            )
        # Every cache is emptied after the pass (`PDDriver.drain`).
        await engines.drain()
        totals = await engines.step_totals("run")

    summary = _write_summary(writer, batch, totals, kv_pool, report, steps_paths)

    print(_BAR)
    print(f"summary -> {summary}")
    metrics = write_metrics_safely(writer.dir)
    if metrics is not None:
        print(f"metrics -> {metrics}")
    print("done")
    return 0

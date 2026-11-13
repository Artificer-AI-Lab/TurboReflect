# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""One reflection program: generation, feedback, refinement, or rounds of the last two.

Each stage is an independent request; no KV is handed between stages. The next prompt
is the previous prompt's ids, the ids that stage generated and the template's next
turn (`render_successor`), so context reuse comes from the engine's prefix cache and
the blocks the scheduler holds for the successor (`turboreflect.context_reuse`).

Only the driver reads the round count: the controller is told whether a stage has a
predecessor, so a policy cannot recover the round, the round count or the stage total.
Each program carries its own cache salt, which enters block zero's hash, so no other
program can reach any block of its chain.

A sample's clips go with every stage and the engine expands each placeholder itself;
later stages' engine ids are predicted from the generation stage's
(`turboreflect.program.media`).
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import time
from typing import Any

import regex as re

from turboreflect.config import (
    PIPELINE_MODE_ROUNDS,
    RunConfig,
    strips_stray_reasoning,
)
from turboreflect.context_reuse.admission import StageSubmit
from turboreflect.metrics.metrics import StageMetrics
from turboreflect.metrics.results import ProgramResult, final_artifact
from turboreflect.program.media import (
    Expansion,
    InputLane,
    SampleMedia,
    bind,
    input_lane,
)
from turboreflect.program.prompt import (
    Rendered,
    join_record,
    render_successor,
)
from turboreflect.program.rounds import next_inputs, record_rounds, stage_plan
from turboreflect.program.stages import merged_parts
from turboreflect.program.task import Task
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM


class PipelineError(Exception):
    """A stage could not be run at all."""


def sampling_draw(cfg: RunConfig, program_index: int = 0) -> dict[str, Any]:
    """How a request of program `program_index` samples, on every path."""
    return dict(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        seed=cfg.request_seed(program_index),
    )


def sampling_params(
    cfg: RunConfig, decode_path: str, program_index: int = 0
) -> SamplingParams:
    """The only place the decode paths diverge.

    All paths sample alike, so `quality` output is what `efficiency` would produce had
    it been allowed to stop. They differ in the cap, in honouring the end token
    (`terminating` is `efficiency` with it honoured) and in consuming the stream: the
    final-only paths yield once, so the host does nothing while tokens are produced.
    The seed (`RunConfig.request_seed`) is per request, since the engine-level seed does
    not reach a request's sampler.
    """
    draw = sampling_draw(cfg, program_index)
    if decode_path == "quality":
        return SamplingParams(
            max_tokens=cfg.max_tokens_quality,
            output_kind=RequestOutputKind.CUMULATIVE,
            detokenize=True,
            **draw,
        )
    if decode_path == "efficiency":
        return SamplingParams(
            max_tokens=cfg.gen_tokens,
            ignore_eos=True,
            output_kind=RequestOutputKind.FINAL_ONLY,
            detokenize=True,
            **draw,
        )
    if decode_path == "terminating":
        return SamplingParams(
            max_tokens=cfg.gen_tokens,
            output_kind=RequestOutputKind.FINAL_ONLY,
            detokenize=True,
            **draw,
        )
    raise PipelineError(f"unknown decode path {decode_path!r}")


async def _run_final_only(
    llm: AsyncLLM,
    prompt: TokensPrompt,
    params: SamplingParams,
    request_id: str,
    priority: int = 0,
    lane: InputLane | None = None,
) -> tuple[Any, float]:
    """Final-only paths: one yield, at the end."""
    start = time.perf_counter()
    if lane is not None:
        prompt = await lane.prepare(prompt)
    final = None
    async for out in llm.generate(
        prompt, params, request_id=request_id, priority=priority
    ):
        final = out
    return final, time.perf_counter() - start


async def _run_streaming(
    llm: AsyncLLM,
    prompt: TokensPrompt,
    params: SamplingParams,
    request_id: str,
    priority: int = 0,
    lane: InputLane | None = None,
) -> tuple[Any, float]:
    """Quality path: consume the stream to its last output."""
    start = time.perf_counter()
    if lane is not None:
        prompt = await lane.prepare(prompt)
    final = None
    async for out in llm.generate(
        prompt, params, request_id=request_id, priority=priority
    ):
        final = out
    return final, time.perf_counter() - start


async def run_program(
    llm: AsyncLLM,
    tokenizer: Any,
    task: Task,
    cfg: RunConfig,
    decode_path: str,
    program_id: str | None = None,
    run_id: str = "",
    run_epoch: float | None = None,
    concurrency: int = 1,
    program_index: int = 0,
    controller: Any = None,
    first_admission_wait_s: float | None = None,
    media: SampleMedia | None = None,
) -> ProgramResult:
    """Run every planned stage over one sample and return what they cost.

    Args:
        llm: A running engine, shared with every other concurrent program.
        decode_path: `quality`, `efficiency` or `terminating`.
        program_id: Overrides the default `{sample}-{path}`; the driver passes
            `RunConfig.program_id`.
        run_id: Salt namespace under program isolation (`RunConfig.cache_salt`).
        run_epoch: The run's shared `perf_counter` origin for stage intervals.
        program_index: Place in the run's job list: the seed and, under
            `context_reuse_class`, the tie-break inside a priority class.
        controller: The admission controller, told when a stage goes out and
            answering its priority; None takes the priority from the config.
        first_admission_wait_s: Wait before the first stage, added to its admission
            wait.
        media: The sample's decoded clips, carried by every stage; None for text.

    Raises:
        PipelineError: On an unknown decode path or a stage with no output.
    """
    program_id = program_id or f"{task.id}-{decode_path}"
    params = sampling_params(cfg, decode_path, program_index)
    epoch = time.perf_counter() if run_epoch is None else run_epoch

    salt: str | None = None
    if cfg.isolate_programs:
        salt = cfg.cache_salt(run_id, program_id)

    media_fields = {} if media is None else media.request_fields(salt)
    lane = None if media is None else input_lane(llm)
    task = bind(task, cfg, media)
    # Set by the first stage of a program with clips.
    expansion: Expansion | None = None

    result = ProgramResult(
        program_id=program_id,
        sample_id=task.id,
        decode_path=decode_path,
        concurrency=concurrency,
        seed=params.seed,
    )

    draft: str | None = None
    critique: str | None = None
    program_start = time.perf_counter()
    result.t_start_s = program_start - epoch
    # Whether a stage has finished: a flag, so no stage index reaches the controller.
    successor = False
    # The last output's arrival; the host gap (render, admission wait) runs from it.
    last_output_at: float | None = None
    previous: Rendered | None = None
    # The previous stage's generated ids as sampled, not as their text re-encodes.
    previous_output_ids: list[int] | None = None
    # The engine's prompt and every id generated, stripped reasoning included.
    history: list[int] | None = None
    loop = asyncio.get_running_loop()

    async def run_stage(
        stage: str,
        output_name: str,
        *,
        stage_draft: str | None = None,
        stage_critique: str | None = None,
        nest: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        """Submit one stage and return its generated text.

        `stage_draft` is what a feedback stage reviews, `stage_critique` what a
        refinement stage applies; the rest is as in `PlannedStage`.
        """
        nonlocal successor, last_output_at, previous, previous_output_ids, history
        nonlocal expansion
        # Off the event loop, where a 128k encode would stall every program's stream;
        # timed on this side of the executor, so `render_s` covers the hop.
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
                draft=stage_draft,
                critique=stage_critique,
                nest=nest,
            ),
        )
        render_s = time.perf_counter() - render_started
        previous = rendered
        prompt = TokensPrompt(prompt_token_ids=rendered.token_ids)
        if salt is not None:
            prompt["cache_salt"] = salt
        prompt.update(media_fields)
        request_id = f"{program_id}-{stage}"
        engine_ids = (
            rendered.token_ids
            if expansion is None
            else expansion.engine_ids(rendered.token_ids)
        )

        stage_params = params
        if max_tokens is not None and max_tokens != params.max_tokens:
            stage_params = params.clone()
            stage_params.max_tokens = max_tokens
        admit_started = time.perf_counter()

        # Before `submitted_at`, so the controller's cost stays inside the host gap.
        if controller is None:
            priority = cfg.stage_priority_value(
                successor=successor,
                program_index=program_index,
            )
        else:
            priority = await controller.submit(
                StageSubmit(has_predecessor=successor, program_index=program_index)
            )
        submitted_at = time.perf_counter()
        stage_start = submitted_at - epoch
        admission_wait_s = submitted_at - admit_started
        if not successor and first_admission_wait_s is not None:
            admission_wait_s += first_admission_wait_s
        host_gap_s = None if last_output_at is None else submitted_at - last_output_at
        # Read off the sampling params, not the path name, so the two cannot drift.
        if stage_params.output_kind == RequestOutputKind.FINAL_ONLY:
            final, wall = await _run_final_only(
                llm, prompt, stage_params, request_id, priority, lane
            )
        else:
            final, wall = await _run_streaming(
                llm, prompt, stage_params, request_id, priority, lane
            )
        last_output_at = time.perf_counter()
        stage_end = last_output_at - epoch
        if final is None:
            raise PipelineError(f"stage {stage!r} produced no output at all")
        if media is not None and expansion is None:
            served = list(final.prompt_token_ids)
            expansion = Expansion.from_generation(rendered.token_ids, served)
            engine_ids = served

        metrics = StageMetrics.from_output(
            stage,
            final,
            wall,
            host_gap_s=host_gap_s,
            render_s=render_s,
            admission_wait_s=admission_wait_s,
        )
        result.stages[stage] = metrics
        if cfg.refinement_gen_tokens is not None:
            result.max_tokens[stage] = stage_params.max_tokens
        result.spans[stage] = (stage_start, stage_end)
        result.priorities[stage] = priority
        result.prompt_shapes[stage] = rendered.shape()
        result.joins[stage] = join_record(
            dataclasses.replace(rendered, token_ids=engine_ids), history
        )

        text = final.outputs[0].text
        if strips_stray_reasoning(cfg.enable_thinking, decode_path):
            text, reasoning = _split_stray_reasoning(text)
            if reasoning:
                result.reasoning[output_name] = reasoning
        result.outputs[output_name] = text
        result.output_token_ids[output_name] = list(final.outputs[0].token_ids)
        history = [*engine_ids, *final.outputs[0].token_ids]
        # Stripped ids no longer spell the answer: the next stage re-renders text.
        previous_output_ids = (
            None
            if output_name in result.reasoning
            else list(final.outputs[0].token_ids)
        )
        successor = True
        return text

    plan = stage_plan(cfg, decode_path=decode_path)
    for planned in plan.stages:
        text = await run_stage(
            planned.stage,
            planned.output_name,
            stage_draft=draft,
            stage_critique=critique,
            nest=planned.nest,
            max_tokens=planned.max_tokens,
        )
        draft, critique = next_inputs(planned.stage, text, draft, critique)
    if cfg.pipeline_mode == PIPELINE_MODE_ROUNDS:
        record_rounds(result, plan)
    else:
        result.artifact = final_artifact(tuple(result.stages), result.outputs)
    if cfg.two_stage:
        result.artifact = merged_parts(result.artifact)[1]

    result.e2e_seconds = time.perf_counter() - program_start
    result.t_end_s = result.t_start_s + result.e2e_seconds
    return result


# A `<think>` block written with thinking off, to its close or the end of the text.
_STRAY_REASONING = re.compile(r"<think>.*?(?:</think>|\Z)", re.S)


def _split_stray_reasoning(text: str) -> tuple[str, str]:
    """`(answer, reasoning)`: reasoning emitted with thinking off, split out.

    The next stage gets the answer only, as chat templates drop reasoning from history
    turns; vLLM's parsers leave the block in the content when thinking is off.
    """
    blocks = _STRAY_REASONING.findall(text)
    if not blocks:
        return text, ""
    return _STRAY_REASONING.sub("", text).strip(), "".join(blocks)

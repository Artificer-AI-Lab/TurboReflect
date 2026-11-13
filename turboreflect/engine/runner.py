# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""One engine startup, then every task down every decode path.

The pass runs all programs under one `gather` capped by `AdmissionController`, as a
batch boundary would wait for its slowest program. Before it, warmups pay kernel JIT,
autotuning and the batch shapes only a concurrent pass produces; the prefix cache is
reset after them, never within the pass. `--dry-run` loads only a tokenizer and checks
that every stage fits under `max_model_len`.

    python -m turboreflect run --config cfg.yaml --dry-run
    python -m turboreflect run --config cfg.yaml --out-dir runs/cfg
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
from collections.abc import Awaitable, Callable, Container, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from turboreflect.config import (
    CONTEXT_REUSE_SCHEDULER,
    PIPELINE_MODE_ROUNDS,
    STAGE_PRIORITY_NONE,
    ConfigError,
    RunConfig,
)
from turboreflect.context_reuse.admission import (
    AdmissionController,
    StageSubmit,
)
from turboreflect.metrics import provenance
from turboreflect.metrics.report import write_metrics_safely
from turboreflect.metrics.results import ProgramResult, write_json
from turboreflect.program import media as media_module
from turboreflect.program.prompt import (
    base_stage,
    filler_ids,
    join_record,
    render,
    render_incremental,
    render_successor,
    stage_name,
)
from turboreflect.program.task import Task, load_task

_BAR = "=" * 78

# Companion `k` keeps `k` times this many context tokens, so the concurrent warmup's
# prompts and prefill chunks all differ in length.
WARMUP_COMPANION_TOKENS = 1024
# Cap on a companion stage's generation, also cut to a quarter of `gen_tokens` so a
# wave of three companion stages fits inside one anchor stage.
WARMUP_COMPANION_GEN_TOKENS = 64


class DriverError(Exception):
    """The run cannot go on as configured."""


def _generation_budget(
    cfg: RunConfig, decode_path: str, stage: str = "generation"
) -> int:
    """Upper bound on tokens `stage` may generate; the worst case is what must fit."""
    return cfg.stage_gen_tokens(base_stage(stage), decode_path)


def _stand_ins(
    tokenizer: Any, cfg: RunConfig, decode_path: str
) -> dict[str, list[int]]:
    """Filler as long as each base stage may generate on this path."""
    return {
        base: filler_ids(tokenizer, _generation_budget(cfg, decode_path, base))
        for base in ("generation", "feedback", "refinement")
    }


def _round_renders(
    tokenizer: Any, task: Any, cfg: RunConfig, stand_ins: Mapping[str, list[int]]
) -> dict[str, Any]:
    """Every stage the round loop reaches up to `max_rounds`, rendered at its worst."""
    texts = {base: tokenizer.decode(ids) for base, ids in stand_ins.items()}
    rendered = render_incremental(tokenizer, task, "generation", cfg)
    renders = {"generation": rendered}
    bases = ("refinement",) if cfg.two_stage else ("feedback", "refinement")
    last = drafter = "generation"
    for round_index in range(1, cfg.max_rounds + 1):
        for base in bases:
            stage = stage_name(base, round_index)
            rendered = render_successor(
                tokenizer,
                task,
                stage,
                cfg,
                previous=rendered,
                generated_ids=stand_ins[last],
                draft=texts[drafter],
                critique=texts["feedback"],
                nest=round_index > 1,
            )
            renders[stage] = rendered
            last = base
            if base == "refinement":
                drafter = base
    return renders


def price_stages(
    tokenizer: Any, task: Any, cfg: RunConfig, decode_path: str
) -> tuple[dict[str, Any], dict[str, int]]:
    """Every stage prompt one program can submit on this path, at its worst.

    Outputs that do not exist yet are filler as long as their stage may produce.

    Returns:
        The generation, feedback and refinement renders (no feedback when two-stage),
        and every stage a program submits, in order: those, or the round loop's.
    """
    stand_ins = _stand_ins(tokenizer, cfg, decode_path)
    draft = tokenizer.decode(stand_ins["generation"])
    generation = render_incremental(tokenizer, task, "generation", cfg)
    if cfg.two_stage:
        refinement = render_successor(
            tokenizer,
            task,
            "refinement",
            cfg,
            previous=generation,
            generated_ids=stand_ins["generation"],
            draft=draft,
        )
        stages = {"generation": generation, "refinement": refinement}
    else:
        feedback = render_successor(
            tokenizer,
            task,
            "feedback",
            cfg,
            previous=generation,
            generated_ids=stand_ins["generation"],
            draft=draft,
        )
        refinement = render_successor(
            tokenizer,
            task,
            "refinement",
            cfg,
            previous=feedback,
            generated_ids=stand_ins["feedback"],
            draft=draft,
            critique=tokenizer.decode(stand_ins["feedback"]),
        )
        stages = {
            "generation": generation,
            "feedback": feedback,
            "refinement": refinement,
        }
    if cfg.pipeline_mode == PIPELINE_MODE_ROUNDS:
        return stages, _round_renders(tokenizer, task, cfg, stand_ins)
    return stages, stages


def longest_sequence(tokenizer: Any, cfg: RunConfig) -> int:
    """The longest sequence any program can reach, its clips and generation included."""
    longest = 0
    for sample in cfg.samples:
        task = media_module.bind(load_task(cfg.sample_path(sample)), cfg)
        expansion = 0
        if task.videos:
            task, expansion = media_module.processor_expansion(cfg, task, tokenizer)
        for decode_path in cfg.decode_paths:
            _, chain = price_stages(tokenizer, task, cfg, decode_path)
            lengths = {stage: r.n_tokens + expansion for stage, r in chain.items()}
            if cfg.pipeline_mode != PIPELINE_MODE_ROUNDS:
                lengths = {stage: lengths[stage] for stage in cfg.stages}
            longest = max(
                longest,
                *(
                    n + _generation_budget(cfg, decode_path, stage)
                    for stage, n in lengths.items()
                ),
            )
    return longest


async def derive_max_num_seqs(cfg: RunConfig) -> int:
    """The sequence limit of a config that states none.

    The pool is known only after startup, so a throwaway engine measures it; the limit
    is the programs it holds at their peak, capped at the run's program count.
    """
    from turboreflect.engine.engine import build_engine, kv_pool_report

    with tempfile.TemporaryDirectory() as trace_dir:
        llm = build_engine(cfg, Path(trace_dir))
        try:
            report = await kv_pool_report(llm)
            tokenizer = llm.get_tokenizer()
        finally:
            llm.shutdown()
    drafts = (cfg.suffix_num_speculative_tokens or 0) if cfg.suffix_decoding else 0
    peak = longest_sequence(tokenizer, cfg) + 1 + drafts
    reuses_context = (
        cfg.stage_priority != STAGE_PRIORITY_NONE
        or cfg.scheduler_cls == CONTEXT_REUSE_SCHEDULER
    )
    programs = len(
        program_jobs(cfg, [load_task(cfg.sample_path(s)) for s in cfg.samples])
    )
    derived = max(1, min(report.sequences_fitting(peak, reuses_context), programs))
    print(
        f"max_num_seqs {derived}: {report.num_gpu_blocks} blocks over a "
        f"{peak}-token peak, {programs} program(s)"
    )
    return derived


def dry_run(cfg: RunConfig, mode: str) -> int:
    """Price every sample without a GPU. Returns a process exit code."""
    from transformers import AutoTokenizer

    # As the engine builds it: some checkpoints ship a custom-code tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model, trust_remote_code=bool(cfg.trust_remote_code)
    )
    ok = True

    tasks = [load_task(cfg.sample_path(sample)) for sample in cfg.samples]
    with_clips = {task.source for task in tasks if task.videos}
    print(_BAR)
    print(
        f"{len(cfg.samples)} sample(s); the warmup runs "
        f"{warmup_sample(cfg, with_clips)}"
    )
    n_programs = len(cfg.samples) * len(cfg.decode_paths)
    print(
        f"programs: {n_programs} = {len(cfg.samples)} sample(s) x "
        f"{len(cfg.decode_paths)} path(s)"
    )
    print(
        f"engine: attention_backend {cfg.attention_backend or 'auto'}, "
        f"kv_cache_dtype {cfg.kv_cache_dtype}, "
        f"tensor_parallel_size {cfg.tensor_parallel_size or 1}"
    )
    if cfg.temperature > 0:
        seeds = [cfg.request_seed(i) for i in range(n_programs)]
        print(
            f"sampling: temperature {cfg.temperature} top_p {cfg.top_p} "
            f"top_k {cfg.top_k}; seeds {seeds}"
        )
    else:
        print("sampling: greedy, no seed")
    if cfg.pipeline_mode == PIPELINE_MODE_ROUNDS:
        print(f"rounds: {cfg.max_rounds}; priced to the deepest")

    for task in tasks:
        print(_BAR)
        print(f"sample {task.id}   ({len(task.context)} context documents)")
        expansion = 0
        if task.videos:
            clips = f"video: {len(task.videos)} clip(s)"
            try:
                task = media_module.bind(task, cfg)
                task, expansion = media_module.processor_expansion(cfg, task, tokenizer)
            except Exception as exc:  # noqa: BLE001 - priced text-only instead
                print(
                    f"{clips} NOT counted -- the prompt sizes below are "
                    f"text-only ({type(exc).__name__}: {exc})"
                )
            else:
                print(f"{clips} add {expansion} tokens to every prompt (processor)")

        for decode_path in cfg.decode_paths:
            budget = _generation_budget(cfg, decode_path)
            refinement_budget = _generation_budget(cfg, decode_path, "refinement")
            stages, chain = price_stages(tokenizer, task, cfg, decode_path)
            lengths = {stage: r.n_tokens + expansion for stage, r in chain.items()}
            rounds = cfg.pipeline_mode == PIPELINE_MODE_ROUNDS
            shown = lengths if rounds else {s: lengths[s] for s in cfg.stages}
            refinement_note = (
                ""
                if refinement_budget == budget
                else f", refinement {refinement_budget}"
            )
            print(
                f"  {decode_path} path (generation budget {budget} tokens"
                f"{refinement_note})"
            )
            for stage, size in shown.items():
                budget = _generation_budget(cfg, decode_path, stage)
                need = size + budget
                headroom = cfg.max_model_len - need
                flag = "ok " if headroom >= 0 else "OVER"
                print(
                    f"    {flag} {stage:<12s} prompt {size:>7d}"
                    f"  + gen {budget:>5d}  = {need:>7d}"
                    f"  / {cfg.max_model_len}   headroom {headroom:>7d}"
                )
                if headroom < 0:
                    ok = False
            stand_ins = _stand_ins(tokenizer, cfg, decode_path)
            names = list(chain)
            for before, stage in zip(names, names[1:]):
                joined = join_record(
                    chain[stage],
                    [*chain[before].token_ids, *stand_ins[base_stage(before)]],
                )
                print(
                    f"    {stage} matches the {before} stage's prompt and output for "
                    f"{joined['matched_tokens']} tokens and appends "
                    f"{joined['append_tokens']}"
                )
            joins = sorted(
                {str(r.join) for s, r in stages.items() if s != "generation"}
            )
            print(f"    joins: {'/'.join(joins)}")

        if mode == "text":
            rendered = render(tokenizer, task, "generation", cfg)
            for msg in rendered.messages:
                body = msg["content"]
                head = body[:160].replace("\n", " ")
                tail = body[-80:].replace("\n", " ")
                print(f"    [{msg['role']}] {len(body)} chars: {head!r} ... {tail!r}")

    print(_BAR)
    print("dry run: fits" if ok else "dry run: DOES NOT FIT -- do not start engine")
    return 0 if ok else 1


def program_jobs(cfg: RunConfig, tasks: Sequence[Task]) -> list[tuple[Task, str]]:
    """Each program's task and decode path, in `RunConfig.program_slot` order."""
    by_entry = dict(zip(cfg.samples, tasks))
    return [
        (by_entry[sample], path)
        for sample, path in map(cfg.program_slot, range(cfg.program_count()))
    ]


def warmup_program_id(sample_id: str, decode_path: str) -> str:
    """The id a discarded program runs under.

    Salted by it, the warmup's prefix chain stays out of reach of the run's programs.
    """
    return f"warmup-{sample_id}-{decode_path}"


def concurrent_warmup_id(sample_id: str, decode_path: str, wave: int | None) -> str:
    """The id of a concurrent-warmup program: the anchor (`wave` None) or a companion.

    Each id is distinct, so each program opens its own salted prefix chain.
    """
    role = "anchor" if wave is None else f"w{wave}"
    return f"warmup-{role}-{sample_id}-{decode_path}"


@dataclass(frozen=True)
class WarmupCompanion:
    """One short program run beside the anchor in every wave."""

    sample: str
    decode_path: str
    context_tokens: int
    program_index: int


def warmup_concurrency(cfg: RunConfig) -> int:
    """Programs the concurrent warmup keeps in flight, one sample each."""
    return min(cfg.concurrency, len(cfg.samples))


async def admitted(
    control: AdmissionController,
    run_one: Callable[[float], Awaitable[ProgramResult]],
) -> ProgramResult:
    """Run a program once `control` admits it, given what it waited."""
    admit_started = time.perf_counter()
    await control.start()
    wait = time.perf_counter() - admit_started
    try:
        return await run_one(wait)
    finally:
        await control.end()


def companion_plan(cfg: RunConfig, concurrency: int) -> tuple[WarmupCompanion, ...]:
    """The companions that run beside the anchor, `samples[0]`, in each wave."""
    return tuple(
        WarmupCompanion(
            sample=cfg.samples[slot],
            decode_path=cfg.decode_paths[slot % len(cfg.decode_paths)],
            context_tokens=slot * WARMUP_COMPANION_TOKENS,
            program_index=slot,
        )
        for slot in range(1, concurrency)
    )


def companion_config(cfg: RunConfig) -> RunConfig:
    """`cfg` with each stage's generation cut to the companion budget."""
    gen = max(1, min(WARMUP_COMPANION_GEN_TOKENS, cfg.gen_tokens // 4))
    return replace(
        cfg,
        gen_tokens=gen,
        refinement_gen_tokens=None,
        max_tokens_quality=min(gen, cfg.max_tokens_quality),
    )


def truncate_context(task: Task, tokenizer: Any, tokens: int) -> Task:
    """`task` with its context documents cut to `tokens` tokens, in order."""
    kept: dict[str, str] = {}
    left = tokens
    for name, text in task.context.items():
        if left <= 0:
            break
        ids = tokenizer.encode(text, add_special_tokens=False)
        kept[name] = text if len(ids) <= left else tokenizer.decode(ids[:left])
        left -= len(ids)
    return replace(task, context=kept)


def companion_tasks(tasks: Sequence[Task]) -> list[Task]:
    """`tasks` with the clips of all but the first that has any dropped.

    One warms the vision encoder in every wave; more would re-encode clips per salt.
    """
    first = next((task for task in tasks if task.videos), None)
    return [
        task if task is first or not task.videos else replace(task, videos=())
        for task in tasks
    ]


class WaveTrigger:
    """Launches one wave of companions each time the anchor submits a stage.

    Stands in for the anchor's admission controller. The anchor is enqueued first, so
    each wave's prefill chunks share steps with its decode; waves run one at a time.
    """

    def __init__(self, cfg: RunConfig, launch: Any) -> None:
        self.cfg = cfg
        self.launch = launch
        self.waves: list[asyncio.Future] = []

    async def _after(self, previous: asyncio.Future | None, wave: int) -> list:
        if previous is not None:
            await previous
        return await self.launch(wave)

    async def submit(self, sub: StageSubmit) -> int:
        previous = self.waves[-1] if self.waves else None
        self.waves.append(asyncio.ensure_future(self._after(previous, len(self.waves))))
        return self.cfg.stage_priority_value(
            successor=sub.has_predecessor,
            program_index=sub.program_index,
        )

    async def cancel(self) -> None:
        """Stop every unfinished wave and wait for its requests to unwind."""
        for wave in self.waves:
            wave.cancel()
        await asyncio.gather(*self.waves, return_exceptions=True)


def warmup_sample(cfg: RunConfig, media: Container[str]) -> str:
    """The serial warmup's sample: the first with clips, else the first.

    A clip sample lets the warmup, not a program, pay the vision encoder's first pass.
    """
    return next(
        (s for s in cfg.samples if str(cfg.sample_path(s)) in media), cfg.samples[0]
    )


def media_kwargs(media: Mapping[str, Any] | None, task: Task) -> dict[str, Any]:
    """`media=` for a program over `task`, or nothing over a task without clips.

    Nothing rather than None, so a runner without clips need not accept the keyword.
    """
    clips = (media or {}).get(task.source) if task.videos else None
    return {} if clips is None else {"media": clips}


async def reset_media_caches(llm: Any) -> None:
    """Drop every processed and encoded clip, as a prefix reset drops blocks."""
    await llm.reset_mm_cache()
    await llm.reset_encoder_cache()


async def reset_prefix_cache(llm: Any, attempts: int = 5, pause_s: float = 0.5) -> None:
    """Clear the prefix cache, retrying while the core still holds finished blocks.

    The frontend sees a request's last token before the core frees its blocks, and the
    reset refuses while any is held; the run goes on either way.
    """
    for attempt in range(attempts):
        if attempt:
            await asyncio.sleep(pause_s)
        if await llm.reset_prefix_cache():
            return


async def warm_up(
    llm: Any,
    tokenizer: Any,
    cfg: RunConfig,
    step_logger: Any,
    run_id: str,
    concurrency: int = 1,
    media: Mapping[str, Any] | None = None,
) -> None:
    """Pay one program's lazy initialisation (JIT, autotuning) before the pass.

    A sample of the run runs at full length down every decode path, since a shorter
    stand-in would hit other chunk counts and cudagraph buckets; `warm_up_concurrent`
    covers shapes above n=1. The prefix cache is then reset, which covers salting off.

    Args:
        step_logger: The run's `StepLogger`, whose run totals are re-based here.
        concurrency: Programs the concurrent phase keeps in flight; 0 or 1 skips it.
        media: Each sample's decoded clips by `Task.source` (`load_media`).
    """
    from turboreflect.program.pipeline import run_program

    media = media or {}
    path = warmup_sample(cfg, media)
    task = load_task(cfg.sample_path(path))
    print(_BAR)
    print(
        f"warmup: {task.id} down {len(cfg.decode_paths)} path(s), "
        "discarded and not summarised"
    )
    step_logger.open_window("warmup")

    for decode_path in cfg.decode_paths:
        result = await run_program(
            llm,
            tokenizer,
            task,
            cfg,
            decode_path,
            program_id=warmup_program_id(task.id, decode_path),
            run_id=run_id,
            concurrency=1,
            **media_kwargs(media, task),
        )
        print(f"{result.program_id}   e2e {result.e2e_seconds:.1f}s")
        for stage, metrics in result.stages.items():
            print("  " + metrics.line())

    await warm_up_concurrent(
        llm, tokenizer, cfg, step_logger, run_id, concurrency, media=media
    )
    await reset_prefix_cache(llm)
    if media:
        await reset_media_caches(llm)
    # Engine totals cover the pass only.
    step_logger.reset_run()


async def warm_up_concurrent(
    llm: Any,
    tokenizer: Any,
    cfg: RunConfig,
    step_logger: Any,
    run_id: str,
    concurrency: int,
    program: Any = None,
    media: Mapping[str, Any] | None = None,
) -> None:
    """Pay the lazy initialisation that only a concurrent pass reaches.

    A serial program never shares a step, so mixed prefill/decode steps, multi-request
    prefills and ragged verification would first compile inside the pass: a mixed step
    runs attention outside the full cudagraph, and split-KV Triton decode specialises on
    the batch's longest context. The anchor, `samples[0]` at full length, gets a wave of
    short companions each time it submits a stage, whose prefill lands in its decode
    steps. Every program is salted, whatever `isolate_programs` says, so each prefills
    cold.

    Args:
        step_logger: Opens this phase's step window: the run's `StepLogger`, or a split
            run's `PDDriver`, which opens one in each worker.
        concurrency: Programs in flight, anchor included; 0 or 1 skips the phase.
        program: Runs one program; `program.pipeline.run_program` when None.
    """
    if program is None:
        from turboreflect.program.pipeline import run_program

        program = run_program
    if concurrency < 1:
        return
    companions = companion_plan(cfg, concurrency)
    if not companions:
        return
    cfg = replace(cfg, isolate_programs=True)
    short = companion_config(cfg)
    tasks = companion_tasks(
        [
            truncate_context(
                load_task(cfg.sample_path(companion.sample)),
                tokenizer,
                companion.context_tokens,
            )
            for companion in companions
        ]
    )
    anchor_task = load_task(cfg.sample_path(cfg.samples[0]))
    anchor_path = cfg.decode_paths[0]
    print(_BAR)
    print(
        f"warmup: {anchor_task.id} again with {len(companions)} companion(s) per "
        f"stage, {short.gen_tokens} token(s) each, discarded and not summarised"
    )
    step_logger.open_window("warmup-concurrent")
    started = time.perf_counter()

    async def wave(index: int) -> list[Any]:
        return list(
            await asyncio.gather(
                *(
                    program(
                        llm,
                        tokenizer,
                        task,
                        short,
                        companion.decode_path,
                        program_id=concurrent_warmup_id(
                            task.id, companion.decode_path, index
                        ),
                        run_id=run_id,
                        concurrency=concurrency,
                        program_index=companion.program_index,
                        **media_kwargs(media, task),
                    )
                    for companion, task in zip(companions, tasks)
                )
            )
        )

    trigger = WaveTrigger(cfg, wave)
    try:
        anchor = await program(
            llm,
            tokenizer,
            anchor_task,
            cfg,
            anchor_path,
            program_id=concurrent_warmup_id(anchor_task.id, anchor_path, None),
            run_id=run_id,
            concurrency=concurrency,
            controller=trigger,
            **media_kwargs(media, anchor_task),
        )
        waves = await asyncio.gather(*trigger.waves)
    except BaseException:
        await trigger.cancel()
        raise
    print(
        f"{anchor.program_id}   e2e {anchor.e2e_seconds:.1f}s, "
        f"{len(waves)} wave(s) of {len(companions)}, "
        f"{time.perf_counter() - started:.1f}s"
    )


def merge_section(summary_path: Path, name: str, record: dict[str, Any]) -> Path:
    """Put one more section into `summary.json`, after the fingerprint."""
    payload = json.loads(summary_path.read_text())
    merged: dict[str, Any] = {}
    for key, value in payload.items():
        if key == name:
            continue
        merged[key] = value
        if key == "fingerprint":
            merged[name] = record
    merged.setdefault(name, record)
    return write_json(summary_path, merged)


async def execute(cfg: RunConfig) -> int:
    """Start the engine or engines this run asks for, and run the pass.

    P/D goes first: its workers must pin devices before this process imports vLLM.
    """
    if cfg.pd is not None and cfg.pd.enabled:
        from turboreflect.locality_pd.session import execute_pd

        return await execute_pd(cfg)

    if cfg.max_num_seqs is None:
        cfg = replace(cfg, max_num_seqs=await derive_max_num_seqs(cfg))

    from turboreflect.engine.engine import (
        build_engine,
        gpu_memory_note,
        kv_pool_report,
    )
    from turboreflect.metrics.observe import StepLogger
    from turboreflect.metrics.results import PassResult, ResultWriter
    from turboreflect.program.pipeline import run_program

    print(gpu_memory_note())
    # The engine trace writes into the run directory, so it must exist first.
    writer = ResultWriter(cfg)
    step_logger = StepLogger()
    llm = build_engine(cfg, writer.dir, stat_loggers=[step_logger.as_factory()])
    try:
        report = await kv_pool_report(llm)
        print(_BAR)
        for line in report.lines():
            print(line)
        print(f"prefix caching: {'on' if cfg.enable_prefix_caching else 'off'}")
        print(_BAR)

        tokenizer = llm.get_tokenizer()
        tasks = [load_task(cfg.sample_path(sample)) for sample in cfg.samples]
        writer.write_run_metadata(kv_pool=report.as_dict())
        writer.write_provenance(provenance.build(llm.vllm_config))
        print(f"results -> {writer.dir}")

        media = media_module.load_media(cfg, tasks, llm.vllm_config.model_config)
        if media:
            print(f"media: decoded the clips of {len(media)} sample(s)")

        await warm_up(
            llm,
            tokenizer,
            cfg,
            step_logger,
            writer.run_id,
            concurrency=warmup_concurrency(cfg),
            media=media,
        )

        jobs = program_jobs(cfg, tasks)

        n = cfg.concurrency
        step_logger.open_window(f"n{n}")
        batch = PassResult(concurrency=n)
        program_ids = [cfg.program_id(task.id, path, n) for task, path in jobs]
        epoch = time.perf_counter()
        controller = AdmissionController(cfg, concurrency=n)

        async def one(program_index, job):
            task, decode_path = job
            result = await admitted(
                controller,
                lambda wait: run_program(
                    llm,
                    tokenizer,
                    task,
                    cfg,
                    decode_path,
                    program_id=program_ids[program_index],
                    run_id=writer.run_id,
                    run_epoch=epoch,
                    concurrency=n,
                    program_index=program_index,
                    controller=controller,
                    first_admission_wait_s=wait,
                    **media_kwargs(media, task),
                ),
            )
            # Written at once, so an interrupted pass keeps every finished program.
            path = writer.write_program(result)
            print(f"  wrote {path.name}   e2e {result.e2e_seconds:.1f}s")
            return result

        print(_BAR)
        print(f"pass n={n}: {len(jobs)} programs")
        batch.programs = list(
            await asyncio.gather(*(one(i, j) for i, j in enumerate(jobs)))
        )
        batch.makespan_s = time.perf_counter() - epoch
        batch.engine = step_logger.window.as_dict()

        for result in batch.programs:
            print(_BAR)
            print(f"{result.program_id}   e2e {result.e2e_seconds:.1f}s")
            for stage, metrics in result.stages.items():
                print("  " + metrics.line())

        print(_BAR)
        print(
            f"pass n={n}: makespan {batch.makespan_s:.1f}s, sum of e2e "
            f"{sum(p.e2e_seconds for p in batch.programs):.1f}s, "
            f"peak kv {batch.engine['peak_kv_cache_usage']:.2f}"
        )
    finally:
        llm.shutdown()

    totals = step_logger.run.as_dict()
    summary = writer.write_summary(batch, totals, report.as_dict())
    print(_BAR)
    print(f"summary -> {summary}")
    metrics = write_metrics_safely(writer.dir)
    if metrics is not None:
        print(f"metrics -> {metrics}")
    print("done")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m turboreflect run")
    RunConfig.add_arguments(parser)
    parser.add_argument(
        "--dry-run",
        nargs="?",
        const="tokens",
        choices=("tokens", "text"),
        default=None,
        help="size the prompts with only a tokenizer loaded, then exit",
    )
    args = parser.parse_args(argv)
    try:
        cfg = RunConfig.from_namespace(args)
    except ConfigError as exc:
        parser.error(str(exc))

    # Set in the parent before any engine starts, so the engine core inherits what the
    # config states rather than what the shell held.
    for name, value in cfg.env_map().items():
        if os.environ.get(name) != value:
            print(f"env {name}={value}")
        os.environ[name] = value

    if args.dry_run:
        return dry_run(cfg, args.dry_run)
    return asyncio.run(execute(cfg))

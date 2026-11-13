# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors
"""Deployment calibration: measure a deployment's own step cost.

    python -m turboreflect.calibration.step_time --config RUN.json --out DIR

Starts the engine a RunConfig describes, without its `adaptive_draft` block, and
times decode and verify steps of B requests with distinct prompts, each request's
suffix draft cut or padded to the row count the step verifies
(`StepTimeScheduler`). A step of B requests at n rows costs

    T(B, n, ctx) = L(B, B (n - 1)) + B * ctx * A(B, n)

with L the context-free cost and A the seconds per context token per request.
Each uniform (B, n) timed at two contexts gives A from the difference and L from
the rest; c_ver is L's jump from no drafts to one. Steps where one to `SWITCH`
requests draft among one-row ones give `KneeStepModel.calibrated_tax`; lone steps
(one request at n rows, the rest at one) either side of each jump of A(B, n) give
`KneeStepModel.calibrated_edges`; ragged steps, priced with the tables, give per
step class the median measured over priced seconds (`ratio_priors`) that the
online correction (`step_cost.ClassRatios`) starts from.

Each point is the median of `MEASURE_STEPS` steps after `WARMUP_STEPS`, with B over
`BATCHES` and max_num_seqs, n over `ROWS` up to the draft limit plus one, and the
most context the KV pool fits per request and a quarter of it. `DIR/step_model.json`
is the step model (`step_cost.load`), `DIR/calibration.json` the points and fitted
terms, and `DIR/steps.jsonl` every timed step.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import regex as re

from turboreflect.adaptive_draft.step_cost import (
    TILE_STEP,
    KneeStepModel,
    coarse_class,
    step_class,
)
from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity, kv_pool_layout
from vllm.v1.core.sched.scheduler import Scheduler

ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 17, 21, 25, 33)
BATCHES = (1, 4, 8, 16)
RAGGED = 3
WARMUP_STEPS = 2
MEASURE_STEPS = 10
# Steps per batch where 1 .. SWITCH requests draft among one-row requests.
SWITCH = 3
PREFIX = "calibration"
_ID = re.compile(rf"{PREFIX}-(c\d+)-b(\d+)-")
TEXT = (
    "The archive keeps every draft, every critique and every revision of the "
    "report, in order, so the next reader can follow how it changed. "
)
# Share of the KV pool the calibration's requests may fill at once.
KV_SHARE = 0.9


class CalibrationError(RuntimeError):
    """This deployment could not be measured."""


def fit_context(capacity: dict[str, Any], batch: int) -> int:
    """Tokens each of `batch` requests can hold at once within `KV_SHARE` of the pool.

    Net of every request's fixed state blocks (`kv_pool_layout`).
    """
    layout = capacity.get("layout")
    if not layout:
        return int(KV_SHARE * capacity["kv_pool_tokens"] / batch)
    per = KV_SHARE * layout["num_blocks"] / batch - layout["fixed_blocks_per_sequence"]
    return int(per / layout["token_groups"]) * capacity["block_size"]


def shapes(
    rows: list[int], ragged: int, batch: int, switch: int = 0
) -> list[tuple[int, ...]]:
    """The row counts of a cohort's steps, request by request.

    Every n for all, then (for B > 1) `ragged` mixes and up to `switch` steps
    where k = 1, 2, ... requests draft among one-row ones.
    """
    out = [(n,) * batch for n in rows]
    if batch > 1:
        out += [
            tuple(
                rows[(i * len(rows) // batch + 3 * j) % len(rows)] for i in range(batch)
            )
            for j in range(ragged)
        ]
        drafting = rows[1:-1] or rows[1:]
        for k in range(1, min(switch, batch - 1) + 1):
            n = drafting[(k - 1) * (len(drafting) - 1) // max(switch - 1, 1)]
            out.append((n,) * k + (1,) * (batch - k))
    return out


def lone_rows(staircase: list[tuple[int, float]] | tuple) -> list[int]:
    """The row counts of a batch's lone steps.

    Either side of each jump of the uniform staircase A(B, n) past the first tile
    by more than `TILE_STEP`, plus the count after the lower side.
    """
    out: set[int] = set()
    for (n0, a0), (n1, a1) in zip(staircase, staircase[1:]):
        if n0 > 1 and a1 > TILE_STEP * a0:
            out |= {n0, min(n0 + 1, n1), n1}
    return sorted(out)


class StepTimeScheduler(Scheduler):
    """vLLM's scheduler, forcing every calibration request's draft to its row count.

    A cohort starts once a phase's B requests (named in their ids) all decode and
    nothing waits, runs `shapes` and then, at its batch's second context, the lone
    steps, `warmup + measure` steps each, appending every timed step to `records`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        plan = (self.vllm_config.additional_config or {})["step_calibration"]
        self._rows: list[int] = list(plan["rows"])
        self._ragged: int = plan["ragged"]
        self._switch: int = plan.get("switch", 0)
        self._lone: bool = plan.get("lone", False)
        self._recorded: list[dict[str, Any]] = []
        self._extended = False
        self._warmup: int = plan["warmup"]
        self._per = plan["warmup"] + plan["measure"]
        self._records = Path(plan["records"])
        # vLLM's token capacity nets a hybrid model's state blocks only for
        # max_concurrency sequences, so the driver sizes from the pool's blocks.
        tokens, concurrency = get_kv_cache_capacity(
            self.vllm_config, self.kv_cache_config
        )
        Path(plan["capacity"]).write_text(
            json.dumps(
                {
                    "kv_pool_tokens": tokens,
                    "max_concurrency": concurrency,
                    "block_size": self.block_size,
                    "layout": kv_pool_layout(self.kv_cache_config),
                }
            )
        )
        self._cohort: list[str] = []
        self._shapes: list[tuple[int, ...]] = []
        self._phase = ""
        self._step = 0
        self._pending: dict[str, Any] | None = None

    def _start(self) -> None:
        found = [_ID.match(r.request_id) for r in self.running]
        if self.waiting or self.skipped_waiting or not found or not all(found):
            return
        phases = {(m.group(1), int(m.group(2))) for m in found if m}
        if len(phases) != 1 or any(r.is_prefill_chunk for r in self.running):
            return
        ((context, batch),) = phases
        if len(self.running) == batch:
            self._cohort = [r.request_id for r in self.running]
            self._shapes = shapes(self._rows, self._ragged, batch, self._switch)
            self._phase = f"{context}-b{batch}"
            self._step = 0
            self._extended = False

    def _lone_shapes(self) -> list[tuple[int, ...]]:
        """The cohort's lone steps, once both of its batch's contexts are in."""
        batch = len(self._cohort)
        records = [r for r in self._recorded if r["batch"] == batch]
        if not self._lone or len({r["phase"] for r in records}) != 2:
            return []
        try:
            model, _ = tables(points(records), self._rows)
        except CalibrationError:
            return []
        lone = [(n,) + (1,) * (batch - 1) for n in lone_rows(model.calibrated_a[0][1])]
        return [s for s in lone if s not in self._shapes]

    def _force(self) -> tuple[int, ...] | None:
        requests = [self.requests.get(i) for i in self._cohort]
        index = self._step // self._per
        if index >= len(self._shapes) and not self._extended:
            self._extended = True
            self._shapes += self._lone_shapes()
        if any(r is None or r.is_finished() for r in requests) or index >= len(
            self._shapes
        ):
            for r in requests:
                if r is not None and not r.is_finished():
                    r.spec_token_ids = []
                    r.max_tokens = min(r.max_tokens, r.num_output_tokens + 1)
            self._cohort = []
            return None
        shape = self._shapes[index]
        for r, rows in zip(requests, shape):
            assert r is not None
            draft = list(r.spec_token_ids)[: rows - 1]
            pad = draft[-1] if draft else r.all_token_ids[-1]
            r.spec_token_ids = draft + [pad] * (rows - 1 - len(draft))
        return shape

    def schedule(self, *args, **kwargs):
        started = time.monotonic()
        if not self._cohort:
            self._start()
        shape = self._force() if self._cohort else None
        if not self._cohort:
            for r in self.running:
                if _ID.match(r.request_id):
                    r.spec_token_ids = []
        contexts = [self.requests[i].num_computed_tokens for i in self._cohort]
        output = super().schedule(*args, **kwargs)
        self._pending = None
        if shape is not None:
            self._pending = {
                "started": started,
                "shape": shape,
                "ids": list(self._cohort),
                "contexts": contexts,
            }
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        pending, self._pending = self._pending, None
        if pending is None:
            return outputs
        shape, ids = pending["shape"], pending["ids"]
        scheduled = scheduler_output.num_scheduled_tokens
        drafts = scheduler_output.scheduled_spec_decode_tokens
        exact = set(scheduled) == set(ids) and all(
            scheduled[i] == n and len(drafts.get(i, ())) == n - 1
            for i, n in zip(ids, shape)
        )
        record = {
            "phase": self._phase,
            "batch": len(ids),
            "shape": list(shape),
            "contexts": pending["contexts"],
            "seconds": time.monotonic() - pending["started"],
            "measured": self._step % self._per >= self._warmup,
            "exact": exact,
        }
        with self._records.open("a") as f:
            f.write(json.dumps(record) + "\n")
        self._recorded.append(record)
        self._step += 1
        return outputs


def points(records: list[dict[str, Any]]) -> dict[tuple, dict]:
    """Median seconds and mean contexts by (phase, shape), over exact steps."""
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for r in records:
        if r["measured"]:
            grouped.setdefault((r["phase"], tuple(r["shape"])), []).append(r)
    out = {}
    for key, steps in grouped.items():
        exact = [s for s in steps if s["exact"]]
        if not exact:
            continue
        out[key] = {
            "seconds": statistics.median(s["seconds"] for s in exact),
            "contexts": [
                statistics.fmean(c) for c in zip(*(s["contexts"] for s in exact))
            ],
            "steps": len(exact),
        }
    return out


def _context(phase: str) -> int:
    return int(phase.split("-")[0][1:])


def tables(
    measured: dict[tuple, dict], rows: list[int]
) -> tuple[KneeStepModel, list[dict[str, Any]]]:
    """L(B, drafts) and A(B, n) from each batch's uniform shapes at its two contexts.

    A is held non-decreasing in n except at the last of more than two counts,
    which runs on its own CUDA graph (`step_cost.table_rows`). L is kept as
    measured, floored at 0.

    Raises:
        CalibrationError: If a point is missing.
    """
    by_batch: dict[int, set[str]] = {}
    for phase, shape in measured:
        by_batch.setdefault(len(shape), set()).add(phase)
    calibrated_l, calibrated_a, report = [], [], []
    for b in sorted(by_batch):
        phases = sorted(by_batch[b], key=_context)
        if len(phases) != 2:
            raise CalibrationError(
                f"batch {b}: measured at {len(phases)} contexts, not 2"
            )
        low, high = phases
        ls: list[tuple[int, float]] = []
        as_: list[tuple[int, float]] = []
        for n in rows:
            try:
                p, q = measured[(low, (n,) * b)], measured[(high, (n,) * b)]
            except KeyError as exc:
                raise CalibrationError(f"batch {b}, {n} rows: not measured") from exc
            cp, cq = statistics.fmean(p["contexts"]), statistics.fmean(q["contexts"])
            a = (q["seconds"] - p["seconds"]) / (b * (cq - cp))
            alone = len(rows) > 2 and n == rows[-1]
            a = max(a, 0.0 if alone or not as_ else as_[-1][1])
            el = max(p["seconds"] - b * cp * a, 0.0)
            ls.append((b * (n - 1), el))
            as_.append((n, a))
            report.append({"batch": b, "rows": n, "low": p, "high": q, "L": el, "A": a})
        calibrated_l.append((b, tuple(ls)))
        calibrated_a.append((b, tuple(as_)))
    model = KneeStepModel(
        c0=0.0,
        c_row=0.0,
        w_dense=0.0,
        w_moe=0.0,
        moe_p=0.0,
        lin_knee=1.0,
        beta=0.0,
        attn_knee=1.0,
        calibrated_l=tuple(calibrated_l),
        calibrated_a=tuple(calibrated_a),
    )
    return model, report


def switch_tax(
    model: KneeStepModel, measured: dict[tuple, dict]
) -> tuple[tuple[int, float], ...]:
    """Per batch, the seconds per context token a one-row request adds.

    The median over the switch steps of the step's growth between the two
    contexts beyond the tables' price and launch edges, over the one-row rows'
    context growth, floored at 0 (`KneeStepModel.calibrated_tax`).
    """
    by_batch: dict[int, list[float]] = {}
    for (phase, shape), p in measured.items():
        k = len(shape) - shape.count(1)
        if not 0 < k < len(shape) or len(set(shape[:k])) != 1 or 1 in shape[:k]:
            continue
        b = len(shape)
        other = [
            (ph, q)
            for (ph, sh), q in measured.items()
            if sh == shape and ph != phase and len(sh) == b
        ]
        if len(other) != 1 or _context(other[0][0]) <= _context(phase):
            continue
        q = other[0][1]
        grown = [c1 - c0 for c0, c1 in zip(p["contexts"], q["contexts"])]
        drafting = sum(model.attn_s(d, n, b) for n, d in zip(shape, grown) if n > 1)
        ones = [d for n, d in zip(shape, grown) if n == 1]
        if sum(ones) <= 0:
            continue
        tax = (q["seconds"] - p["seconds"] - drafting) / sum(ones) - (
            model.attn_s(1.0, 1, b) + model.launch_tax(b, shape[0])
        )
        by_batch.setdefault(b, []).append(tax)
    return tuple(
        (b, max(0.0, statistics.median(v))) for b, v in sorted(by_batch.items())
    )


def edges(
    model: KneeStepModel, measured: dict[tuple, dict]
) -> tuple[KneeStepModel, list[dict[str, Any]]]:
    """Place and classify every batch's edges from its lone steps (`lone_rows`).

    An edge is a jump of A(B, n) past the first tile by more than `TILE_STEP`;
    where measured counts skip rows, the lone step one row past the lower side
    places it. If, across it, the lone steps' one-row requests add more than half
    the jump per context token beyond the tables' price, the launch is set by the
    longest query and the edge goes into `calibrated_edges` at the jump;
    otherwise only the request's own tile opens and it gets 0, as does an
    `unclassified` edge no lone step measured.
    """
    calibrated_a, listed, report = [], [], []
    for b, staircase in model.calibrated_a:
        phase = max((ph for ph, sh in measured if len(sh) == b), key=_context)
        lone = {
            sh[0]: p
            for (ph, sh), p in measured.items()
            if ph == phase and len(sh) == b and sh[1:] == (1,) * (b - 1)
        }
        a = dict(staircase)
        found: list[tuple[int, float]] = []
        for (n0, a0), (n1, a1) in zip(staircase, staircase[1:]):
            if n0 < 2 or a1 <= TILE_STEP * a0:
                continue
            entry: dict[str, Any] = {"batch": b, "rows": n0, "next": n1}
            if n1 > n0 + 1 and {n0, n0 + 1, n1} <= set(lone):
                t0, t1, t2 = (lone[n]["seconds"] for n in (n0, n0 + 1, n1))
                own = lone[n1]["contexts"][0] * (a1 - a0)
                if t2 - t0 > own / 2 and t1 - t0 < (t2 - t0) / 2:
                    a[n0 + 1] = a0
                    entry["rows"] = n0 + 1
            e = entry["rows"]
            hi = e + 1 if e + 1 in lone else n1
            jump, launch, kind = a1 - a0, 0.0, "request"
            if b > 1 and not {e, hi} <= set(lone):
                kind = "unclassified"
            elif b > 1:
                placed = dataclasses.replace(
                    model,
                    calibrated_a=((b, tuple(sorted(a.items()))),),
                    calibrated_edges=((b, ()),),
                )
                priced, ones = [], 0.0
                for n in (e, hi):
                    rows = [(n, round(c), False) for c in lone[n]["contexts"][:1]]
                    rows += [(1, round(c), False) for c in lone[n]["contexts"][1:]]
                    priced.append(lone[n]["seconds"] - placed.price(rows))
                    ones = sum(lone[n]["contexts"][1:])
                extra = (priced[1] - priced[0]) / ones
                launch = jump if extra > jump / 2 else 0.0
                kind = "launch" if launch else "request"
                entry.update(lone=[e, hi], extra=extra, jump=jump)
            entry["kind"] = kind
            found.append((e, launch))
            report.append(entry)
        calibrated_a.append((b, tuple(sorted(a.items()))))
        if b > 1:
            listed.append((b, tuple(found)))
    placed = dataclasses.replace(
        model, calibrated_a=tuple(calibrated_a), calibrated_edges=tuple(listed)
    )
    return placed, report


def validate(model: KneeStepModel, measured: dict[tuple, dict]) -> list[dict]:
    """The ragged shapes' measured seconds and the tables' price, by step class."""
    out = []
    for (_, shape), p in measured.items():
        if len(set(shape)) > 1:
            rows = [(n, round(c), False) for n, c in zip(shape, p["contexts"])]
            out.append(
                {
                    "class": coarse_class(step_class(rows)),
                    "seconds": p["seconds"],
                    "priced": model.price(rows),
                }
            )
    return out


def ragged_ratios(ragged: list[dict]) -> dict[str, float]:
    """Each step class's median measured over priced seconds of the ragged steps."""
    by_class: dict[str, list[float]] = {}
    for v in ragged:
        if v["priced"] > 0:
            by_class.setdefault(v["class"], []).append(v["seconds"] / v["priced"])
    return {k: statistics.median(r) for k, r in sorted(by_class.items())}


async def _generate(
    llm, cfg, index: int, ids: list[int], max_tokens: int, request_id: str
) -> None:
    """One request sampling as program `index` would, which sets the verify cost."""
    from turboreflect.program.pipeline import sampling_draw
    from vllm import SamplingParams

    params = SamplingParams(
        max_tokens=max_tokens, ignore_eos=True, **sampling_draw(cfg, index)
    )
    async for _ in llm.generate({"prompt_token_ids": ids}, params, request_id):
        pass


async def measure_steps(cfg, out: Path, warmup: int, measure: int) -> dict[str, Any]:
    """Run deployment calibration for `cfg` into `out` and return its summary.

    Raises:
        CalibrationError: If `cfg` cannot be measured or a point is missing.
    """
    from turboreflect.engine.engine import build_engine_args
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM

    if not cfg.suffix_decoding:
        raise CalibrationError("calibration times verification: enable suffix_decoding")
    if cfg.max_num_seqs is None:
        raise CalibrationError("calibration times batches up to max_num_seqs: state it")
    k = cfg.suffix_num_speculative_tokens
    rows = sorted({n for n in ROWS if n <= k + 1} | {k + 1})
    batches = sorted({b for b in (*BATCHES, cfg.max_num_seqs) if b <= cfg.max_num_seqs})
    budget = (warmup + measure) * (
        sum(rows) + sum(range(2, rows[-1] + 1)) + (RAGGED + SWITCH) * rows[-1]
    )
    out.mkdir(parents=True, exist_ok=True)
    records = out / "steps.jsonl"
    records.unlink(missing_ok=True)
    capacity = out / "capacity.json"
    capacity.unlink(missing_ok=True)
    engine_args = build_engine_args(cfg)
    engine_args.scheduler_cls = "turboreflect.calibration.step_time.StepTimeScheduler"
    engine_args.additional_config = {
        **(engine_args.additional_config or {}),
        "step_calibration": {
            "rows": rows,
            "ragged": RAGGED,
            "switch": SWITCH,
            "lone": True,
            "warmup": warmup,
            "measure": measure,
            "records": str(records),
            "capacity": str(capacity),
        },
    }
    started = time.monotonic()
    llm = AsyncLLM.from_engine_args(
        engine_args, usage_context=UsageContext.ENGINE_CONTEXT
    )
    try:
        fit = json.loads(capacity.read_text())
        pool = fit["kv_pool_tokens"]
        phases = []
        for b in batches:
            high = min(cfg.max_model_len, fit_context(fit, b)) - budget - 1024
            high = high // 256 * 256
            low = high // 4 // 256 * 256
            if low < 1024:
                raise CalibrationError(
                    f"batch {b}: the KV pool ({pool} tokens) is too small"
                )
            phases += [(low, b), (high, b)]
        tokenizer = llm.get_tokenizer()
        unit = tokenizer.encode(TEXT, add_special_tokens=False)
        # Ascending context: each prompt extends its shorter phase's, so a later
        # phase reads that part from the prefix cache.
        for c, b in sorted(phases):
            tail = budget + math.ceil(b * c / cfg.max_num_batched_tokens) + 64
            prompts = []
            for i in range(b):
                head = tokenizer.encode(f"Record {i:04d}. ", add_special_tokens=False)
                prompts.append((head + unit * (c // len(unit) + 1))[:c])
            await asyncio.gather(
                *(
                    _generate(llm, cfg, i, ids, tail, f"{PREFIX}-c{c}-b{b}-{i}")
                    for i, ids in enumerate(prompts)
                )
            )
    finally:
        llm.shutdown()
    lines = records.read_text().splitlines() if records.exists() else []
    measured = points([json.loads(x) for x in lines])
    run = {
        "config": {
            "model": cfg.model,
            "kv_cache_dtype": cfg.kv_cache_dtype,
            "max_num_seqs": cfg.max_num_seqs,
            "max_model_len": cfg.max_model_len,
            "num_speculative_tokens": k,
            "cudagraph_mode": cfg.cudagraph_mode,
            "env": cfg.env_map(),
        },
        "kv_pool_tokens": pool,
        "phases": sorted(phases),
        "rows": rows,
        "warmup": warmup,
        "measure": measure,
        "seconds": time.monotonic() - started,
    }
    return write_model(measured, rows, out, run)


def write_model(
    measured: dict[tuple, dict], rows: list[int], out: Path, run: dict[str, Any]
) -> dict[str, Any]:
    """Fit the step model, write `calibration.json` and `step_model.json` to `out`."""
    model, report = tables(measured, rows)
    model, found = edges(model, measured)
    model = dataclasses.replace(model, calibrated_tax=switch_tax(model, measured))
    # The lone steps, at one context, place and classify the edges; the
    # ragged and switch steps, at both, set the ratio priors.
    both = Counter(shape for _, shape in measured)
    mixed = {k: p for k, p in measured.items() if both[k[1]] > 1}
    model = dataclasses.replace(
        model, ratio_priors=ragged_ratios(validate(model, mixed))
    )
    summary = {
        **run,
        "points": report,
        "switch_tax": model.calibrated_tax,
        "edges": found,
        "ratio_priors": model.ratio_priors,
    }
    (out / "calibration.json").write_text(json.dumps(summary, indent=1))
    (out / "step_model.json").write_text(
        json.dumps({"verify_cost": model.to_dict()}, indent=1)
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m turboreflect.calibration.step_time"
    )
    parser.add_argument("--config", required=True, help="the run's RunConfig JSON")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    import tempfile

    from turboreflect.config import RunConfig

    payload = json.loads(Path(args.config).read_text())
    payload.pop("adaptive_draft", None)
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        json.dump(payload, f)
        f.flush()
        cfg = RunConfig.from_namespace(argparse.Namespace(config=f.name))
    for name, value in cfg.env_map().items():
        os.environ[name] = value
    if cfg.max_num_seqs is None:
        from turboreflect.engine.runner import derive_max_num_seqs

        cfg = dataclasses.replace(
            cfg, max_num_seqs=asyncio.run(derive_max_num_seqs(cfg))
        )
    try:
        summary = asyncio.run(measure_steps(cfg, args.out, WARMUP_STEPS, MEASURE_STEPS))
    except CalibrationError as exc:
        print(f"calibration failed: {exc}", file=sys.stderr)
        return 2
    print(f"calibration: {args.out / 'step_model.json'} in {summary['seconds']:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

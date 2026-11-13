# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""Result files: one directory per run, one JSON per program, one summary.

The directory is `{timestamp}-{fingerprint}` unless a run id is given. Program files
hold prompt sizes, not prompt text, which the sample id rebuilds.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import regex as re

from turboreflect.config import RunConfig
from turboreflect.locality_pd.config import ROLE_D, ROLE_P
from turboreflect.metrics.metrics import StageMetrics

STAGE_OUTPUT_NAMES = {
    "generation": "draft",
    "feedback": "critique",
    "refinement": "revision",
}

# The JSON files in a run directory that are not one program's result.
RUN_FILES = (
    "config.json",
    "summary.json",
    "provenance.json",
)

# Kept apart from `program.prompt` so reading a summary loads no bench module.
_ROUND_SUFFIX = re.compile(r"^(generation|feedback|refinement)(?:_(\d+))?$")


def round_output_name(stage: str) -> str:
    """What one stage of the round loop writes its generation under.

    `generation` -> `draft`, `feedback_2` -> `critique_2`, `refinement_1` ->
    `revision_1`. Scoring reads `draft` and `revision`, which the driver sets to
    what the program delivered.

    Raises:
        KeyError: On a stage name that is not one of the three, numbered or not.
    """
    match = _ROUND_SUFFIX.match(stage)
    if match is None:
        raise KeyError(stage)
    base, round_index = match.group(1), match.group(2)
    name = STAGE_OUTPUT_NAMES[base]
    return name if round_index is None else f"{name}_{int(round_index)}"


def final_artifact(stages: Sequence[str], outputs: Mapping[str, str]) -> str:
    """The one text a program delivered: the output of the last stage it ran.

    Raises:
        KeyError: If the last stage that ran wrote no output.
    """
    return outputs[STAGE_OUTPUT_NAMES[stages[-1]]]


def write_json(path: Path, payload: Any) -> Path:
    """Write `payload` to `path` atomically: beside it first, then renamed."""
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    tmp = path.parent / f"{path.name}.tmp"
    tmp.write_text(text)
    tmp.replace(path)
    return path


def percentile(values: Sequence[float], q: float) -> float | None:
    """The q-th percentile, q in [0, 100], interpolated as numpy does; None if empty."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def prefill_counts(metrics: StageMetrics) -> dict[str, Any]:
    """The prompt as the engine that prefilled it saw it.

    That engine is P on a split stage, D for a stage D ran alone, else the one
    engine. What its KV connector loaded (a P's pull from D) is
    `prefill_external_tokens`, apart from its own prefix-cache hits.
    """
    if metrics.p_cached_tokens is not None:
        external = metrics.p_pulled_tokens or 0
        served = metrics.p_cached_tokens
        computed = metrics.prompt_tokens - served
        role = ROLE_P
    else:
        external = metrics.external_cached_tokens or 0
        served = metrics.cached_tokens
        computed = metrics.computed_tokens
        role = ROLE_D if metrics.engine else None
    counts: dict[str, Any] = {
        "prefill_cached_tokens": served - external,
        "prefill_external_tokens": external,
        "prefill_computed_tokens": computed,
    }
    if role is not None:
        counts["prefill_role"] = role
    return counts


@dataclass
class ProgramResult:
    """One generation-feedback-refinement run over one sample."""

    program_id: str
    sample_id: str
    decode_path: str
    concurrency: int = 1
    # The one text this program delivered.
    artifact: str = ""
    # Feedback stages the round loop ran; 0, and not written, in a fixed pipeline.
    rounds_run: int = 0
    e2e_seconds: float = 0.0
    t_start_s: float = 0.0
    t_end_s: float = 0.0
    stages: dict[str, StageMetrics] = field(default_factory=dict)
    # Stage intervals on the run-wide clock.
    spans: dict[str, tuple[float, float]] = field(default_factory=dict)
    priorities: dict[str, int] = field(default_factory=dict)
    method_decisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Per stage, merged into its row: how its prompt continued the one before
    # (`join_record`), and what D offered its P to pull and P recomputed.
    joins: dict[str, dict[str, Any]] = field(default_factory=dict)
    pulls: dict[str, dict[str, Any]] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    # `outputs` before detokenisation, which is not one-to-one.
    output_token_ids: dict[str, list[int]] = field(default_factory=dict)
    # Reasoning stripped from `outputs` with thinking off; written only if any.
    reasoning: dict[str, str] = field(default_factory=dict)
    # Per-stage generation caps, when `refinement_gen_tokens` sets its own.
    max_tokens: dict[str, int] = field(default_factory=dict)
    prompt_shapes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # None, and not written, for a greedy program.
    seed: int | None = None

    def stage_entry(self, stage: str) -> dict[str, Any]:
        """One stage's row: its timings and its counts.

        Per request on the engine-core clock: `tpot_s` is `decode_s_per_token`,
        `e2e_s` is `StageMetrics.e2e_s`, and `queue_excluded_e2e_s` is
        `last_token_ts - scheduled_ts` (D's on a split stage), dropping the wait
        before first scheduling but not a preemption's re-queue. The reuse split
        sums to `prompt_tokens`: `cpu_restore_tokens` is what the KV connector
        supplied, `gpu_hit_tokens` the rest of `cached_tokens`, and
        `recomputed_tokens` is `computed_tokens`.
        """
        metrics = self.stages[stage]
        entry = metrics.as_dict()
        if stage in self.method_decisions:
            entry["pd_method"] = self.method_decisions[stage]
        entry["hit_ratio"] = metrics.hit_ratio
        entry["tpot_s"] = metrics.decode_s_per_token
        entry["e2e_s"] = metrics.e2e_s
        entry["queue_excluded_e2e_s"] = metrics.last_token_ts - metrics.scheduled_ts
        external = metrics.external_cached_tokens
        entry["cpu_restore_tokens"] = external
        entry["gpu_hit_tokens"] = (
            None if external is None else metrics.cached_tokens - external
        )
        entry["recomputed_tokens"] = metrics.computed_tokens
        start, end = self.spans.get(stage, (0.0, 0.0))
        entry["t_start_s"] = start
        entry["t_end_s"] = end
        if stage in self.priorities:
            entry["priority"] = self.priorities[stage]
        if stage in self.max_tokens:
            entry["max_tokens"] = self.max_tokens[stage]
        entry.update(self.joins.get(stage, {}))
        entry.update(self.pulls.get(stage, {}))
        entry.update(prefill_counts(metrics))
        return entry

    def as_dict(self) -> dict[str, Any]:
        stages: dict[str, Any] = {}
        for stage in self.stages:
            entry = self.stage_entry(stage)
            entry["prompt_messages"] = self.prompt_shapes.get(stage, [])
            stages[stage] = entry
        payload: dict[str, Any] = {
            "program_id": self.program_id,
            "sample_id": self.sample_id,
            "decode_path": self.decode_path,
            "concurrency": self.concurrency,
            "artifact": self.artifact,
            "e2e_seconds": self.e2e_seconds,
            "t_start_s": self.t_start_s,
            "t_end_s": self.t_end_s,
            "stages": stages,
            "outputs": dict(self.outputs),
            "output_token_ids": {
                name: list(ids) for name, ids in self.output_token_ids.items()
            },
        }
        if self.rounds_run:
            payload["rounds_run"] = self.rounds_run
        if self.reasoning:
            payload["reasoning"] = dict(self.reasoning)
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload


@dataclass
class PassResult:
    """Every program of the run, plus what the engine saw."""

    concurrency: int
    makespan_s: float = 0.0
    programs: list[ProgramResult] = field(default_factory=list)
    engine: dict[str, Any] = field(default_factory=dict)

    def throughput(self) -> dict[str, Any]:
        """Tokens over the pass makespan, cache hits counted as prompt tokens."""
        stages = [m for p in self.programs for m in p.stages.values()]
        output = sum(m.gen_tokens for m in stages)
        prompt = sum(m.prompt_tokens for m in stages)
        span = self.makespan_s
        return {
            "output_tokens": output,
            "prompt_tokens": prompt,
            "total_tokens_per_s": (prompt + output) / span if span > 0 else 0.0,
        }

    def occupancy(self) -> dict[str, Any]:
        """Mean KV usage over the makespan, idle time counted as an empty pool."""
        span = self.makespan_s
        usage = self.engine.get("kv_usage_s", 0.0)
        return {"mean_kv_cache_usage_makespan": usage / span if span > 0 else 0.0}

    def as_dict(self) -> dict[str, Any]:
        return {
            "concurrency": self.concurrency,
            "makespan_s": self.makespan_s,
            "engine": dict(self.engine),
            "throughput": self.throughput(),
            "occupancy": self.occupancy(),
            "program_ids": [p.program_id for p in self.programs],
        }


class ResultWriter:
    """Owns one run directory."""

    def __init__(self, cfg: RunConfig, run_id: str | None = None) -> None:
        self.cfg = cfg
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_id = (
            run_id
            or os.environ.get("TURBOREFLECT_RUN_ID")
            or f"{stamp}-{cfg.fingerprint()[:8]}"
        )
        self.dir = Path(cfg.out_dir) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)

    def write_run_metadata(self, **extra: Any) -> Path:
        """Write the run's `config.json`, including anything measured at startup."""
        payload = {
            "run_id": self.run_id,
            "fingerprint": self.cfg.fingerprint(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config": self.cfg.as_dict(),
            **extra,
        }
        return write_json(self.dir / "config.json", payload)

    def write_provenance(self, record: dict[str, Any]) -> Path:
        """Write `provenance.json`: commit and engine facts, outside the fingerprint."""
        return write_json(self.dir / "provenance.json", record)

    def write_program(self, result: ProgramResult) -> Path:
        """Put one program on disk as it finishes, so an interrupted pass keeps it."""
        return write_json(self.dir / f"{result.program_id}.json", result.as_dict())

    def write_summary(
        self,
        batch: PassResult,
        engine_totals: dict[str, Any],
        kv_pool: dict[str, Any],
    ) -> Path:
        """Write `summary.json`: the run's totals and its pass."""
        payload = {
            "run_id": self.run_id,
            "fingerprint": self.cfg.fingerprint(),
            "enable_prefix_caching": self.cfg.enable_prefix_caching,
            "gpus": self.cfg.gpu_count(),
            "gpu_hour_usd": self.cfg.gpu_hour_usd,
            "kv_pool": kv_pool,
            "engine": engine_totals,
            "passes": [batch.as_dict()],
        }
        return write_json(self.dir / "summary.json", payload)

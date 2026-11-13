# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""One run directory, as the driver and the engine trace left it.

Nothing here imports vLLM or torch. A trace step is a pass's when all its request
ids (`<program_id>-<stage>` plus vLLM's `-<8 hex>`) are that pass's programs, and is
dropped otherwise (the warmup's). Without `summary.json`, a program that never
finished is placed by its id's `-n<concurrency>` suffix.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import regex as re

from turboreflect.metrics.results import RUN_FILES

STEPS_FILE = "engine_steps_core.jsonl"
REQUESTS_FILE = "engine_requests_core.jsonl"
SUMMARY_FILE = "summary.json"
METRICS_ERROR_FILE = "metrics_error.txt"

STAGES = ("generation", "feedback", "refinement")

# `turboreflect.metrics.trace.STEP_REQUEST_FIELDS`, for a file whose header is lost.
_REQUEST_FIELDS = (
    "request_id",
    "scheduled_tokens",
    "is_prefill",
    "context_tokens",
    "emitted_tokens",
)
_INTERNAL_SUFFIX = re.compile(r"-[0-9a-f]{8}$")
_LEVEL_SUFFIX = re.compile(r"-n(\d+)$")
_WARMUP_PREFIX = "warmup-"
_STAGE_ID = re.compile(
    r"^(?P<program>.+)-(?P<stage>(?:generation|feedback|refinement)(?:_\d+)?)$"
)


def read_json(path: Path) -> Any:
    """The file's JSON, or None when it is missing or unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Every object line of a JSONL file; a line cut off by a kill is skipped."""
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return
    with handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                yield row


def split_request_id(request_id: str) -> tuple[str, str] | None:
    """`(program_id, stage)` from an engine or frontend request id."""
    match = _STAGE_ID.match(_INTERNAL_SUFFIX.sub("", request_id))
    return None if match is None else (match["program"], match["stage"])


def base_stage(stage: str) -> str:
    """`feedback_2` -> `feedback`."""
    return stage.split("_", 1)[0]


@dataclass(frozen=True, slots=True)
class StepRequest:
    """One request's share of one engine step."""

    request_id: str
    program_id: str
    scheduled: int
    is_prefill: bool
    context: int
    emitted: int


@dataclass(frozen=True, slots=True)
class Step:
    """One engine step, on the engine core's monotonic clock."""

    step: int
    pass_label: str
    t_start: float
    t_end: float
    reqs: tuple[StepRequest, ...]

    @property
    def seconds(self) -> float:
        """Schedule start to update end."""
        return self.t_end - self.t_start

    @property
    def generating(self) -> tuple[StepRequest, ...]:
        return tuple(r for r in self.reqs if not r.is_prefill)


@dataclass
class Pass:
    """A run's pass: its concurrency and the programs it ran."""

    label: str
    concurrency: int | None
    record: dict[str, Any] | None
    program_ids: list[str]


class RunDir:
    """Everything one run directory holds, read once and on demand."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.config = read_json(self.path / "config.json")
        self.summary = read_json(self.path / SUMMARY_FILE)
        self.programs = self._read_programs()
        self.passes = self._build_passes()
        self._pass_of = {
            program_id: p.label for p in self.passes for program_id in p.program_ids
        }
        self._unfinished_pass = {
            p.concurrency: p.label for p in self.passes if p.record is None
        }
        self.steps_header: dict[str, Any] | None = None
        self._engine_steps: dict[str, list[Step] | None] = {}
        self._engine_headers: dict[str, dict[str, Any]] = {}
        self._engine_requests: dict[str, dict[str, dict[str, Any]] | None] = {}

    @property
    def run_id(self) -> str:
        for record in (self.summary, self.config):
            if isinstance(record, dict) and record.get("run_id"):
                return str(record["run_id"])
        return self.path.name

    def config_value(self, name: str) -> Any:
        config = (self.config or {}).get("config") if self.config else None
        return config.get(name) if isinstance(config, dict) else None

    @property
    def trace_dirs(self) -> dict[str, str]:
        """A split run's engine-trace directory per role; empty for one engine."""
        record = (self.summary or {}).get("engine_trace")
        dirs = record.get("dirs") if isinstance(record, dict) else None
        return dict(dirs) if isinstance(dirs, dict) else {}

    @property
    def engine_roles(self) -> dict[str, str]:
        """A split run's engines and their roles, as each reported itself."""
        record = (self.summary or {}).get("pd")
        engines = record.get("engines") if isinstance(record, dict) else None
        if not isinstance(engines, dict):
            return {}
        return {
            str(name): str(entry.get("role"))
            for name, entry in engines.items()
            if isinstance(entry, dict)
        }

    def kv_pool(self) -> dict[str, Any]:
        for record in (self.summary, self.config):
            pool = (record or {}).get("kv_pool") if isinstance(record, dict) else None
            if isinstance(pool, dict) and pool:
                return pool
        return {}

    def pass_of(self, program_id: str) -> str | None:
        """The pass that ran `program_id`; None for the warmup or an unknown id."""
        label = self._pass_of.get(program_id)
        if label is not None or program_id.startswith(_WARMUP_PREFIX):
            return label
        level = _LEVEL_SUFFIX.search(program_id)
        if level is not None:
            return self._unfinished_pass.get(int(level[1]))
        # The ids of a run that declares its program set name no concurrency.
        unfinished = list(self._unfinished_pass.values())
        if len(unfinished) == 1 and self.config_value("program_total") is not None:
            return unfinished[0]
        return None

    def _read_programs(self) -> dict[str, dict[str, Any]]:
        programs: dict[str, dict[str, Any]] = {}
        for path in sorted(self.path.glob("*.json")):
            if path.name in RUN_FILES:
                continue
            payload = read_json(path)
            if (
                isinstance(payload, dict)
                and "program_id" in payload
                and isinstance(payload.get("stages"), dict)
            ):
                programs[str(payload["program_id"])] = payload
        return programs

    def _build_passes(self) -> list[Pass]:
        passes: list[Pass] = []

        entries = (self.summary or {}).get("passes") if self.summary else None
        if isinstance(entries, list) and entries:
            for entry in entries:
                concurrency = entry.get("concurrency")
                ids = entry.get("program_ids")
                if not isinstance(ids, list):
                    ids = [
                        pid
                        for pid, program in self.programs.items()
                        if program.get("concurrency") == concurrency
                    ]
                passes.append(Pass(f"n{concurrency}", concurrency, entry, ids))
            return passes
        by_level: dict[Any, list[str]] = {}
        for program_id, program in self.programs.items():
            by_level.setdefault(program.get("concurrency"), []).append(program_id)
        for concurrency in sorted(by_level, key=lambda c: (c is None, c or 0)):
            passes.append(
                Pass(f"n{concurrency}", concurrency, None, by_level[concurrency])
            )
        return passes

    @cached_property
    def steps(self) -> list[Step] | None:
        """Measured steps in file order, or None without an engine trace."""
        return self._read_steps(self.path / STEPS_FILE, keep_header=True)

    def engine_steps(self, name: str) -> list[Step] | None:
        """One split-run engine's steps; None if its trace is not named or gone."""
        if name not in self._engine_steps:
            directory = self.trace_dirs.get(name)
            self._engine_steps[name] = (
                None
                if directory is None
                else self._read_steps(self.path / directory / STEPS_FILE, engine=name)
            )
        return self._engine_steps[name]

    def engine_header(self, name: str) -> dict[str, Any] | None:
        self.engine_steps(name)
        return self._engine_headers.get(name)

    def engine_requests(self, name: str) -> dict[str, dict[str, Any]] | None:
        """Native request rows scoped to one engine, never joined across P/D."""
        if name not in self._engine_requests:
            directory = self.trace_dirs.get(name)
            self._engine_requests[name] = (
                None
                if directory is None
                else self._read_requests(self.path / directory / REQUESTS_FILE)
            )
        return self._engine_requests[name]

    def _read_steps(
        self, path: Path, keep_header: bool = False, engine: str | None = None
    ) -> list[Step] | None:
        if not path.is_file():
            return None
        fields = {name: i for i, name in enumerate(_REQUEST_FIELDS)}
        steps: list[Step] = []
        for row in read_jsonl(path):
            if row.get("kind") == "header":
                if keep_header and self.steps_header is None:
                    self.steps_header = row
                if engine is not None:
                    self._engine_headers[engine] = row
                names = row.get("request_fields")
                if isinstance(names, list):
                    fields = {name: i for i, name in enumerate(names)}
                continue
            step = self._step(row, fields)
            if step is not None:
                steps.append(step)
        return steps

    def _step(self, row: dict[str, Any], fields: dict[str, int]) -> Step | None:
        entries = row.get("reqs")
        t_schedule, t_update = row.get("t_schedule"), row.get("t_update")
        if not entries or not t_schedule or not t_update:
            return None
        reqs = []
        labels = set()
        for entry in entries:
            req = _step_request(entry, fields)
            if req is None:
                return None
            label = self.pass_of(req.program_id)
            if label is None:
                return None
            labels.add(label)
            reqs.append(req)
        if len(labels) != 1:
            return None
        return Step(
            step=int(row.get("step", 0)),
            pass_label=labels.pop(),
            t_start=float(t_schedule[0]),
            t_end=float(t_update[-1]),
            reqs=tuple(reqs),
        )

    @cached_property
    def requests(self) -> dict[str, dict[str, Any]] | None:
        """Finished-request rows of the run's programs by external id, or None."""
        return self._read_requests(self.path / REQUESTS_FILE)

    def _read_requests(self, path: Path) -> dict[str, dict[str, Any]] | None:
        if not path.is_file():
            return None
        rows: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(path):
            if row.get("kind") == "header":
                continue
            rid = row.get("request_id")
            parts = split_request_id(str(rid)) if rid else None
            if parts is not None and self.pass_of(parts[0]) is not None:
                rows[_INTERNAL_SUFFIX.sub("", str(rid))] = row
        return rows


def _step_request(entry: Any, fields: dict[str, int]) -> StepRequest | None:
    if isinstance(entry, dict):
        get = entry.get
    elif isinstance(entry, list):

        def get(name: str) -> Any:
            index = fields.get(name)
            return entry[index] if index is not None and index < len(entry) else None
    else:
        return None
    rid = get("request_id")
    parts = split_request_id(str(rid)) if rid else None
    if parts is None:
        return None
    return StepRequest(
        request_id=str(rid),
        program_id=parts[0],
        scheduled=int(get("scheduled_tokens") or 0),
        is_prefill=bool(get("is_prefill")),
        context=int(get("context_tokens") or 0),
        emitted=int(get("emitted_tokens") or 0),
    )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""The metric set of one run, per pass; `summary.json` carries `results`.

    python -m turboreflect report RUN_DIR [RUN_DIR ...]

`results` holds the makespan (driver clock, gather start to end), cost (`cost`),
decode latency (`windowed_tpot`), mean program E2E, throughput, prefix hit ratio
(cached over prompt tokens), matched prefix and append lengths (`join_record`), draft
acceptance (`_speculation`), each engine's KV use (`PassResult.occupancy`) and P/D
handoff times (`stage_handoffs`). The CLI rewrites them for each directory given (a
directory that is not a run stands for every run under it) and prints one table row
per pass.

Sources, warmup excluded: `summary.json` for the makespan, throughput and KV use; the
program files for hit counts, matched prefix and append lengths and handoffs;
`engine_steps_core.jsonl` (one per engine in a split run) for decode latency;
`engine_requests_core.jsonl` for draft acceptance. A number that cannot be derived is
null.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import traceback
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from turboreflect.locality_pd.config import ROLE_D
from turboreflect.locality_pd.policy import TPOT_WINDOW_S
from turboreflect.metrics.results import percentile, write_json
from turboreflect.metrics.rundir import (
    METRICS_ERROR_FILE,
    STAGES,
    SUMMARY_FILE,
    Pass,
    RunDir,
    Step,
    base_stage,
    read_json,
    split_request_id,
)

# Regressors of the step cost model: work terms of one engine step.
COST_TERMS = (
    "const",
    "prefill_tokens",
    "prefill_pairs",
    "prefill_context_tokens",
    "decode_requests",
    "decode_kv_tokens",
    "decode_query_tokens",
    "verify_pairs",
)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value


def _present(values: Iterable[Any]) -> list[float]:
    return [v for v in map(_num, values) if v is not None]


def _total(values: Iterable[Any]) -> float | None:
    found = _present(values)
    return sum(found) if found else None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    numerator, denominator = _num(numerator), _num(denominator)
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def tail(values: Iterable[Any], scale: float = 1.0) -> dict[str, Any]:
    """Count, mean, p50, p95, p99 and max, each value multiplied by `scale`."""
    found = [scale * v for v in _present(values)]
    return {
        "count": len(found),
        "mean": sum(found) / len(found) if found else None,
        "p50": percentile(found, 50),
        "p95": percentile(found, 95),
        "p99": percentile(found, 99),
        "max": max(found) if found else None,
    }


def _stage_order(stage: str) -> tuple[int, int, str]:
    base = base_stage(stage)
    rank = STAGES.index(base) if base in STAGES else len(STAGES)
    suffix = stage[len(base) + 1 :]
    return (int(suffix) if suffix.isdigit() else 0, rank, stage)


def build_report(run_dir: str | Path) -> dict[str, Any]:
    """Every metric of the standard set that this run's files can give."""
    run = RunDir(run_dir)
    steps_by_pass: dict[str, list[Step]] = defaultdict(list)
    for step in run.steps or ():
        steps_by_pass[step.pass_label].append(step)
    return {
        "run_id": run.run_id,
        "passes": {
            p.label: _pass_metrics(run, p, steps_by_pass[p.label]) for p in run.passes
        },
    }


def _pass_metrics(run: RunDir, p: Pass, steps: list[Step]) -> dict[str, Any]:
    record = p.record or {}
    programs = [run.programs[pid] for pid in p.program_ids if pid in run.programs]
    all_rows = [row for program in programs for row in program["stages"].values()]
    found = _present(prog.get("e2e_seconds") for prog in programs)
    makespan = _num(record.get("makespan_s"))
    output_tokens = (record.get("throughput") or {}).get("output_tokens")
    prompt = _total(r.get("prompt_tokens") for r in all_rows)
    cached = _total(r.get("cached_tokens") for r in all_rows)
    out: dict[str, Any] = {
        "programs": len(p.program_ids),
        "makespan_s": makespan,
        "programs_per_s": _ratio(len(p.program_ids), makespan),
        "output_tokens_per_s": _ratio(output_tokens, makespan),
        "program_e2e_s": {"mean": sum(found) / len(found) if found else None},
        "apc_hit_rate": _ratio(cached, prompt),
    }
    if run.trace_dirs:
        out["engines"] = _engines_metrics(run, p, makespan)
    out["speculation"] = _speculation(run, p)
    if run.trace_dirs:
        decoded = [
            entry
            for name, entry in out["engines"].items()
            if run.engine_roles.get(name) == ROLE_D
        ]
        if decoded and all(entry and entry.get("speculation") for entry in decoded):
            counts = Counter()
            for entry in decoded:
                counts.update(
                    {
                        key: entry["speculation"][key]
                        for key in (
                            "proposed_tokens",
                            "accepted_tokens",
                            "verify_steps",
                        )
                    }
                )
            out["speculation"] = {
                "scope": "decode-engine native request records only",
                "method": "suffix",
                **_spec_entry(counts),
            }
    out["occupancy"] = _occupancy(record)
    out["headline"] = _headline(run, p, programs, out, steps)
    out["results"] = _results(run, record, out)
    return out


def _headline(
    run: RunDir,
    p: Pass,
    programs: list[dict[str, Any]],
    out: Mapping[str, Any],
    steps: list[Step],
) -> dict[str, Any]:
    """The pass's handoffs, decode latency, KV use and matched and append lengths."""
    later = [
        (stage, row)
        for program in programs
        for stage, row in program["stages"].items()
        if base_stage(stage) != "generation"
    ]
    window = _num((run.config_value("pd_method") or {}).get("tpot_window_s"))
    engine_steps = _engine_step_lists(run, p)
    return {
        "makespan_s": out["makespan_s"],
        "handoff_s": stage_handoffs(programs),
        "tpot_window": None
        if engine_steps is None
        else windowed_tpot(engine_steps, window or TPOT_WINDOW_S),
        "engines": _kv_use(run, out, steps),
        "matched_append": {
            stage: {
                name: tail(r.get(name) for s, r in later if base_stage(s) == stage)
                for name in ("matched_tokens", "append_tokens")
            }
            for stage in sorted({base_stage(s) for s, _ in later}, key=_stage_order)
        },
    }


def cost(
    makespan_s: float | None,
    gpus: int,
    tokens: float | None,
    gpu_hour_usd: float | None = None,
) -> dict[str, Any]:
    """A pass's cost: G * T GPU-seconds, and $ per 1M tokens at p $/GPU-hour.

    `G p T / 3600 * 1e6 / D`, with T the makespan, G the GPUs the run allocates
    and D the input and output tokens of every call (prefix-cache hits as input,
    rejected drafts not at all).
    """
    gpu_seconds = None if makespan_s is None else gpus * makespan_s
    out: dict[str, Any] = {
        "gpus": gpus,
        "gpu_seconds": gpu_seconds,
        "logical_tokens": tokens,
    }
    if gpu_hour_usd is not None and gpu_seconds is not None and tokens:
        out["usd_per_1m_tokens"] = gpu_seconds * gpu_hour_usd / 3600 * 1e6 / tokens
    return out


def _results(
    run: RunDir, record: Mapping[str, Any], out: Mapping[str, Any]
) -> dict[str, Any]:
    """One pass's reported numbers, as `summary.json` carries them."""
    summary = run.summary or {}
    throughput = record.get("throughput") or {}
    tokens = _total((throughput.get("prompt_tokens"), throughput.get("output_tokens")))
    headline = out["headline"]
    decode = headline["tpot_window"] or {}
    speculation = out.get("speculation") or {}
    accepted = speculation.get("mean_accepted_per_verify_step")
    use = headline["engines"] or {}
    return {
        "makespan_s": out["makespan_s"],
        "cost": cost(
            out["makespan_s"],
            int(summary.get("gpus") or 1),
            tokens,
            summary.get("gpu_hour_usd"),
        ),
        "decode_latency_ms": decode.get("token_mean"),
        "program_e2e_mean_s": out["program_e2e_s"]["mean"],
        "programs_per_s": out["programs_per_s"],
        "output_tokens_per_s": out["output_tokens_per_s"],
        "total_tokens_per_s": throughput.get("total_tokens_per_s"),
        "prefix_hit_ratio": out["apc_hit_rate"],
        "matched_append": headline["matched_append"],
        "mean_accepted_length": None if accepted is None else 1 + accepted,
        "acceptance_rate": speculation.get("acceptance_rate"),
        "kv_usage": {
            name: {"peak": entry["kv_peak"], "mean": entry["kv_mean"]}
            for name, entry in use.items()
        },
        "handoff_s": headline["handoff_s"]["all"],
    }


def _engine_step_lists(run: RunDir, p: Pass) -> list[list[Step]] | None:
    """Each engine's steps of the pass, one list per clock; None if one is lost."""
    names = sorted(run.trace_dirs)
    found = [run.engine_steps(name) for name in names] if names else [run.steps]
    if any(steps is None for steps in found):
        return None
    return [[s for s in steps or () if s.pass_label == p.label] for steps in found]


def windowed_tpot(
    engine_steps: Iterable[Sequence[Step]], window_s: float
) -> dict[str, Any]:
    """Decode latency in ms: every decode gap over every token.

    On each engine's own clock, a decode row's gap is its step's end less the end
    of the step it last emitted in, the first token's included. Gaps are summed per
    request and `window_s` bin; `token_mean` is what a token-weighted TPOT budget
    bounds.
    """
    gaps = tokens = 0.0
    for steps in engine_steps:
        ordered = sorted(steps, key=lambda s: s.t_end)
        if not ordered:
            continue
        origin = ordered[0].t_start
        last: dict[str, float] = {}
        windows: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0.0, 0])
        for s in ordered:
            for r in s.reqs:
                if r.emitted <= 0:
                    continue
                before = last.get(r.request_id)
                last[r.request_id] = s.t_end
                if before is None or r.is_prefill:
                    continue
                entry = windows[(r.request_id, int((s.t_end - origin) // window_s))]
                entry[0] += s.t_end - before
                entry[1] += r.emitted
        for gap, emitted in windows.values():
            gaps += gap
            tokens += emitted
    return {"token_mean": 1e3 * gaps / tokens if tokens else None}


def stage_handoffs(programs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Each later stage's handoff: the stage before's last token to its first.

    On the driver's clock where the stage recorded it (`pd_method.handoff_s`),
    else on the engine clock when both stages ran on one engine, else skipped.
    """
    by_stage: dict[str, list[float]] = defaultdict(list)
    for program in programs:
        rows = sorted(program["stages"].items(), key=lambda item: _stage_order(item[0]))
        for (_, before), (stage, row) in zip(rows, rows[1:]):
            gap = _num((row.get("pd_method") or {}).get("handoff_s"))
            if gap is None:
                first = _num(row.get("first_token_ts"))
                last = _num(before.get("last_token_ts"))
                if (
                    first is None
                    or last is None
                    or before.get("engine", "") != row.get("engine", "")
                ):
                    continue
                gap = first - last
            by_stage[base_stage(stage)].append(gap)
    return {"all": tail(v for values in by_stage.values() for v in values)}


def _kv_use(
    run: RunDir, out: Mapping[str, Any], steps: list[Step]
) -> dict[str, Any] | None:
    """Per engine: KV peak and mean."""
    if run.trace_dirs:
        engines = out["engines"]
        if any(entry is None for entry in engines.values()):
            return None
        found = {name: entry["occupancy"] for name, entry in engines.items()}
    elif steps:
        found = {"single": out["occupancy"]}
    else:
        return None
    return {
        name: {
            "kv_peak": (occupancy or {}).get("kv_usage_peak"),
            "kv_mean": (occupancy or {}).get("kv_usage_mean_makespan"),
        }
        for name, occupancy in found.items()
    }


def _engines_metrics(run: RunDir, p: Pass, makespan: float | None) -> dict[str, Any]:
    """Each engine of a split run: its KV use and, on a D, its draft acceptance."""
    roles = run.engine_roles
    out: dict[str, Any] = {}
    for name in sorted(run.trace_dirs):
        if run.engine_steps(name) is None:
            out[name] = None
            continue
        totals = ((p.record or {}).get("engine") or {}).get(name) or {}
        entry: dict[str, Any] = {
            "occupancy": _occupancy(
                {
                    "engine": totals,
                    "occupancy": {
                        "mean_kv_cache_usage_makespan": _ratio(
                            totals.get("kv_usage_s"), makespan
                        ),
                    },
                }
            ),
        }
        if roles.get(name) == ROLE_D:
            spec = (run.engine_header(name) or {}).get("spec_method")
            if spec is not None:
                requests = run.engine_requests(name)
                entry["speculation"] = (
                    {"method": spec, **_request_speculation(run, p, requests)}
                    if requests is not None
                    else None
                )
        out[name] = entry
    return out


def _speculation(run: RunDir, p: Pass) -> dict[str, Any] | None:
    """Suffix speculation, from the engine's finished-request rows.

    `acceptance_rate` is committed over proposed drafts (scheduled for
    verification, invalid ones excluded); the mean is accepted tokens per
    verification step.
    """
    method = (run.steps_header or {}).get("spec_method")
    if method is None and run.steps_header is None:
        method = "suffix" if run.config_value("suffix_decoding") else None
    if method is None:
        return None
    if run.requests is None:
        return {"method": method}
    return {"method": method, **_request_speculation(run, p, run.requests)}


def _request_speculation(run: RunDir, p: Pass, requests: Mapping) -> dict[str, Any]:
    totals: Counter = Counter()
    for rid, row in requests.items():
        parts = split_request_id(rid)
        if parts is None or run.pass_of(parts[0]) != p.label:
            continue
        totals["proposed_tokens"] += int(row.get("proposed_tokens") or 0)
        totals["accepted_tokens"] += int(row.get("accepted_committed_tokens") or 0)
        totals["verify_steps"] += int(row.get("verification_steps") or 0)
    return _spec_entry(totals)


def _spec_entry(counts: Mapping[str, int]) -> dict[str, Any]:
    return {
        "proposed_tokens": counts.get("proposed_tokens", 0),
        "accepted_tokens": counts.get("accepted_tokens", 0),
        "verify_steps": counts.get("verify_steps", 0),
        "acceptance_rate": _ratio(
            counts.get("accepted_tokens", 0), counts.get("proposed_tokens", 0)
        ),
        "mean_accepted_per_verify_step": _ratio(
            counts.get("accepted_tokens", 0), counts.get("verify_steps", 0)
        ),
    }


def _occupancy(record: Mapping[str, Any]) -> dict | None:
    engine = record.get("engine") or {}
    if not engine.get("steps"):
        return None
    return {
        "kv_usage_peak": engine.get("peak_kv_cache_usage"),
        "kv_usage_mean_makespan": (record.get("occupancy") or {}).get(
            "mean_kv_cache_usage_makespan"
        ),
    }


def write_metrics(
    run_dir: str | Path, report: dict[str, Any] | None = None
) -> Path | None:
    """Put each pass's `results` in `summary.json` and clear a stale error file.

    Returns:
        The path written, or None for a run that wrote no summary.
    """
    run_dir = Path(run_dir)
    if report is None:
        report = build_report(run_dir)
    summary = read_json(run_dir / SUMMARY_FILE)
    if summary is None:
        return None
    summary["results"] = {label: p["results"] for label, p in report["passes"].items()}
    path = write_json(run_dir / SUMMARY_FILE, summary)
    (run_dir / METRICS_ERROR_FILE).unlink(missing_ok=True)
    return path


def write_metrics_safely(run_dir: str | Path) -> Path | None:
    """`write_metrics` that never raises; a failure goes to `metrics_error.txt`."""
    try:
        return write_metrics(run_dir)
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        with contextlib.suppress(Exception):
            (Path(run_dir) / METRICS_ERROR_FILE).write_text(traceback.format_exc())
        print(f"NOTE profiling: results were not written: {exc!r}")
        return None


def run_dirs(root: str | Path) -> list[Path]:
    """`root` if it is a run directory, else every run directory under it."""
    root = Path(root)
    if (root / "config.json").is_file() or (root / "summary.json").is_file():
        return [root]
    found = set()
    for name in ("config.json", "summary.json"):
        for path in root.rglob(name):
            payload = read_json(path)
            if isinstance(payload, dict) and (
                "run_id" in payload or "passes" in payload
            ):
                found.add(path.parent)
    return sorted(found)


TABLE_COLUMNS = (
    "run",
    "pass",
    "programs",
    "makespan_s",
    "gpu_seconds",
    "decode_latency_ms",
    "programs_per_s",
    "output_tok_s",
    "e2e_mean_s",
    "apc_hit",
    "spec_accept",
)


def table_rows(report: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    """One row per pass: the columns of `TABLE_COLUMNS`."""
    rows = []
    for label, p in report["passes"].items():
        rows.append(
            {
                "run": name,
                "pass": label,
                "programs": p["programs"],
                "makespan_s": p["makespan_s"],
                "gpu_seconds": p["results"]["cost"]["gpu_seconds"],
                "decode_latency_ms": p["results"]["decode_latency_ms"],
                "programs_per_s": p["programs_per_s"],
                "output_tok_s": p["output_tokens_per_s"],
                "e2e_mean_s": p["program_e2e_s"]["mean"],
                "apc_hit": p["apc_hit_rate"],
                "spec_accept": (p.get("speculation") or {}).get("acceptance_rate"),
            }
        )
    return rows


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.1f}"
        return f"{value:.2f}" if abs(value) >= 1 else f"{value:.3f}"
    return str(value)


def markdown(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_cell(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m turboreflect report")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    failed = 0
    rows: list[dict[str, Any]] = []
    for root in args.run_dirs:
        dirs = run_dirs(root)
        if not dirs:
            print(f"{root}: no run directory", file=sys.stderr)
            failed += 1
        for run_dir in dirs:
            try:
                report = build_report(run_dir)
                write_metrics(run_dir, report)
            except Exception as exc:  # noqa: BLE001 - reported per directory
                print(f"{run_dir}: {exc!r}", file=sys.stderr)
                failed += 1
                continue
            rows.extend(table_rows(report, run_dir.name))
    print(markdown(rows, TABLE_COLUMNS))
    return 1 if failed else 0

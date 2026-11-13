# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""Fit the P/D cost profile from one engine's measured run.

    python -m turboreflect.calibration.pd_profile --run-dir RUN --bringup NIXL.json \\
        --out-profile PROFILE.json

Deployment calibration for locality-aware P/D, which prices routes with cost
functions fitted here: from the engine step trace of a run whose engine is built
as every P and D engine is, and from the node's NIXL bring-up, which
`turboreflect.calibration.nixl_sweep` measures. No separate profiling run is
needed.

The step model (`policy.StepModel`): each measured step's time is fitted, by
least squares on relative error with coefficients held non-negative, on the 80%
of the steps `HOLDOUT_EVERY` keeps, to `a + b*L + c*L*(H + L/2) + e*L**2 + f*H`
summed over the step's prefill chunks (L new tokens attending to H earlier ones;
f*H is what every chunk repeats over the H cached tokens, such as MLA's
re-expansion of the cached latents through `kv_b_proj`), plus decode terms that
take out what the step's generating rows cost.

Transfer is the alpha-beta model, `alpha + tokens * kv_bytes_per_token /
bandwidth`, fitted to the bring-up's points, or its headline GB/s with alpha 0.
One engine configuration serves every worker, so the profile states it once, for
`ANY_WORKER`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from turboreflect.locality_pd.config import source_identity
from turboreflect.locality_pd.policy import ANY_WORKER, step_features
from turboreflect.metrics.report import COST_TERMS
from turboreflect.metrics.rundir import (
    REQUESTS_FILE,
    STEPS_FILE,
    RunDir,
    Step,
    read_json,
)

# The fit leaves out every step whose number is a multiple of this: 20% of them.
HOLDOUT_EVERY = 5
# The step model: `report.COST_TERMS`, with the square of each prefill chunk.
TERMS = (*COST_TERMS, "prefill_square")


class FitError(Exception):
    """The run or the bring-up cannot give this profile."""


def features(step: Step) -> list[float]:
    x = step_features((r.scheduled, r.context, r.is_prefill) for r in step.reqs)
    return [x[term] for term in TERMS]


def nonnegative_fit(rows: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Coefficients minimising relative squared error, none of them negative.

    The most negative term is dropped and the rest refitted until none is.
    """
    weighted = rows / times[:, None]
    scale = np.abs(weighted).max(axis=0)
    scale[scale == 0] = 1.0
    keep = [i for i in range(rows.shape[1]) if rows[:, i].any()]
    while keep:
        solved, *_ = np.linalg.lstsq(
            weighted[:, keep] / scale[keep], np.ones(len(times)), rcond=None
        )
        solved = solved / scale[keep]
        if (solved >= 0).all():
            break
        keep.pop(int(np.argmin(solved)))
    coef = np.zeros(rows.shape[1])
    coef[keep] = solved
    return coef


def affine(feature: str, bounds: tuple[int, int], intercept: float, slope: float):
    return {
        "bounds": {feature: [int(bounds[0]), int(bounds[1])]},
        "intercept_s": float(intercept),
        "coefficients_s": {feature: float(slope)},
    }


def fit_steps(steps: Sequence[Step]) -> tuple[dict[str, float], dict[str, Any]]:
    """The step model, fitted on the steps `HOLDOUT_EVERY` keeps."""
    usable = [s for s in steps if s.seconds > 0]
    train = [s for s in usable if s.step % HOLDOUT_EVERY]
    rows = np.array([features(s) for s in train])
    times = np.array([s.seconds for s in train])
    coef = nonnegative_fit(rows, times)
    model = dict(zip(TERMS, map(float, coef)))
    report = {
        "terms": model,
        "form": "step seconds = sum over terms; prefill_pairs = chunk * "
        "(context + chunk / 2), prefill_context_tokens = context, "
        "prefill_square = chunk ** 2",
        "weighting": "relative error, coefficients >= 0",
        "train_steps": len(train),
    }
    return model, report


def fit_transfer(
    bringup: Mapping[str, Any], bytes_per_token: int, top: int
) -> tuple[dict, dict[str, Any]]:
    """The alpha-beta transfer model from the bring-up's points, per KV token."""
    if not (bringup.get("gb_per_s") or 0) > 0:
        raise FitError(f"the bring-up measured no transfer rate: {dict(bringup)}")
    points = [
        (float(p["bytes"]), float(p["seconds"]))
        for p in bringup.get("points") or []
        if (p.get("bytes") or 0) > 0 and (p.get("seconds") or 0) > 0
    ]
    if len({b for b, _ in points}) >= 2:
        rows = np.array([[1.0, b] for b, _ in points])
        seconds = np.array([s for _, s in points])
        alpha, per_byte = nonnegative_fit(rows, seconds)
        source = f"alpha-beta fit to {len(points)} bring-up points"
    else:
        alpha, per_byte = 0.0, 1 / (float(bringup["gb_per_s"]) * 1e9)
        source = "headline gb_per_s, alpha 0"
    segment = affine("tokens", (0, top), alpha, per_byte * bytes_per_token)
    report = {
        "alpha_s": float(alpha),
        "gb_per_s": float(1 / per_byte / 1e9),
        "bring_up_gb_per_s": bringup["gb_per_s"],
        "kv_bytes_per_token": bytes_per_token,
        "source": source,
        "bring_up_source": bringup.get("source"),
        "applies_to": "every P->D and D->P pair (one pair measured)",
    }
    return segment, report


def fit(run_dir: Path, bringup_path: Path) -> tuple[dict, dict]:
    """The profile and the fit report."""
    run = RunDir(run_dir)
    record = read_json(run.path / "provenance.json") or {}
    engine = record.get("engine") or {}
    identity = source_identity(record)
    if not identity["model"] or not identity["gpu"]:
        raise FitError(f"{run.path}/provenance.json names no engine or GPU")
    chunk = engine.get("max_num_batched_tokens")
    length = engine.get("max_model_len")
    if not all(isinstance(v, int) and v > 0 for v in (chunk, length)):
        raise FitError("provenance.json lacks the engine's token budget or length")
    steps = run.steps
    if not steps:
        raise FitError(f"no measured steps in {run.path / STEPS_FILE}")
    bytes_per_token = run.kv_pool().get("kv_bytes_per_token")
    bringup = read_json(bringup_path)
    if not isinstance(bringup, dict):
        raise FitError(f"{bringup_path} is not a JSON object")
    if not isinstance(bytes_per_token, int) or bytes_per_token <= 0:
        raise FitError("the run records no kv_bytes_per_token")

    model, step_report = fit_steps(steps)
    transfer, transfer_report = fit_transfer(bringup, bytes_per_token, length + 1)

    evidence = [
        str(path.resolve())
        for path in (
            run.path / STEPS_FILE,
            run.path / REQUESTS_FILE,
            run.path / "config.json",
            run.path / "summary.json",
            run.path / "provenance.json",
            bringup_path,
        )
        if path.is_file()
    ]
    code = record.get("code") or {}
    profile = {
        "schema_version": 1,
        "provenance": {
            "kind": "measured",
            "profile_id": f"pd-profile-{run.run_id}",
            "model_revision": str(
                engine.get("hf_commit")
                or engine.get("model_revision")
                or identity["model"]
            ),
            "vllm_revision": str(code.get("commit"))
            + ("+dirty" if code.get("dirty") else ""),
            "gpu": identity["gpu"],
            "kv_dtype": str(identity["kv_cache_dtype"]),
            "kernel_fingerprint": "/".join(
                str(identity[k])
                for k in ("attention_backend", "mla_prefill_backend", "moe_backend")
            ),
            "evidence_files": evidence,
            "method": "fitted from one engine's measured step trace and the NIXL "
            "bring-up (pd_profile); no profiling run",
            "source_identity": identity,
            "chunk_tokens": chunk,
            "step_terms": model,
        },
        "transfer": {ANY_WORKER: {ANY_WORKER: [transfer]}},
    }
    report = {
        "run_dir": str(run.path),
        "bringup": str(bringup_path),
        "step_model": step_report,
        "transfer": transfer_report,
    }
    return profile, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bringup", type=Path, required=True)
    parser.add_argument("--out-profile", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        profile, report = fit(args.run_dir, args.bringup)
    except FitError as exc:
        print(f"P/D fit failed: {exc}", file=sys.stderr)
        return 1
    args.out_profile.write_text(json.dumps(profile, indent=2) + "\n")
    report_path = args.out_profile.with_suffix(".fit.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    steps = report["step_model"]
    print(
        f"step model: {steps['train_steps']} steps, context re-expansion "
        f"{steps['terms']['prefill_context_tokens']:.3g} s/token"
    )
    print(f"transfer: {report['transfer']}")
    print(f"wrote {args.out_profile}, {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

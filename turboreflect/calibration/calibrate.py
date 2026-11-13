# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""Deployment calibration: measure what a run's decisions read.

    python -m turboreflect calibrate --config turboreflect/configs/qwen3-14b.yaml

Writes into `config.calibration_dir`, where `run` finds the files on its own:

- `step_model.json`, the step times adaptive draft length prices drafts with
  (`calibration.step_time`).
- `pd_profile.json`, with a `pd` block: the costs locality-aware P/D places stages with
  (`calibration.pd_profile`), fitted from one pass over the tasks on an engine built as
  every P and D engine is, and a NIXL read sweep (`calibration.nixl_sweep`).

Each step runs in its own process, so one engine's GPU memory is freed before the
next; a step whose result exists is skipped unless `--force` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from turboreflect.config import (
    PD_PROFILE_FILE,
    STEP_MODEL_FILE,
    ConfigError,
    RunConfig,
    calibration_dir,
    config_document,
    expand_switches,
    hf_snapshot_dir,
    step_model_block,
)

REFERENCE_RUN = "reference"


class StepFailed(Exception):
    """One calibration step exited with an error."""

    def __init__(self, code: int) -> None:
        super().__init__(f"exit {code}")
        self.code = code


def run_step(*command: str, env: Mapping[str, str] | None = None) -> None:
    """Run `python -m <command>` to its end; `StepFailed` on an error exit."""
    print(f"calibrate: python -m {' '.join(command)}", flush=True)
    code = subprocess.run(
        [sys.executable, "-m", *command], env={**os.environ, **(env or {})}
    ).returncode
    if code:
        raise StepFailed(code)


def attention_layers(model: str) -> int:
    """The layers that keep KV per token, from the checkpoint's `config.json`.

    Raises:
        ConfigError: If the checkpoint is not on disk.
    """
    root = Path(model) if Path(model).is_dir() else hf_snapshot_dir(model)
    if root is None or not (root / "config.json").is_file():
        raise ConfigError(f"no config.json on disk for {model}")
    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config") or config
    kinds = text.get("layer_types")
    if kinds:
        return sum(kind == "full_attention" for kind in kinds)
    return text["num_hidden_layers"]


def sweep_arguments(run_dir: Path, model: str) -> list[str]:
    """`nixl_sweep`'s KV shape, as the reference run's pool recorded it."""
    pool = json.loads((run_dir / "config.json").read_text())["kv_pool"]
    layers = attention_layers(model)
    return [
        "--layers",
        str(layers),
        "--head-bytes",
        str(pool["kv_bytes_per_token"] // layers),
        "--block-tokens",
        str(pool["block_size"]),
    ]


def reference_run(
    fields: Mapping[str, Any], doc: Mapping[str, Any], directory: Path
) -> dict[str, Any]:
    """The configuration the P/D fit reads: every program at once on one engine."""
    config = {
        **fields,
        "concurrency": fields["program_total"],
        "out_dir": str(directory / REFERENCE_RUN),
    }
    if doc.get("adaptive_draft_length", True):
        config["adaptive_draft"] = step_model_block(
            directory, doc.get("adaptive_draft") or {}
        )
    return config


def write_config(path: Path, fields: Mapping[str, Any]) -> str:
    path.write_text(json.dumps(fields, indent=1))
    return str(path)


def fit_pd_profile(
    doc: Mapping[str, Any], fields: Mapping[str, Any], directory: Path, force: bool
) -> None:
    """Measure and fit `pd_profile.json`."""
    run_dir = directory / REFERENCE_RUN / REFERENCE_RUN
    if force or not (run_dir / "summary.json").is_file():
        config = reference_run(fields, doc, directory)
        run_step(
            "turboreflect",
            "run",
            "--config",
            write_config(directory / "reference_config.json", config),
            env={"TURBOREFLECT_RUN_ID": REFERENCE_RUN},
        )
    sweep = directory / "nixl_sweep.json"
    if force or not sweep.is_file():
        run_step(
            "turboreflect.calibration.nixl_sweep",
            "--src",
            "0",
            "--dst",
            str(doc["pd"]["prefill"] * fields["tensor_parallel_size"]),
            "--out",
            str(sweep),
            *sweep_arguments(run_dir, fields["model"]),
        )
    run_step(
        "turboreflect.calibration.pd_profile",
        "--run-dir",
        str(run_dir),
        "--bringup",
        str(sweep),
        "--out-profile",
        str(directory / PD_PROFILE_FILE),
    )


def calibrate(doc: Mapping[str, Any], force: bool = False) -> None:
    """Write what `doc`'s run reads into its calibration directory.

    Args:
        doc: A config document of component switches (`config_document`).

    Raises:
        ConfigError: If the document does not resolve.
        StepFailed: If a step fails.
    """
    drafts = doc.get("adaptive_draft_length", True)
    if not drafts and not doc.get("pd"):
        print("calibrate: this configuration reads no calibration")
        return
    fields = expand_switches(doc, calibrated=False)
    directory = calibration_dir(doc)
    directory.mkdir(parents=True, exist_ok=True)
    engine_config = write_config(directory / "engine_config.json", fields)
    RunConfig.from_namespace(argparse.Namespace(config=engine_config))
    if drafts and (force or not (directory / STEP_MODEL_FILE).is_file()):
        run_step(
            "turboreflect.calibration.step_time",
            "--config",
            engine_config,
            "--out",
            str(directory),
        )
    if doc.get("pd") and (force or not (directory / PD_PROFILE_FILE).is_file()):
        fit_pd_profile(doc, fields, directory, force)
    print(f"calibrate: {directory}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m turboreflect calibrate")
    RunConfig.add_arguments(parser)
    parser.add_argument(
        "--force", action="store_true", help="repeat steps that have already run"
    )
    args = parser.parse_args(argv)
    try:
        doc = config_document(args)
        if "samples" in doc:
            raise ConfigError(
                "calibrate reads a configuration of component switches, not `samples`"
            )
        calibrate(doc, args.force)
    except ConfigError as exc:
        parser.error(str(exc))
    except StepFailed as exc:
        return exc.code
    return 0

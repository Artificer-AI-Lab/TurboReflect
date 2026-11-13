# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""What a run says of the engine it built, for deployment calibration.

`provenance.json` holds the commit, the GPU and the engine facts a calibration fit
reads back (`calibration.pd_profile`). A failed lookup leaves its fact empty.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def code_version() -> dict[str, Any]:
    """The commit this tree was built from, and whether it had edits."""
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


def local_devices() -> list[dict[str, Any]]:
    """The devices this process can see."""
    try:
        import torch

        count = torch.accelerator.device_count()
        return [{"name": torch.cuda.get_device_name(i)} for i in range(count)]
    except Exception:  # noqa: BLE001 - left empty, not raised
        return []


def mla_prefill_backend(vllm_config: Any) -> str | None:
    """The MLA prefill backend the workers select, by vLLM's own selector."""
    try:
        from vllm.config import set_current_vllm_config
        from vllm.v1.attention.backends.mla.prefill.selector import (
            get_mla_prefill_backend,
        )

        if not getattr(vllm_config.model_config, "use_mla", False):
            return None
        with set_current_vllm_config(vllm_config):
            return get_mla_prefill_backend(vllm_config).get_name()
    except Exception:  # noqa: BLE001 - left empty, not raised
        return None


def engine_facts(vllm_config: Any) -> dict[str, Any]:
    """What the engine was built as; `attention_backend` is None unless forced."""

    def get(obj: Any, *names: str) -> Any:
        for name in names:
            obj = getattr(obj, name, None)
            if obj is None:
                return None
        return obj if isinstance(obj, bool | int | float | str) else str(obj)

    cfg = vllm_config
    return {
        "model": get(cfg, "model_config", "model"),
        "model_revision": get(cfg, "model_config", "revision"),
        "hf_commit": get(cfg, "model_config", "hf_config", "_commit_hash"),
        "model_dtype": get(cfg, "model_config", "dtype"),
        "max_model_len": get(cfg, "model_config", "max_model_len"),
        "enforce_eager": get(cfg, "model_config", "enforce_eager"),
        "attention_backend": get(cfg, "attention_config", "backend"),
        "mla_prefill_backend": mla_prefill_backend(cfg),
        "kv_cache_dtype": get(cfg, "cache_config", "cache_dtype"),
        "tensor_parallel_size": get(cfg, "parallel_config", "tensor_parallel_size"),
        "max_num_seqs": get(cfg, "scheduler_config", "max_num_seqs"),
        "max_num_batched_tokens": get(
            cfg, "scheduler_config", "max_num_batched_tokens"
        ),
        "cudagraph_mode": get(cfg, "compilation_config", "cudagraph_mode"),
        "moe_backend": get(cfg, "kernel_config", "moe_backend"),
    }


def build(vllm_config: Any) -> dict[str, Any]:
    """The record `ResultWriter.write_provenance` writes."""
    return {
        "code": code_version(),
        "engine": engine_facts(vllm_config),
        "hardware": {"local_devices": local_devices()},
    }

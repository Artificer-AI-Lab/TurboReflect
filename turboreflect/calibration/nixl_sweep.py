# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""A raw NIXL READ sweep between two GPUs, for `pd_profile --bringup`.

    python -m turboreflect.calibration.nixl_sweep --src 0 --dst 1 --out nixl.json \\
        --layers LAYERS --head-bytes BYTES --block-tokens TOKENS

One process registers a buffer shaped as the model's KV; the other READs growing
prefixes of it with one descriptor per layer and block, as vLLM's MLA connector
registers them, timed after one warm-up read.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

MODULE = "turboreflect.calibration.nixl_sweep"
SIZES = (1024, 8192, 32768, 131072)
TIMED_ROUNDS = 3


def _nixl_agent(name: str) -> Any:
    from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config

    if NixlWrapper is None:
        raise RuntimeError("NIXL did not load")
    return NixlWrapper(
        name, nixl_agent_config(num_threads=4) if nixl_agent_config else None
    )


def _descriptors(base: int, entries: int, block_len: int, device: int) -> Any:
    import numpy as np

    out = np.empty((entries, 3), dtype=np.int64)
    out[:, 0] = base + np.arange(entries, dtype=np.int64) * block_len
    out[:, 1] = block_len
    out[:, 2] = device
    return out


def _wait_for(path: Path, timeout: float) -> None:
    end = time.time() + timeout
    while not path.exists():
        if time.time() > end:
            raise TimeoutError(f"{path} did not appear in {timeout:.0f} s")
        time.sleep(0.2)


def _register(agent: Any, buf: Any, device: int) -> None:
    reg = agent.get_reg_descs([(buf.data_ptr(), buf.numel(), device, "")], "VRAM")
    agent.register_memory(reg, backends=["UCX"])


def _target(args: argparse.Namespace, shared: Path) -> None:
    import torch

    blocks = max(SIZES) // args.block_tokens
    block_len = args.block_tokens * args.head_bytes
    buf = torch.empty(
        args.layers * blocks * block_len, dtype=torch.uint8, device="cuda"
    )
    torch.cuda.synchronize()
    agent = _nixl_agent("sweep-target")
    device = torch.cuda.current_device()
    _register(agent, buf, device)
    (shared / "target.meta").write_bytes(agent.get_agent_metadata())
    (shared / "target.json").write_text(
        json.dumps({"base": buf.data_ptr(), "device": device})
    )
    _wait_for(shared / "reader.done", args.timeout)


def _timed_read(
    agent: Any, local: Any, remote: Any, ids: list[int], timeout: float
) -> float:
    handle = agent.make_prepped_xfer(
        "READ", local, ids, remote, ids, notif_msg=b"sweep"
    )
    started = time.perf_counter()
    agent.transfer(handle)
    while (state := agent.check_xfer_state(handle)) != "DONE":
        if state != "PROC":
            raise RuntimeError(f"NIXL transfer state {state}")
        if time.perf_counter() - started > timeout:
            raise TimeoutError(f"NIXL transfer of {len(ids)} blocks timed out")
    seconds = time.perf_counter() - started
    agent.release_xfer_handle(handle)
    return seconds


def _reader(args: argparse.Namespace, shared: Path) -> None:
    import torch

    _wait_for(shared / "target.json", args.timeout)
    target = json.loads((shared / "target.json").read_text())
    blocks = max(SIZES) // args.block_tokens
    block_len = args.block_tokens * args.head_bytes
    entries = args.layers * blocks
    buf = torch.zeros(entries * block_len, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    agent = _nixl_agent("sweep-reader")
    device = torch.cuda.current_device()
    _register(agent, buf, device)
    peer = agent.add_remote_agent((shared / "target.meta").read_bytes())
    local = agent.prep_xfer_dlist(
        "NIXL_INIT_AGENT",
        agent.get_xfer_descs(
            _descriptors(buf.data_ptr(), entries, block_len, device), "VRAM"
        ),
    )
    remote = agent.prep_xfer_dlist(
        peer,
        agent.get_xfer_descs(
            _descriptors(target["base"], entries, block_len, target["device"]), "VRAM"
        ),
    )
    points = []
    for tokens in (size for size in SIZES if size >= args.block_tokens):
        n = tokens // args.block_tokens
        ids = [layer * blocks + b for layer in range(args.layers) for b in range(n)]
        times = [
            _timed_read(agent, local, remote, ids, args.timeout)
            for _ in range(TIMED_ROUNDS + 1)
        ][1:]
        moved = args.layers * n * block_len
        mean = sum(times) / len(times)
        points.append(
            {
                "tokens": tokens,
                "bytes": moved,
                "seconds": mean,
                "gb_per_s": moved / mean / 1e9,
            }
        )
    record = sweep_record(points)
    (shared / "reader.json").write_text(json.dumps(record))
    (shared / "reader.done").touch()


def sweep_record(points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """What `pd_profile --bringup` reads: the largest read's rate and every point."""
    return {
        "kind": "nixl_bringup",
        "source": "nixl_sweep",
        "gb_per_s": points[-1]["gb_per_s"],
        "points": list(points),
    }


def _spawn(role: str, device: str, argv: Sequence[str]) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", MODULE, *argv, "--role", role],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": device},
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m {MODULE}")
    parser.add_argument("--src", required=True, help="the GPU the KV is read from")
    parser.add_argument("--dst", required=True, help="the GPU that reads it")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--head-bytes", type=int, required=True)
    parser.add_argument("--block-tokens", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--role", choices=("target", "reader"), help=argparse.SUPPRESS)
    parser.add_argument("--shared", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.role is not None:
        (_target if args.role == "target" else _reader)(args, args.shared)
        return 0
    with tempfile.TemporaryDirectory() as shared:
        passed = [
            *(sys.argv[1:] if argv is None else argv),
            "--shared",
            shared,
        ]
        target = _spawn("target", args.src, passed)
        reader = _spawn("reader", args.dst, passed)
        code = reader.wait()
        result = Path(shared) / "reader.json"
        # NIXL's teardown can crash after the result is written; trust the file.
        if not result.is_file():
            target.kill()
            target.wait()
            return code or 1
        target.wait()
        record = json.loads(result.read_text())
    record.update(src=args.src, dst=args.dst)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"NIXL READ {args.src}->{args.dst}: {record['gb_per_s']} GB/s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

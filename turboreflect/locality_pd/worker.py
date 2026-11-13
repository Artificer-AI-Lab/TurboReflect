# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""One engine, in a process that pins its device before it imports vLLM.

Spawn imports this module before calling the target, so a module-level vLLM import
would leave the engine on the parent's devices and its NIXL side channel on port
5600; every vLLM or torch import sits inside a function. The worker acknowledges
admission in a message of its own: from outside, an unadmitted request and one
waiting for KV look the same but need opposite cleanups.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Coroutine, Mapping
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from turboreflect.locality_pd import message

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from turboreflect.config import RunConfig
    from turboreflect.locality_pd.config import PDSettings

# Package roots whose first import decides which device this process lives on.
AFTER_PINNING = ("vllm", "torch")


class WorkerError(Exception):
    """This process cannot serve the side of the run it was handed."""


def worker_main(
    name: str,
    cfg: RunConfig,
    settings: PDSettings,
    conn: Connection,
    steps_path: str | None = None,
) -> None:
    """Serve one engine of a P/D run until the driver says stop.

    Args:
        name: The engine's name in `settings`, which gives its devices and port.
        steps_path: A run-directory path; the engine's trace and console log go
            beside it, so None refuses to build an engine.
    """
    try:
        _refuse_a_shared_interpreter()
        _pin_device(settings.devices(name), settings.port(name))
        _serve(name, cfg, settings, conn, steps_path)
    except Exception as exc:
        # The driver is waiting on a `ready` that is not coming: say why first.
        _report_failure(conn, exc)
        raise


def _refuse_a_shared_interpreter() -> None:
    """Raise unless this interpreter is still free to choose its device."""
    already = [name for name in AFTER_PINNING if name in sys.modules]
    if not already:
        return
    raise WorkerError(
        f"{' and '.join(already)} was imported before this worker pinned its "
        "card. CUDA_VISIBLE_DEVICES and VLLM_NIXL_SIDE_CHANNEL_PORT are read "
        "once, so setting them now leaves the engine on whichever devices the "
        "parent could see and both sides on the default side-channel port 5600 "
        "-- two engines start, the handshake finds the wrong peer, and nothing "
        "downstream calls that an error"
    )


def _pin_device(devices: tuple[int | str, ...], port: int) -> None:
    """Fix this process's devices and side channel, before anything reads them."""
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    allocated = _device_indices(inherited)
    devices = tuple(
        int(d) if str(d).isascii() and str(d).isdigit() else d for d in devices
    )
    missing = [d for d in devices if allocated is not None and d not in allocated]
    if missing:
        raise WorkerError(
            f"CUDA_VISIBLE_DEVICES is already {inherited!r}, which does not "
            f"include device(s) {missing} -- overwriting it would pin this "
            "engine outside the visible devices, and that surfaces as a memory "
            "or permission error from inside the engine core rather than as the "
            "configuration mistake it is"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(port)


def _device_indices(value: str | None) -> list[int | str] | None:
    """Allocated physical identifiers; numeric and UUID namespaces stay distinct."""
    if value is None:
        return None
    if not value.strip() or value.strip() == "-1":
        return []
    entries = [entry.strip() for entry in value.split(",")]
    if any(not entry or entry.startswith("-") for entry in entries):
        raise WorkerError(f"invalid CUDA_VISIBLE_DEVICES allocation: {value!r}")
    return [
        int(entry) if entry.isascii() and entry.isdigit() else entry
        for entry in entries
    ]


def _report_failure(conn: Any, exc: BaseException) -> None:
    """Tell the driver what went wrong, without replacing it if that fails."""
    with suppress(Exception):
        conn.send(
            {
                "kind": message.FAILED,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _serve(
    name: str,
    cfg: RunConfig,
    settings: PDSettings,
    conn: Connection,
    steps_path: str | None,
) -> None:
    asyncio.run(_run(name, cfg, settings, conn, steps_path))


def trace_dir(steps_path: str, name: str) -> Path:
    """One engine's own trace directory; the trace's file names are fixed
    (`turboreflect.metrics.trace`), so engines sharing one would interleave rows.
    """
    return Path(steps_path).parent / f"engine_trace_{name}"


def log_path(steps_path: str, name: str) -> Path:
    """Where one engine's console output is kept, in the run directory."""
    return Path(steps_path).parent / f"engine_log_{name}.txt"


def _capture_output(path: Path) -> None:
    """Also copy what this process and its engine-core children print into `path`,
    via a `tee` per descriptor in its own session, so an interrupt that stops the
    engines does not stop their logs first.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for stream in (sys.stdout, sys.stderr):
        stream.flush()
        fd = stream.fileno()
        read_end, write_end = os.pipe()
        subprocess.Popen(
            ["tee", "-a", str(path)],
            stdin=read_end,
            stdout=fd,
            start_new_session=True,
        )
        os.close(read_end)
        os.dup2(write_end, fd)
        os.close(write_end)


async def _run(
    name: str,
    cfg: RunConfig,
    settings: PDSettings,
    conn: Connection,
    steps_path: str | None,
) -> None:
    """Capture output before vLLM starts children, build this engine, then serve."""
    if steps_path is None:
        raise WorkerError(
            f"the {name} worker was given no path, so its engine trace has "
            "no run directory to go in"
        )
    _capture_output(log_path(steps_path, name))
    from turboreflect.engine.engine import (
        build_pd_engine,
        collect_engine_facts,
        kv_pool_report,
    )
    from turboreflect.locality_pd.runtime import DecodeWindowStats
    from turboreflect.metrics.observe import StepLogger

    logger = StepLogger(role=settings.role(name))
    window_stats = DecodeWindowStats(cfg.pd_method.tpot_window_s)
    llm = build_pd_engine(
        cfg,
        settings,
        name,
        trace_dir(steps_path, name),
        stat_loggers=[logger.as_factory(), window_stats.as_factory()],
    )
    try:
        facts = collect_engine_facts(llm)
        # Before `ready`, so an undecodable clip fails the bring-up, not a program.
        clips, lane, layouts = _load_clips(llm, cfg, settings, name)
        conn.send(
            {
                "kind": message.READY,
                "name": name,
                "role": settings.role(name),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "kv_pool": (await kv_pool_report(llm)).as_dict(),
                "facts": dataclasses.asdict(facts),
                **({message.CLIP_LAYOUTS: layouts} if layouts else {}),
            }
        )
        await _dispatch(
            llm,
            conn,
            logger,
            window_stats=window_stats,
            clips=clips,
            lane=lane,
        )
    finally:
        llm.shutdown()


def _load_clips(
    llm: Any, cfg: RunConfig, settings: PDSettings, name: str
) -> tuple[Any, Any, dict[Any, Any]]:
    """Decode the samples' clips; the first engine also expands their lines once.

    Returns the clips, their lane (None without clips), and expansions by clip paths.
    """
    from turboreflect.program.media import (
        FRAME_STORE_ENV,
        clip_layouts,
        clip_table,
        input_lane,
        layouts_needed,
    )

    clips = clip_table(cfg, lambda: llm.vllm_config.model_config)
    lane = input_lane(llm) if clips else None
    if clips:
        print(f"{name}: decoded the clips of {len(clips)} sample(s)", flush=True)
        print(
            f"{name}: pid {os.getpid()}, frame store "
            f"{os.environ.get(FRAME_STORE_ENV) or 'off'}",
            flush=True,
        )
    layouts = {}
    if clips and layouts_needed(cfg) and name == settings.names()[0]:
        started = time.perf_counter()
        layouts = clip_layouts(
            cfg, clips, llm.vllm_config.model_config, llm.get_tokenizer()
        )
        print(
            f"{name}: expanded the clip lines of {len(layouts)} sample(s) in "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return clips, lane, layouts


@dataclasses.dataclass
class _GenerationState:
    admission_done: bool = False
    admitted: bool = False
    finished: bool = False
    cancel_requested: bool = False
    reject_params: Any = None
    cleanup_done: bool = False


async def _cancel_generation(llm: Any, request_id: str, state: _GenerationState):
    """Choose native cleanup only after the local admission call settles."""
    if not state.cancel_requested or not state.admission_done or state.cleanup_done:
        return
    state.cleanup_done = True
    if state.finished:
        return
    if state.admitted:
        await llm.abort(request_id)
    elif state.reject_params is not None:
        await llm.notify_kv_transfer_request_rejected(request_id, state.reject_params)


async def _cancel_request(
    llm: Any, requests: dict[str, _GenerationState], inbound: Mapping[str, Any]
) -> None:
    """Take the driver's abort or rejection of one request."""
    request_id = inbound["request_id"]
    state = requests.get(request_id)
    if state is None:
        # Untracked here (never sent, or pruned): reject natively.
        state = _GenerationState(admission_done=True)
        requests[request_id] = state
    state.cancel_requested = True
    if inbound.get("kind") == message.REJECT:
        state.reject_params = inbound["kv_transfer_params"]
    await _cancel_generation(llm, request_id, state)


async def _dispatch(
    llm: Any,
    conn: Connection,
    logger: Any,
    *,
    window_stats: Any,
    clips: Mapping[tuple[str, ...], Any] | None = None,
    lane: Any = None,
) -> None:
    """Read the pipe until the driver stops talking, running what it asks.

    Requests run as tasks, so an abort for one is read while others run. `clips`
    (`program.media.clip_table`) are processed off the event loop on `lane`.
    """
    reader = message.PipeReader(conn)
    running: set[asyncio.Task] = set()
    requests: dict[str, _GenerationState] = {}

    def completed(task: asyncio.Task) -> None:
        running.discard(task)
        if not task.cancelled():
            try:
                task.result()
            except Exception as exc:
                _report_failure(conn, exc)

    def start(work: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(work)
        running.add(task)
        task.add_done_callback(completed)

    try:
        start(_push_headroom(llm, conn, window_stats))
        while True:
            try:
                inbound = await reader.recv()
            except EOFError:
                return
            kind = inbound.get("kind")
            if kind == message.SHUTDOWN:
                return
            if kind == message.GENERATE:
                request_id = inbound["request_id"]
                state = requests[request_id] = _GenerationState()
                start(_generate(llm, conn, inbound, logger, state, clips, lane))
            elif kind in (message.ABORT, message.REJECT):
                await _cancel_request(llm, requests, inbound)
            elif kind == message.RESET_PREFIX_CACHE:
                # Refused while blocks are pinned; the driver retries (`drain`).
                ok = await llm.reset_prefix_cache()
                if ok:
                    requests = {
                        rid: state
                        for rid, state in requests.items()
                        if not (state.finished or state.cleanup_done)
                    }
                conn.send({"kind": message.RESET, "ok": bool(ok)})
            elif kind in (message.OPEN_WINDOW, message.RESET_RUN, message.STEPS):
                if kind != message.STEPS:
                    window_stats.clear()
                _steps(conn, logger, kind, inbound)
            elif kind == message.RELEASE_HOLD:
                start(_release(llm, inbound["request_id"]))
            else:
                raise WorkerError(
                    f"the driver sent {kind!r}, which this worker does not "
                    f"handle; the protocol is {list(message.TO_WORKER)}"
                )
    finally:
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)


def _steps(conn: Any, logger: Any, kind: str, inbound: Mapping[str, Any]) -> None:
    """Work this engine's step logger, which only this process can reach."""
    if kind == message.STEPS:
        conn.send(
            {
                "kind": message.STEP_TOTALS,
                "run": logger.run.as_dict(),
                "window": logger.window.as_dict(),
            }
        )
    elif kind == message.OPEN_WINDOW:
        logger.open_window(inbound["label"])
    else:
        logger.reset_run()


async def _release(llm: Any, request_id: str) -> None:
    """Free a hold no pull will read; a failure leaves it to its TTL."""
    try:
        await llm.engine_core.call_utility_async(
            message.RELEASE_HOLD_UTILITY, request_id
        )
    except Exception as exc:  # noqa: BLE001 - the hold then ends at its TTL
        print(f"release of {request_id}'s KV hold failed: {exc!r}", flush=True)


async def _scheduler_state(llm: Any, window_stats: Any) -> dict[str, Any]:
    """This engine's scheduler state and latency windows, or why they were not read."""
    try:
        state = await llm.engine_core.call_utility_async(
            message.SCHEDULER_STATE_UTILITY
        )
        answer = {"locality_pd": {**state, **window_stats.snapshot()}}
    except Exception as exc:
        answer = {"error": f"{type(exc).__name__}: {exc}"}
    return answer


async def _push_headroom(
    llm: Any,
    conn: Any,
    window_stats: Any,
    poll_s: float = message.HEADROOM_POLL_S,
    refresh_s: float = message.HEADROOM_REFRESH_S,
) -> None:
    """Push the scheduler state, read every `poll_s`, when it changed or `refresh_s`
    after the last push, so an idle engine still reads as alive.
    """
    last: dict[str, Any] | None = None
    pushed_at = -math.inf
    while True:
        answer = await _scheduler_state(llm, window_stats)
        now = time.monotonic()
        if answer != last or now - pushed_at >= refresh_s:
            conn.send({"kind": message.HEADROOM, **answer})
            last, pushed_at = answer, now
        await asyncio.sleep(poll_s)


async def _generate(
    llm: Any,
    conn: Connection,
    inbound: Mapping[str, Any],
    logger: Any,
    state: _GenerationState,
    clips: Mapping[tuple[str, ...], Any] | None = None,
    lane: Any = None,
) -> None:
    """Run one request and report it, admission first.

    The step-logger span brackets this request, as connector telemetry names none.
    """
    request_id = inbound["request_id"]
    queue = None
    try:
        stream_first_token = bool(inbound.get("stream_first_token"))
        prompt, params = _prepare(inbound)
        expanded = message.MEDIA in inbound
        if expanded:
            prompt = await _with_clips(prompt, inbound, clips, lane)
        logger.open_span(request_id)
        queue = await llm.add_request(
            request_id, prompt, params, priority=inbound.get("priority", 0)
        )
        state.admitted = state.admission_done = True
        if state.cancel_requested:
            return
        conn.send({"kind": message.ADMITTED, "request_id": request_id})

        stream = _Stream(inbound["sampling"]["output_kind"], expanded)
        await _read_outputs(queue, state, conn, request_id, stream, stream_first_token)
        span = logger.close_span(request_id)
        output = stream.project()
        conn.send(
            {
                "kind": message.OUTPUT,
                "request_id": request_id,
                "output": output,
                message.TRANSFER: span,
            }
        )
    except Exception as exc:
        state.admission_done = True
        conn.send(
            {
                "kind": message.ERROR,
                "request_id": request_id,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        # On failure too: an open span is fed by every later request.
        logger.close_span(request_id)
        if queue is not None:
            queue.close()
        await _cancel_generation(llm, request_id, state)


async def _with_clips(
    prompt: Any,
    inbound: Mapping[str, Any],
    clips: Mapping[tuple[str, ...], Any] | None,
    lane: Any,
) -> Any:
    """`prompt` carrying the clips `inbound` names, processed on `lane` if any."""
    from turboreflect.program.media import split_request_fields

    prompt.update(
        split_request_fields(clips, inbound[message.MEDIA], inbound.get("cache_salt"))
    )
    if lane is not None:
        prompt = await lane.prepare(prompt)
    return prompt


async def _read_outputs(
    queue: Any,
    state: _GenerationState,
    conn: Connection,
    request_id: str,
    stream: _Stream,
    stream_first_token: bool,
) -> None:
    """Fold a request's outputs into `stream` until it finishes."""
    finished = False
    reported_first = False
    while not finished:
        out = queue.get_nowait() or await queue.get()
        finished = out.finished
        state.finished = finished
        if stream_first_token and not reported_first and _has_token(out):
            reported_first = True
            conn.send({"kind": message.FIRST_TOKEN, "request_id": request_id})
        stream.add(out)


def _prepare(inbound: Mapping[str, Any]) -> tuple[Any, Any]:
    """The prompt and the sampling parameters, in vLLM's own types.

    `kv_transfer_params` goes into `extra_args` unrebuilt, under the key
    `vllm/v1/request.py` reads: a missing field stops D renewing P's lease.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    fields = dict(inbound["sampling"])
    fields["output_kind"] = RequestOutputKind[fields["output_kind"].upper()]
    transfer_params = inbound.get("kv_transfer_params")
    if transfer_params is not None:
        extra_args = dict(fields.get("extra_args") or {})
        extra_args[message.KV_TRANSFER_PARAMS_KEY] = transfer_params
        fields["extra_args"] = extra_args

    prompt = TokensPrompt(prompt_token_ids=list(inbound["prompt_token_ids"]))
    salt = inbound.get("cache_salt")
    if salt:
        prompt["cache_salt"] = salt
    return prompt, SamplingParams(**fields)


class _Stream:
    """One request's outputs as the one answer the driver is sent.

    A `delta` stream's completion is joined here in order; everything else is the
    last output's.
    """

    def __init__(self, output_kind: str, expanded: bool = False) -> None:
        self._delta = output_kind == message.DELTA
        self._expanded = expanded
        self._token_ids: list[int] = []
        self._text: list[str] = []
        self._last: Any = None

    def add(self, out: Any) -> None:
        self._last = out
        if self._delta:
            for completion in out.outputs:
                self._token_ids.extend(completion.token_ids)
                self._text.append(completion.text)

    def project(self) -> message.RemoteOutput:
        """The whole request, as it crosses the pipe."""
        projected = message.project(self._last, self._expanded)
        if not self._delta:
            return projected
        # One completion per request.
        [last] = projected.outputs
        joined = dataclasses.replace(
            last, token_ids=tuple(self._token_ids), text="".join(self._text)
        )
        return dataclasses.replace(projected, outputs=(joined,))


def _has_token(out: Any) -> bool:
    """Whether this output carries a sampled token; one reported at
    `WAITING_FOR_REMOTE_KVS` may not, and would time the token before its KV.
    """
    return bool(out.outputs) and bool(out.outputs[0].token_ids)

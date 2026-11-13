# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""One stage across two engines, and the cleanup for a decode that never ran.

The handover rests on three properties of vLLM's `NixlConnector`:

- It never reads `kv_role`, so a request without P's transfer params runs locally:
  D prefills what its own cache lacks (`generate_local`).
- D's scheduler heartbeats P's lease from enqueue, but `on_new_request` returns
  silently when a field is missing and the lease lapses, so P's params are handed
  on as P returned them.
- Pull mode has no watchdog: a request whose KV never arrives sits in
  `WAITING_FOR_REMOTE_KVS`, heartbeating P's blocks. `request_deadline_s`, from
  handover to first token, is the only bound; past it `_clean_up` releases P.

Engine cores stamp `time.monotonic()`, one CLOCK_MONOTONIC only on one host; every
worker is spawned on this host, so engine timestamps compare across engines.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import multiprocessing
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from turboreflect.locality_pd import message
from turboreflect.locality_pd.admission import Ledger
from turboreflect.locality_pd.config import (
    ROLE_D,
    ROLE_P,
    PDSettings,
)
from turboreflect.locality_pd.dispatch import (
    Assignment,
    Dispatcher,
    EngineView,
    WorkItem,
    p_decode_line,
)
from turboreflect.locality_pd.policy import (
    EngineCosts,
    StepModel,
)
from turboreflect.locality_pd.worker import AFTER_PINNING, worker_main
from turboreflect.metrics.metrics import Handoff, TransferWindow, split_engine

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from turboreflect.config import RunConfig

# The cleanup action per state, so exactly one follows; a finished request needs none.
PRE_ADMISSION_FAILURE = "pre_admission_failure"
ADMITTED = "admitted"
FINISHED = "finished"

CLEANUP = {
    PRE_ADMISSION_FAILURE: message.REJECT,
    ADMITTED: message.ABORT,
    FINISHED: None,
}

_PER_REQUEST = (
    message.ADMITTED,
    message.FIRST_TOKEN,
    message.OUTPUT,
    message.ERROR,
)

# Pause between refused resets, and grace past the longest hold (`drain`).
_DRAIN_PAUSE_S = 0.5
_DRAIN_GRACE_S = 5.0

# How long a worker gets to leave on its own before the card is taken back.
_STOP_GRACE_S = 30.0

# The dispatcher places at most once in this long, however many events wake it.
_PUMP_INTERVAL_S = 0.01


class PDDriverError(Exception):
    """A worker failed, or a stage did not finish within its deadline."""


def prefill_transfer_params() -> dict[str, bool]:
    """The params that make P's `request_finished` export its blocks, not free them;
    a new dict per call, since the connector writes into the one it is given.
    """
    return {"do_remote_prefill": False, "do_remote_decode": True}


def prefill_sampling(sampling: Mapping[str, Any]) -> dict[str, Any]:
    """P's sampling: D's draw settings, one token, yielded only at the end, since D
    samples the stage's first token itself and nobody times P's.
    """
    return {**sampling, "max_tokens": 1, "output_kind": message.FINAL_ONLY}


def _decode_sampling(sampling: Mapping[str, Any]) -> dict[str, Any]:
    """D's sampling without `PREFILL_TAG`, which must mark only a local prefill."""
    d_sampling = dict(sampling)
    d_extra = dict(d_sampling.get("extra_args") or {})
    d_extra.pop(message.PREFILL_TAG, None)
    if d_extra or "extra_args" in d_sampling:
        d_sampling["extra_args"] = d_extra
    return d_sampling


def _local_params(
    export_for_pull: bool, pull_params: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Transfer params of a stage one engine runs alone; None holds nothing."""
    params: dict[str, Any] | None = (
        {"do_remote_prefill": False, "do_remote_decode": False}
        if export_for_pull
        else None
    )
    if pull_params is not None:
        # Pulled as a P pulls, then decoded here and held as a D holds.
        params = {
            **pull_params,
            "decode_after_pull": True,
            **({} if export_for_pull else {"decoder_kv_blocks_hold": False}),
        }
    return params


@dataclass(frozen=True)
class SplitStage:
    """One stage as two engines ran it, and the gap between them.

    Attributes:
        p_output: The prefill engine's answer; its token is not the stage's.
        to_first_token_wall_s: Driver clock from P's output to D's first token, both
            read as the messages land, so a busy loop stays out of the number.
        ttft_wall_s: Driver clock from sending P the request to D's first token.
        d: The decode engine, whose clock the stage's engine timestamps are on.
        transfer: What D's connector moved over the request's span; None when the
            worker counted no span.
    """

    p_output: message.RemoteOutput
    d_output: message.RemoteOutput
    to_first_token_wall_s: float
    ttft_wall_s: float
    d: str
    transfer: TransferWindow | None = None

    def handoff(self) -> Handoff:
        """P's half of this stage, in the shape the stage record takes."""
        return Handoff.from_prefill(
            self.p_output,
            to_first_token_wall_s=self.to_first_token_wall_s,
            ttft_wall_s=self.ttft_wall_s,
            transfer=self.transfer,
            engine=split_engine(self.d),
        )


@dataclass(frozen=True)
class LocalStage:
    """One stage its D ran alone; fields as `SplitStage`, timed from sending D."""

    d_output: message.RemoteOutput
    ttft_wall_s: float
    d: str
    transfer: TransferWindow | None = None

    def handoff(self) -> Handoff:
        """The record's P/D half: D's label and clock, and no P at all."""
        return Handoff(
            engine=split_engine(self.d),
            ttft_wall_s=self.ttft_wall_s,
            transfer=self.transfer,
        )


class Link:
    """The driver's end of one worker: a process to join, a pipe to read."""

    def __init__(self, process: Any, conn: Connection) -> None:
        self._process = process
        self._conn = conn
        self._reader = message.PipeReader(conn)

    def send(self, payload: dict[str, Any]) -> None:
        self._conn.send(payload)

    async def recv(self) -> dict[str, Any]:
        """The next message, read on its own thread so it never waits on renders."""
        return await self._reader.recv()

    def stop(self, grace_s: float) -> None:
        """Join the worker, then close the pipe: a worker writing to a closed reader
        blocks in its event loop and never reaches `llm.shutdown()`.
        """
        self._process.join(grace_s)
        if self._process.exitcode is None:
            self._process.terminate()
            self._process.join(grace_s)
        self._conn.close()


def spawn_worker(
    name: str, cfg: RunConfig, settings: PDSettings, steps_path: str
) -> Link:
    """Start one engine in a spawned interpreter; a fork would inherit vLLM imports."""
    context = multiprocessing.get_context("spawn")
    driver_end, worker_end = context.Pipe(duplex=True)
    process = context.Process(
        target=worker_main,
        args=(name, cfg, settings, worker_end, steps_path),
        name=f"pd-{name}",
        # AsyncLLM starts engine-core children, which a daemonic process may not have.
        daemon=False,
    )
    process.start()
    # Else a dead worker leaves `recv` waiting on a pipe this process holds open.
    worker_end.close()
    return Link(process, driver_end)


@dataclass
class _Pending:
    """One submitted request; `admitted` and `returned_at` give its cleanup state."""

    request_id: str
    first_token: asyncio.Future
    output: asyncio.Future
    admitted: bool = False
    returned_at: float | None = None
    # The worker's step-logger transfer window, carried only by `output`.
    transfer: TransferWindow | None = None

    @property
    def state(self) -> str:
        if self.returned_at is not None:
            return FINISHED
        return ADMITTED if self.admitted else PRE_ADMISSION_FAILURE

    def fail(self, exc: Exception) -> None:
        if not self.output.done():
            self.output.set_exception(exc)


class _Side:
    """One worker as the driver talks to it: a reader task routes each message to
    its waiting future, and the latest pushed `headroom` is kept.
    """

    def __init__(self, name: str, link: Any) -> None:
        loop = asyncio.get_running_loop()
        self.name = name
        self.link = link
        self.report: dict[str, Any] | None = None
        self.headroom: dict[str, Any] | None = None
        # Called with this side's name on each push, so a waiting stage looks again.
        self.on_headroom: Callable[[str], None] | None = None
        self._pending: dict[str, _Pending] = {}
        self._replies: dict[str, asyncio.Queue] = {
            kind: asyncio.Queue() for kind in (message.RESET, message.STEP_TOTALS)
        }
        self._ready: asyncio.Future = loop.create_future()
        self._broken: Exception | None = None
        self._reader = loop.create_task(self._read(), name=f"pd-read-{name}")

    def send(self, payload: dict[str, Any]) -> None:
        """Send to this worker; once it is gone, raise a named `PDDriverError`."""
        if self._broken is not None:
            raise PDDriverError(
                f"the {self.name} worker is no longer answering, so it cannot be "
                f"sent {payload.get('kind')!r}"
            ) from self._broken
        self.link.send(payload)

    async def wait_ready(self) -> dict[str, Any]:
        """What this engine said about itself once it was built."""
        self.report = await self._ready
        return self.report

    def expect(self, request_id: str) -> _Pending:
        """Register a request before it is sent, so no reply can arrive first."""
        loop = asyncio.get_running_loop()
        pending = _Pending(request_id, loop.create_future(), loop.create_future())
        self._pending[request_id] = pending
        return pending

    def forget(self, request_id: str) -> None:
        self._pending.pop(request_id, None)

    async def ask(self, payload: dict[str, Any], reply: str) -> dict[str, Any]:
        """Send one request and await its reply; one in flight per reply kind."""
        self.send(payload)
        answer = await self._replies[reply].get()
        if isinstance(answer, Exception):
            raise answer
        return answer

    def fail(self, exc: Exception) -> None:
        """Give up on this side, resolving every future it owns so no waiter hangs."""
        if self._broken is None:
            self._broken = exc
        if not self._ready.done():
            self._ready.set_exception(exc)
        for pending in list(self._pending.values()):
            pending.fail(exc)
        for queue in self._replies.values():
            queue.put_nowait(exc)

    async def close(self, grace_s: float) -> None:
        """Stop reading, stop the worker, and leave nothing unretrieved."""
        self.fail(PDDriverError(f"the {self.name} worker was shut down"))
        self._reader.cancel()
        await asyncio.gather(self._reader, return_exceptions=True)
        self.link.stop(grace_s)
        # Retrieved, so asyncio does not report them at collection time.
        for future in [self._ready, *(p.output for p in self._pending.values())]:
            if future.done() and not future.cancelled():
                future.exception()

    async def _read(self) -> None:
        try:
            while True:
                self._deliver(await self.link.recv())
        except (EOFError, OSError):
            self.fail(
                PDDriverError(
                    f"the {self.name} worker's pipe closed without it saying why "
                    "-- the process died rather than reporting a failure"
                )
            )
        except Exception as exc:  # noqa: BLE001 - the caller is owed the reason
            self.fail(exc)

    def _deliver(self, payload: Mapping[str, Any]) -> None:
        """Hand one message to its waiter; an unknown kind raises, not hangs it."""
        # Driver clock at landing; worker timestamps are on its own monotonic clock.
        at = time.perf_counter()
        kind = payload.get("kind")
        if kind == message.READY:
            if not self._ready.done():
                self._ready.set_result(dict(payload))
        elif kind == message.FAILED:
            self.fail(
                PDDriverError(
                    f"the {self.name} worker failed before it was serving: "
                    f"{payload.get('error')}\n{payload.get('traceback')}"
                )
            )
        elif kind == message.HEADROOM:
            self.headroom = dict(payload)
            if self.on_headroom is not None:
                self.on_headroom(self.name)
        elif kind in self._replies:
            self._replies[kind].put_nowait(dict(payload))
        elif kind in _PER_REQUEST:
            self._route(kind, payload, at)
        else:
            raise PDDriverError(
                f"the {self.name} worker sent {kind!r}, which this driver does "
                f"not handle; the protocol is {list(message.TO_DRIVER)}"
            )

    def _route(self, kind: str, payload: Mapping[str, Any], at: float) -> None:
        request_id = payload["request_id"]
        pending = self._pending.get(request_id)
        if pending is None:
            return
        if kind == message.ADMITTED:
            pending.admitted = True
        elif kind == message.FIRST_TOKEN:
            if not pending.first_token.done():
                pending.first_token.set_result(at)
        elif kind == message.OUTPUT:
            pending.returned_at = at
            pending.transfer = payload.get(message.TRANSFER)
            if not pending.output.done():
                pending.output.set_result(payload["output"])
        else:
            pending.fail(
                PDDriverError(
                    f"the {self.name} worker failed {request_id!r}: "
                    f"{payload.get('error')}\n{payload.get('traceback')}"
                )
            )


class PDDriver:
    """Every engine of the run, started together, keyed by name.

    Each engine lives in a worker process that pinned its device first; this process
    talks to it over a pipe and never imports vLLM (`_refuse_a_polluted_parent`).
    """

    # Program-level admission (`open_ledger`) and the stage queue (`open_dispatcher`).
    ledger: Ledger | None = None
    dispatcher: Dispatcher | None = None

    def __init__(
        self,
        cfg: RunConfig,
        settings: PDSettings,
        steps_paths: Mapping[str, str],
        launch: Any = spawn_worker,
    ) -> None:
        """Hold the run's engines, unstarted; each trace goes beside its steps path."""
        self._cfg = cfg
        self._settings = settings
        self._steps_paths = dict(steps_paths)
        self._launch = launch
        self._sides: dict[str, _Side] = {}
        # Tokens sent to each P to write, and the count per program's last visit.
        self._p_written: dict[str, int] = dict.fromkeys(settings.names(ROLE_P), 0)
        self._p_seen: dict[tuple[str, str], int] = {}
        self.facts: dict[str, message.EngineFacts] = {}
        # Engine clip layouts by clip paths; empty unless the run places by position.
        self.clip_layouts: dict[tuple[str, ...], Any] = {}

    async def __aenter__(self) -> PDDriver:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.stop()

    async def start(self) -> None:
        """Start every engine, all spawned before any is waited on.

        Engines sharing a device start one at a time: an engine sizes its KV pool
        from free memory while it profiles, so a neighbour loading beside it leaves
        it no KV budget.

        Raises:
            PDDriverError: On a missing path, a polluted parent, or a dead worker.
        """
        names = self._settings.names()
        missing = [name for name in names if not self._steps_paths.get(name)]
        if missing:
            raise PDDriverError(
                f"no path was given for {missing}: each engine keeps its trace "
                "in the run directory, so no worker is started"
            )
        _refuse_a_polluted_parent()
        devices = [d for name in names for d in self._settings.devices(name)]
        one_at_a_time = len(devices) != len(set(devices))
        try:
            reports = []
            for name in names:
                link = self._launch(
                    name, self._cfg, self._settings, self._steps_paths.get(name)
                )
                self._sides[name] = _Side(name, link)
                self._sides[name].on_headroom = self._pushed
                if one_at_a_time:
                    reports.append(await self._sides[name].wait_ready())
            if not one_at_a_time:
                reports = await asyncio.gather(
                    *(self._sides[name].wait_ready() for name in names)
                )
            facts = {
                name: message.EngineFacts(**report["facts"])
                for name, report in zip(names, reports)
            }
        except BaseException:
            await self.stop()
            raise
        self.facts = facts
        # Kept off `report`, which is written beside the run's config.
        for report in reports:
            self.clip_layouts.update(report.pop(message.CLIP_LAYOUTS, None) or {})

    async def stop(self, grace_s: float = _STOP_GRACE_S) -> None:
        """Shut every engine down, failing whatever still waits on one."""
        for side in self._sides.values():
            # A worker that has already died cannot be told to stop.
            with suppress(Exception):
                side.send({"kind": message.SHUTDOWN})
        for side in self._sides.values():
            await side.close(grace_s)
        self._sides = {}

    def report(self) -> dict[str, dict[str, Any]]:
        """What each engine said about itself once it was up, by name."""
        return {name: dict(side.report or {}) for name, side in self._sides.items()}

    def leave(self, program_id: str) -> None:
        """Forget a finished program's stages and reservations."""
        for key in [key for key in self._p_seen if key[0] == program_id]:
            del self._p_seen[key]
        if self.ledger is not None:
            self.ledger.release(program_id)
        if self.dispatcher is not None:
            self.dispatcher.leave(program_id)
        self._wake()

    def open_ledger(self, homes: Sequence[str], watermark: float) -> Ledger:
        """Start program-level admission on `homes`, by KV pool and `max_num_seqs`.

        Args:
            homes: The engines a program's stages may decode on.
            watermark: The share of each pool no reservation may take.
        """
        capacity, slots = {}, {}
        for name in homes:
            pool = (self._side(name).report or {}).get("kv_pool") or {}
            tokens = pool.get("capacity_tokens")
            if not isinstance(tokens, int) or tokens <= 0:
                raise PDDriverError(f"{name}: no measured KV pool to admit against")
            capacity[name] = tokens // self.facts[name].block_size
            slots[name] = self._settings.engine(name).max_num_seqs
        self.ledger = Ledger(
            capacity,
            {name: int(watermark * blocks) for name, blocks in capacity.items()},
            slots,
            block_tokens={name: self.facts[name].block_size for name in homes},
        )
        return self.ledger

    def open_dispatcher(
        self,
        step: StepModel,
        kv_transfer_s: Callable[[int], float],
        *,
        tail_programs_per_engine: int = 2,
        drain_on_p: bool = True,
        role_flex: bool = True,
        tpot_delta: float | None = None,
    ) -> Dispatcher:
        """Place every stage through one `Dispatcher`, priced by `step` and pushes."""
        self._step = step
        self._placing: dict[str, asyncio.Future] = {}
        self._permits: dict[str, asyncio.Future] = {}
        self._pump_handle: asyncio.TimerHandle | None = None
        self._pumped_at = -math.inf
        self.dispatcher = Dispatcher(
            self.ledger,
            {name: self._settings.role(name) for name in self._settings.names()},
            kv_transfer_s,
            lease_s=self._settings.kv_lease_duration_s,
            tail_programs_per_engine=tail_programs_per_engine,
            drain_on_p=drain_on_p,
            role_flex=role_flex,
            tpot_delta=tpot_delta,
            shared_devices=[
                frozenset(pair)
                for pair in itertools.combinations(self._settings.names(), 2)
                if set(self._settings.engine(pair[0]).devices)
                & set(self._settings.engine(pair[1]).devices)
            ],
        )
        return self.dispatcher

    def resize(self, program_id: str, peak_blocks: int) -> None:
        """Lower the program's reservation (`Ledger.resize`) and place what fits."""
        if self.ledger is not None and program_id in self.ledger.reservations:
            self.ledger.resize(program_id, peak_blocks)
            self._wake()

    async def render_permit(self, program_id: str) -> float:
        """Wait for `Dispatcher.permits` to let the program render; return the wait."""
        dispatcher = self.dispatcher
        future = asyncio.get_running_loop().create_future()
        self._permits[program_id] = future
        dispatcher.announce(program_id)
        self._wake()
        started = time.monotonic()
        try:
            done, _ = await asyncio.wait(
                (future,), timeout=self._settings.request_deadline_s
            )
            if not done:
                raise PDDriverError(
                    f"{program_id}: not let render its generation stage within "
                    f"{self._settings.request_deadline_s:.0f}s"
                )
            future.result()
        finally:
            self._permits.pop(program_id, None)
        return time.monotonic() - started

    async def dispatch(self, item: WorkItem) -> Assignment:
        """Queue a stage and wait for its place.

        A generation stage's KV peak is reserved on its home before any prompt is sent.

        Raises:
            AdmissionError: If the program can never be admitted.
        """
        dispatcher = self.dispatcher
        future = asyncio.get_running_loop().create_future()
        self._placing[item.request_id] = future
        dispatcher.submit(self._unevicted(item))
        self._wake()
        try:
            done, _ = await asyncio.wait(
                (future,), timeout=self._settings.request_deadline_s
            )
            if not done:
                raise PDDriverError(
                    f"{item.request_id}: not placed within "
                    f"{self._settings.request_deadline_s:.0f}s"
                )
            return future.result()
        finally:
            self._placing.pop(item.request_id, None)
            dispatcher.withdraw(item.request_id)

    def _unevicted(self, item: WorkItem) -> WorkItem:
        """The item with no hit predicted on a non-home P that lost the blocks."""
        evicted = [
            p
            for p in self._evicted_on_p(item.program_id, list(self._p_written))
            if p != item.home and item.hits.get(p)
        ]
        if not evicted:
            return item
        return replace(item, hits={**item.hits, **dict.fromkeys(evicted, 0)})

    def _placed(self, item: WorkItem, assignment: Assignment) -> None:
        """Book what a stage placed at its handoff writes into a P's pool."""
        if assignment.engine in self._p_written:
            hit = min(item.hits.get(assignment.engine, 0), item.prompt_tokens)
            self._p_written[assignment.engine] += item.prompt_tokens - hit
            self._p_seen[(item.program_id, assignment.engine)] = self._p_written[
                assignment.engine
            ]

    def _views(self) -> dict[str, EngineView]:
        """Every engine as its last push showed it, priced at its own rates.

        A P that speculates as D does (`RunConfig.pd_p_speculates`) with no timed
        decode step yet takes the Ds' mean accepted and query tokens per row.
        """
        views = {}
        for name in self._settings.names():
            state = (self._side(name).headroom or {}).get("locality_pd") or {}
            views[name] = EngineView(
                self._settings.role(name),
                EngineCosts.from_state(self._step, state),
                state,
            )
        if not self._cfg.pd_p_speculates():
            return views
        measured = [
            views[d].costs
            for d in self._settings.names(ROLE_D)
            if views[d].state.get("accept_per_row")
        ]
        if not measured:
            return views
        accept = math.fsum(c.accept_per_row for c in measured) / len(measured)
        query = math.fsum(c.query_per_request for c in measured) / len(measured)
        for p in self._settings.names(ROLE_P):
            if not views[p].state.get("accept_per_row"):
                costs = replace(
                    views[p].costs, accept_per_row=accept, query_per_request=query
                )
                views[p] = replace(views[p], costs=costs)
        return views

    def _wake(self) -> None:
        """Have the dispatcher place what it can, at most once a `_PUMP_INTERVAL_S`."""
        if self.dispatcher is None or self._pump_handle is not None:
            return
        loop = asyncio.get_running_loop()
        self._pump_handle = loop.call_at(
            max(loop.time(), self._pumped_at + _PUMP_INTERVAL_S), self._pump
        )

    def _pump(self) -> None:
        self._pump_handle = None
        dispatcher = self.dispatcher
        if dispatcher is None:
            return
        self._pumped_at = asyncio.get_running_loop().time()
        try:
            placed = dispatcher.pump(self._views())
            granted = dispatcher.permits()
        except Exception as exc:  # noqa: BLE001 - every waiter is owed it
            for waiting in (*self._placing.values(), *self._permits.values()):
                if not waiting.done():
                    waiting.set_exception(exc)
            return
        for item, outcome in placed:
            placing = self._placing.get(item.request_id)
            if placing is None or placing.done():
                continue
            if isinstance(outcome, Exception):
                placing.set_exception(outcome)
            else:
                self._placed(item, outcome)
                note = outcome.record.get("p_decode")
                if note is not None:
                    print(f"dispatch: {item.request_id} {p_decode_line(note)}")
                placing.set_result(outcome)
        for program_id in granted:
            permit = self._permits.get(program_id)
            if permit is not None and not permit.done():
                permit.set_result(None)

    def _pushed(self, engine: str) -> None:
        """Hand every push to the ledger and dispatcher; a later one may omit a read."""
        state = (self._side(engine).headroom or {}).get("locality_pd") or {}
        if self.dispatcher is not None:
            self.dispatcher.pushed(engine, state)
        if self.ledger is not None and engine in self.ledger and "free_blocks" in state:
            self.ledger.reconcile(
                engine,
                state["free_blocks"],
                state.get("inflight_reserved_blocks", 0),
            )
        self._wake()

    def _evicted_on_p(self, program_id: str, prefills: Sequence[str]) -> list[str]:
        """The Ps sent more than their pool since the program last ran there.

        Past a pool's worth, P's prefix cache has handed the program's blocks on,
        which the reuse ledger cannot see.
        """
        evicted = []
        for p in prefills:
            seen = self._p_seen.get((program_id, p))
            pool = ((self._side(p).report or {}).get("kv_pool") or {}).get(
                "capacity_tokens"
            )
            if (
                seen is not None
                and isinstance(pool, int)
                and self._p_written[p] - seen > pool
            ):
                evicted.append(p)
        return evicted

    def _handed_off(self, request_id: str) -> None:
        """P has answered: the stage leaves P's horizon and is on its way to D."""
        if self.dispatcher is not None:
            self.dispatcher.handed_off(request_id)
            self._wake()

    def _landed(self, request_id: str) -> None:
        if self.dispatcher is not None:
            self.dispatcher.first_token(request_id)
            self._wake()

    def _stage_over(self, request_id: str) -> None:
        """The stage has returned, or will not."""
        if self.dispatcher is not None:
            self.dispatcher.finished(request_id)
            self._wake()

    def release_hold(self, d: str, request_id: str) -> None:
        """Let an engine free a finished request's blocks before its hold runs out.

        Unanswered: an id no longer held is ignored, and a failed release leaves the
        hold to `decoder_kv_blocks_ttl_s`, or a P's to its lease.

        Args:
            d: The holding engine, a D or a P a program moved to.
            request_id: The engine's `remote_request_id` for the request.
        """
        self._side(d).send({"kind": message.RELEASE_HOLD, "request_id": request_id})

    def _first_token(
        self,
        request_id: str,
        holder: str,
        held: str | None,
        on_first_token: Callable[[], None] | None,
    ) -> None:
        """A stage's first token landed: let go of the stage before's hold, if any."""
        self._landed(request_id)
        if held:
            self.release_hold(holder, held)
        if on_first_token:
            on_first_token()

    async def generate_stage(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        cache_salt: str | None,
        sampling: Mapping[str, Any],
        *,
        p: str,
        d: str,
        prefill_params: Mapping[str, Any] | None = None,
        priority: int = 0,
        export_for_pull: bool = True,
        held: str | None = None,
        on_first_token: Callable[[], None] | None = None,
        media: Mapping[str, Any] | None = None,
    ) -> SplitStage:
        """Prefill one stage on P and decode it on D, with the KV pulled across.

        D is submitted the moment P returns: P's lease starts at its
        `request_finished`, so prefilling a batch first would hand D requests whose
        blocks P had already released.

        Args:
            cache_salt: The program's salt, on both sides so P's cache keys are the
                ones the next stage hits.
            sampling: D's sampling; P's is derived from it (`prefill_sampling`).
            prefill_params: None for `prefill_transfer_params()`, or the previous
                stage's D params as returned, so that P first pulls that history
                from D; the deadline then bounds P's answer too.
            priority: vLLM's request priority on P and D, each under its role's policy.
            export_for_pull: Whether D keeps the stage's blocks for a later P to
                pull; False sets `DECODER_KV_HOLD_KEY` False in D's params.
            held: D's id for the previous stage's hold, released at this stage's
                first token on D, after any pull of it; None releases nothing.
            media: The clips behind the placeholders (`program.media.clip_ref`), sent
                to both: P encodes them, D needs them for the same block keys.

        Raises:
            PDDriverError: If a pulling P or D's first token misses the deadline.
        """
        try:
            p_sampling = prefill_sampling(sampling)
            d_sampling = _decode_sampling(sampling)
            prefill, decode = self._side(p), self._side(d)
            p_out, sent_at, returned_at = await self._prefill(
                prefill,
                request_id,
                prompt_token_ids,
                cache_salt,
                p_sampling,
                prefill_transfer_params() if prefill_params is None else prefill_params,
                priority,
                None if prefill_params is None else self._settings.request_deadline_s,
                media=media,
            )
            self._handed_off(request_id)
            d_params = p_out.kv_transfer_params
            if not export_for_pull:
                d_params = {**d_params, message.DECODER_KV_HOLD_KEY: False}
            d_out, first_at, transfer = await self._decode(
                decode,
                request_id,
                prompt_token_ids,
                cache_salt,
                d_sampling,
                d_params,
                priority,
                on_first_token=lambda: self._first_token(
                    request_id, d, held, on_first_token
                ),
                media=media,
            )
        finally:
            self._stage_over(request_id)
        return SplitStage(
            p_output=p_out,
            d_output=d_out,
            to_first_token_wall_s=first_at - returned_at,
            ttft_wall_s=first_at - sent_at,
            d=d,
            transfer=transfer,
        )

    async def generate_local(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        cache_salt: str | None,
        sampling: Mapping[str, Any],
        *,
        d: str,
        priority: int = 0,
        export_for_pull: bool = False,
        held: str | None = None,
        held_by: str | None = None,
        pull_params: Mapping[str, Any] | None = None,
        on_first_token: Callable[[], None] | None = None,
        media: Mapping[str, Any] | None = None,
    ) -> LocalStage:
        """Prefill and decode one stage on `d` alone, against its own cache.

        With no transfer params the connector answers `(0, False)`, so `d` prefills
        what its prefix cache lacks: for a later stage, the append. `d` may be a P
        the program moves to: given the old home's `pull_params`, it pulls the
        matched prefix first, then holds its own blocks, let go at once when no
        later stage reads them. `held` (held by `held_by`, `d` when None) is
        released at the first token; other arguments are `generate_stage`'s.
        """
        decode = self._side(d)
        sent_at = time.perf_counter()
        params = _local_params(export_for_pull, pull_params)
        try:
            d_out, first_at, transfer = await self._decode(
                decode,
                request_id,
                prompt_token_ids,
                cache_salt,
                sampling,
                params,
                priority,
                on_first_token=lambda: self._first_token(
                    request_id, held_by or d, held, on_first_token
                ),
                media=media,
            )
        finally:
            self._stage_over(request_id)
        own = (d_out.kv_transfer_params or {}).get("remote_request_id")
        if pull_params is not None and not export_for_pull and own:
            self.release_hold(d, own)
        return LocalStage(
            d_output=d_out,
            ttft_wall_s=first_at - sent_at,
            d=d,
            transfer=transfer,
        )

    async def drain(self, timeout_s: float | None = None) -> None:
        """Reset every prefix cache, which waits out P's leases and D's holds.

        `reset_prefix_cache` refuses while any block is held (a lease, or a D's
        `decoder_kv_blocks_ttl_s` when no P pulled), so retrying is the wait. Ds go
        first: a D finishing is what tells P to release.

        Args:
            timeout_s: How long to keep asking; default the longer hold plus a grace.
        """
        hold = max(
            self._settings.kv_lease_duration_s,
            self._settings.decoder_kv_blocks_ttl_s or 0.0,
        )
        limit = hold + _DRAIN_GRACE_S if timeout_s is None else timeout_s
        for role in (ROLE_D, ROLE_P):
            for name in self._settings.names(role):
                await self._reset(name, limit)

    def open_window(self, label: str) -> None:
        """Start a fresh step-counting window in every engine."""
        for side in self._sides.values():
            side.send({"kind": message.OPEN_WINDOW, "label": label})

    def reset_steps(self) -> None:
        """Re-base every run total, so the warmup's steps leave with it."""
        for side in self._sides.values():
            side.send({"kind": message.RESET_RUN})

    async def step_totals(self, span: str) -> dict[str, dict[str, Any] | None]:
        """Every engine's scheduler counters over `run` or `window`, by name."""
        return {
            name: (
                await self._side(name).ask({"kind": message.STEPS}, message.STEP_TOTALS)
            )[span]
            for name in self._settings.names()
        }

    async def _reset(self, name: str, timeout_s: float) -> None:
        """Ask one engine to reset until it agrees to or `timeout_s` runs out."""
        side = self._side(name)
        deadline = time.perf_counter() + timeout_s
        while True:
            remaining = deadline - time.perf_counter()
            try:
                answer = await asyncio.wait_for(
                    side.ask({"kind": message.RESET_PREFIX_CACHE}, message.RESET),
                    timeout=max(remaining, 0.0),
                )
            except asyncio.TimeoutError:
                return
            if answer["ok"]:
                return
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            await asyncio.sleep(min(_DRAIN_PAUSE_S, remaining))

    def _side(self, name: str) -> _Side:
        return self._sides[name]

    async def _prefill(
        self,
        side: _Side,
        request_id: str,
        prompt_token_ids: Sequence[int],
        cache_salt: str | None,
        sampling: Mapping[str, Any],
        params: Mapping[str, Any] | None,
        priority: int,
        deadline_s: float | None = None,
        media: Mapping[str, Any] | None = None,
    ) -> tuple[message.RemoteOutput, float, float]:
        """Run the prompt through P; return its answer and when it was sent and done.

        `deadline_s` bounds a pull from D: past it an admitted request is aborted,
        and any D export it was pulling lapses under vLLM's TTL.
        """
        pending = side.expect(request_id)
        sent_at = time.perf_counter()
        try:
            payload = {
                "kind": message.GENERATE,
                "request_id": request_id,
                "prompt_token_ids": list(prompt_token_ids),
                "cache_salt": cache_salt,
                "sampling": sampling,
                "kv_transfer_params": params,
                "priority": priority,
                "stream_first_token": False,
            }
            if media is not None:
                payload[message.MEDIA] = media
            side.send(payload)
            done, _ = await asyncio.wait((pending.output,), timeout=deadline_s)
            if not done:
                raise PDDriverError(
                    f"{request_id}: no answer from the prefill engine within "
                    f"{deadline_s:.1f}s of submitting the prefill; an admitted "
                    "request is aborted, and remote KV remains under native TTL"
                )
            out = pending.output.result()
        except (asyncio.CancelledError, Exception):
            if pending.state != FINISHED:
                with suppress(Exception):
                    side.send({"kind": message.ABORT, "request_id": request_id})
            raise
        finally:
            side.forget(request_id)
        return out, sent_at, pending.returned_at

    async def _decode(
        self,
        side: _Side,
        request_id: str,
        prompt_token_ids: Sequence[int],
        cache_salt: str | None,
        sampling: Mapping[str, Any],
        params: Any,
        priority: int,
        on_first_token: Callable[[], None] | None = None,
        media: Mapping[str, Any] | None = None,
    ) -> tuple[message.RemoteOutput, float, TransferWindow | None]:
        """Hand the request to D and wait, under the only deadline there is.

        The deadline ends at the first token, once the KV has arrived; a token later
        than the lease is fine, as D heartbeats P's lease while it queues.

        Args:
            params: P's transfer params, or None for a request D runs alone.

        Returns:
            D's answer, its first token's driver-clock time, and its transfer window.
        """
        pending = side.expect(request_id)
        payload = {
            "kind": message.GENERATE,
            "request_id": request_id,
            "prompt_token_ids": list(prompt_token_ids),
            "cache_salt": cache_salt,
            "sampling": dict(sampling),
            # As P returned it: a field lost in a rebuild is a lease D never renews.
            "kv_transfer_params": params,
            "priority": priority,
            "stream_first_token": True,
        }
        if media is not None:
            payload[message.MEDIA] = media
        try:
            side.send(payload)
            done, _ = await asyncio.wait(
                (pending.first_token, pending.output),
                timeout=self._settings.request_deadline_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise PDDriverError(
                    f"{request_id}: no first token from the decode engine within "
                    f"{self._settings.request_deadline_s:.1f}s of the handover. "
                    "Pull mode has no watchdog of its own, so a request whose KV "
                    "never arrives waits in WAITING_FOR_REMOTE_KVS behind a debug "
                    "log for ever, heartbeating P's lease"
                )
            if pending.output.done() and not pending.first_token.done():
                # Raises on a reported failure; else it ended without a token.
                pending.output.result()
                raise PDDriverError(
                    f"{request_id}: the decode engine returned without ever "
                    "reporting a token, so this stage generated nothing"
                )
            first_at = await pending.first_token
            if on_first_token is not None:
                on_first_token()
            out = await pending.output
            return out, first_at, pending.transfer
        except asyncio.CancelledError:
            _clean_up(side, request_id, params, pending)
            raise
        except Exception as exc:
            state, action = _clean_up(side, request_id, params, pending)
            raise PDDriverError(
                f"{request_id}: the decode half of this stage failed ({exc}) -- D "
                f"was {state}, so the driver ran {action or 'no cleanup'}"
            ) from exc
        finally:
            side.forget(request_id)


def _clean_up(
    side: _Side, request_id: str, params: Any, pending: _Pending
) -> tuple[str, str | None]:
    """Release what a failed request left, by how far it got; return state and action.

    A request the engine never admitted needs `notify_kv_transfer_request_rejected`,
    whose dummy pre-aborted request runs `request_finished` so P unpins; an admitted
    one runs it when aborted. The wrong one, or neither, pins P's blocks until the
    lease lapses. A missing admission ack is no proof: the worker settles in-flight
    admission before choosing. A local request has no export, so it is aborted.
    """
    state = pending.state
    action = CLEANUP[state]
    if action == message.REJECT and (
        params is None
        or (
            params.get("do_remote_prefill") is False
            and params.get("do_remote_decode") is False
        )
    ):
        action = message.ABORT
    if action == message.REJECT:
        side.send(
            {
                "kind": message.REJECT,
                "request_id": request_id,
                "kv_transfer_params": params,
            }
        )
    elif action == message.ABORT:
        side.send({"kind": message.ABORT, "request_id": request_id})
    return state, action


def _refuse_a_polluted_parent() -> None:
    """Raise if this process has already imported what the workers may not.

    Spawn re-runs the main module in each child (`_fixup_main_from_name` in
    `multiprocessing/spawn.py`), so an entry point's module-level imports reach every
    worker before `worker_main` pins its device, and each would refuse.
    """
    already = [name for name in AFTER_PINNING if name in sys.modules]
    if not already:
        return
    main = sys.modules.get("__main__")
    entry = getattr(getattr(main, "__spec__", None), "name", None) or getattr(
        main, "__file__", "the entry point"
    )
    raise PDDriverError(
        f"{' and '.join(already)} has been imported into the driver process, and "
        f"spawn re-runs {entry} inside each worker before it can pin a card. Both "
        "engines would attach to whichever devices this process can see and open "
        "the NIXL side channel on the one default port -- import vLLM inside the "
        "functions that need it, as turboreflect.engine.runner does"
    )

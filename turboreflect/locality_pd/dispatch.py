# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""One queue of stages, each placed where the projected makespan grows least.

Every stage of every program is a work item: a generation stage, which no engine
holds any of, or a later stage, whose matched prefix its home (the engine that
decoded the stage before) holds. The min-max rule (`decide_minmax`) places an item
on the route of least projected makespan

    Phi = max(max_i T_i, (sum_i T_i + U) / G)

where T_i is engine i's committed busy seconds at its measured rates (`EngineCosts`),
U the prefill no engine has taken on, and G the engine count. KV room (`Ledger`), the
TPOT window budget, the home's hold and seq slots close routes.

Late binding: a generation stage is offered, in arrival order, only to engines with
less unstarted prefill than `keep_busy_s`, so it is placed on state as of when one
frees up. It takes the best open route unless the rule prefers a P that is not open
and the open route prefills on a D, whose KV and decode budget it would hold. A D
whose TPOT budget paces the stages it took takes no more until that delay is within
`keep_busy_s`. The KV peak is reserved on the home at placement.

Role flexibility: an idle D prefills generation stages, and once no D has room for
the next one a P may decode, homing it if that shortens the pass or taking a later
stage over by pulling the matched prefix from the home's hold. Such a route is
closed (`TPOT_GUARD`) when the P's prefill, run at its own pace, would take the
window's TPOT past `1 + delta` times a pure decode step's (`mixed_tpot_ratio`). A
decoding engine takes on prefill at its budget's share of its time (`prefill_rate`).

The drain: once no generation stage is left, a stage may move to a P with none left
to prefill, or from a P back to a D, only where it starts its decode no later than
at home (while generation stages wait) or ends no later (once none does). In the
tail (few programs left, or one outlasting Phi) items go by critical path, each
where it is done first.

The dispatcher is pure: engine pushes (`EngineView`) and a clock are handed in.
"""

from __future__ import annotations

import itertools
import math
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from turboreflect.locality_pd.admission import AdmissionError, Ledger
from turboreflect.locality_pd.config import D_READ_MARGIN_S, ROLE_D, ROLE_P
from turboreflect.locality_pd.message import HEADROOM_REFRESH_S, submitted_id
from turboreflect.locality_pd.policy import (
    TPOT_GUARD,
    TPOT_WINDOW_S,
    EngineCosts,
    Placement,
    budget_delay_s,
    critical_path_s,
    decide_minmax,
    drain_s,
    makespan_s,
    marginal_drain_s,
    mixed_tpot_ratio,
    prefill_rate,
    steady_decode_s,
)

GENERATION = "generation"
LATER_STAGE = "later_stage"

# Prefill and decode on one engine; prefill on a P that hands the KV to the home;
# or prefill and decode on an engine that becomes the program's home.
LOCAL = "local"
REMOTE_PD = "remote_pd"
REMOTE_HOME = "remote_home"

NO_KV_ROOM = "no_kv_room"
HOLD_UNREADABLE = "hold_unreadable"
HOLD_LAPSES = "hold_lapses"
SLOWER_THAN_HOME = "slower_than_home"
READS_PENDING = "reads_pending"
# The weight of a landed move in its link's running tokens and seconds.
LINK_RATE_WEIGHT = 0.3

# The state a decision was made in, for its record only.
FILL = "fill"
STEADY = "steady"
DRAIN = "drain"
TAIL = "tail"

# Why an item waits: for room on a home, for an engine to take it, or for a D whose
# TPOT budget paces the generation stages it took.
ADMISSION = "admission"
GATE = "gate"
PACED = "paced"

# How many push intervals late binding reads.
PUSH_INTERVALS = 128


def p_decode_line(note: Mapping[str, Any]) -> str:
    """A log line for a record's `p_decode`: a P made a home, or left for a D."""
    move = f"{note['from'] or 'generation'} -> {note['to']}"
    if note["kind"] == "p_to_d_moves":
        return (
            f"P->D move {move}: ends in {note['ends_s']:.1f}s, "
            f"{note['home_ends_s']:.1f}s on its P"
        )
    if "tpot_s" not in note:
        return f"P home {move}: no TPOT budget"
    return (
        f"P home {move}: TPOT {note['tpot_s'] * 1e3:.2f} ms predicted, bound "
        f"{note['bound_s'] * 1e3:.2f} ms, P prefill backlog {note['backlog_s']:.2f}s"
    )


# A later stage: (matched, append, max_tokens).
PlannedStage = tuple[int, int, int]


@dataclass(frozen=True)
class WorkItem:
    """One stage waiting to be placed.

    Attributes:
        hits: Each engine's predicted prefix hit for the prompt, by name.
        home: The engine holding the program's history; None for a generation stage.
        offered_tokens: What the home's hold offers a pull of this prompt.
        hold_remaining_s: Seconds before that hold ends; None if it does not.
        peak_blocks: A generation stage's KV peak over every stage.
        rest: The declared stages after it.
    """

    request_id: str
    program_id: str
    kind: str
    prompt_tokens: int
    max_tokens: int
    hits: Mapping[str, int] = field(default_factory=dict)
    home: str | None = None
    offered_tokens: int = 0
    hold_remaining_s: float | None = None
    peak_blocks: int = 0
    rest: tuple[PlannedStage, ...] = ()
    media: bool = False

    @property
    def cold(self) -> bool:
        return self.kind == GENERATION

    @property
    def decode_tokens(self) -> int:
        """Its output bound and every later stage's."""
        return self.max_tokens + sum(t for _, _, t in self.rest)


@dataclass(frozen=True)
class Assignment:
    """Where a work item runs: prefill on `engine`, decode on `home` from now on."""

    route: str
    engine: str
    home: str
    record: Mapping[str, Any]


@dataclass(frozen=True)
class EngineView:
    """One engine as its last push showed it; `state` is empty before the first."""

    role: str
    costs: EngineCosts
    state: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class _Prefill:
    """A placed prefill not finished; `held` is its hit and pull, `done` last reported.

    `prefilled` is set once a push shows it started and no longer prefilling.
    """

    program: str
    engine: str
    home: str
    held: int
    done: int
    prompt: int
    cold: bool = False
    started: bool = False
    prefilled: bool = False


@dataclass
class _Transfer:
    """KV a placed stage moves between two engines, read by `reader`.

    A pull of the home's hold before the prefill, or (`after`) the home's read of a
    P's prefill once P answers; on the link from `since`, on the wire from `read_at`.
    """

    engines: frozenset[str]
    tokens: int
    reader: str
    after: bool = False
    since: float | None = None
    landed: bool = False
    read_at: float | None = None
    measured: list[float | None] | None = None
    index: int = 0


@dataclass
class _Program:
    """A placed program: its home and the stage it runs."""

    seq: int
    home: str
    request_id: str | None = None
    prompt: int = 0
    max_tokens: int = 0
    rest: tuple[PlannedStage, ...] = ()
    first_at: float | None = None
    over: bool = False
    moving_from: str | None = None
    moved: bool = False


@dataclass
class _Snapshot:
    """Every engine's load as a pump starts, updated as it places.

    `backlog` is placed prefill not done; `warm`, the later stages' part, runs first.
    """

    views: Mapping[str, EngineView]
    rows: dict[str, list[tuple[int, int]]]
    mine: dict[str, tuple[int, int]]
    loads: dict[str, float]
    backlog: dict[str, float]
    warm: dict[str, float]
    unstarted: dict[str, float]
    generations: dict[str, int]
    credit: dict[str, float]
    keep_busy: dict[str, float]
    steady: dict[str, tuple[int, float]]
    rates: dict[str, float]
    pool_s: float
    cold_waiting: int
    unfinished: int
    d_full: bool


@dataclass(frozen=True)
class _Option:
    placement: Placement
    held: int
    pull: bool
    moves: tuple[_Transfer, ...] = ()
    # A route that makes a P a home: the TPOT its rows would see there.
    tpot: Mapping[str, Any] | None = None


class Dispatcher:
    """The run's queue of stages, and what it has placed on each engine.

    Args:
        ledger: The KV ledger; a program's home is one of its engines.
        kv_transfer_s: The prior's seconds to move KV of so many tokens.
        lease_s: How long a P holds a finished stage's blocks.
        tail_programs_per_engine: Unfinished programs per engine that start the tail.
        drain_on_p: Whether a P may decode: home a generation stage or take one over.
        role_flex: Whether an engine may do the other role's work; gates `drain_on_p`.
        tpot_delta: The budget delta until an engine pushes its own; None for none.
        shared_devices: Pairs of engines on one device, whose KV reads slow down
            while the other computes.
    """

    def __init__(
        self,
        ledger: Ledger,
        roles: Mapping[str, str],
        kv_transfer_s: Callable[[int], float],
        *,
        lease_s: float,
        tail_programs_per_engine: int = 2,
        drain_on_p: bool = True,
        role_flex: bool = True,
        tpot_delta: float | None = None,
        shared_devices: Sequence[frozenset[str]] = (),
    ) -> None:
        self.ledger = ledger
        self.shared_devices = set(shared_devices)
        # Per link: running (tokens, seconds) moved, and when its last timed read ended.
        self._link_moved: dict[frozenset[str], tuple[float, float]] = {}
        self._link_landed: dict[frozenset[str], float] = {}
        # Untimed moves by request id until the stage is over; the engines that
        # report their reads, and the last read taken from each.
        self._untimed: dict[str, list[_Transfer]] = {}
        self._reporting: set[str] = set()
        self._reads_taken: dict[str, int] = {}
        self.roles = dict(roles)
        self.kv_transfer_s = kv_transfer_s
        self.lease_s = lease_s
        # How long before a hold ends a pull is declined.
        self.read_margin_s = D_READ_MARGIN_S
        self.tail_programs = tail_programs_per_engine
        self.drain_on_p = drain_on_p and role_flex
        self.role_flex = role_flex
        self.tpot_delta = tpot_delta
        self._clock = time.monotonic
        self._seq = itertools.count()
        # Queued items by request id: arrival order, when queued, the item.
        self._queue: dict[str, tuple[int, float, WorkItem]] = {}
        # Queued items' cheapest prefill (terms of U) and critical path, priced once.
        self._pool: dict[str, float] = {}
        self._paths: dict[str, float] = {}
        # Announced programs in arrival order, and those permitted to render.
        self._announced: dict[str, int] = {}
        self._permitted: set[str] = set()
        self._programs: dict[str, _Program] = {}
        self._requests: dict[str, str] = {}
        self._prefills: dict[str, _Prefill] = {}
        # Sums of submitted generation stages' prompt, decode tokens, peak; count.
        self._generations = [0, 0, 0, 0]
        self._pushed_at: dict[str, float] = {}
        self._intervals: deque[float] = deque(maxlen=PUSH_INTERVALS)
        # Per queued item: admission wait, engine wait, last reckoned, why it waits.
        self._waits: dict[str, list[Any]] = {}
        # By request id, each placed stage's (program, moves) until its first token.
        self._transfers: dict[str, tuple[str, list[_Transfer]]] = {}

    def announce(self, program_id: str) -> None:
        """A program will submit its generation stage once it has rendered it."""
        if program_id not in self._programs:
            self._announced.setdefault(program_id, next(self._seq))

    def permits(self) -> list[str]:
        """Announced programs that may render their generation stage now, in order.

        At most two per P (or engine) plus the peaks the homes' room holds are rendered
        and unplaced: rendering all at once holds the driver's CPU for minutes.
        """
        peak = self._mean(2)
        room = 0
        if peak > 0:
            room = sum(
                max(0, self.ledger.room(e)) // math.ceil(peak)
                for e in self._homes(ROLE_D)
            )
        prefills = sum(r == ROLE_P for r in self.roles.values()) or len(self.roles)
        limit = 2 * prefills + room
        out = sum(p in self._permitted for p in self._announced) + sum(
            item.cold for _, _, item in self._queue.values()
        )
        granted = []
        for program_id in sorted(self._announced, key=self._announced.__getitem__):
            if out >= limit:
                break
            if program_id not in self._permitted:
                self._permitted.add(program_id)
                granted.append(program_id)
                out += 1
        return granted

    def submit(self, item: WorkItem) -> None:
        """Queue an item, a generation stage in its program's arrival order."""
        seq = self._announced.pop(item.program_id, None) if item.cold else None
        self._permitted.discard(item.program_id)
        if item.cold:
            for index, value in enumerate(
                (item.prompt_tokens, item.decode_tokens, item.peak_blocks, 1)
            ):
                self._generations[index] += value
        self._queue[item.request_id] = (
            next(self._seq) if seq is None else seq,
            self._clock(),
            item,
        )

    def withdraw(self, request_id: str) -> None:
        """Take an item off the queue; one placed or gone is ignored."""
        self._queue.pop(request_id, None)
        self._pool.pop(request_id, None)
        self._paths.pop(request_id, None)
        self._waits.pop(request_id, None)

    def _waited(self, request_id: str, why: str | None) -> list[Any]:
        """Book the wait since last reckoned to its old reason; then wait for `why`."""
        now = self._clock()
        entry = self._queue.get(request_id)
        wait = self._waits.setdefault(
            request_id, [0.0, 0.0, now if entry is None else entry[1], GATE]
        )
        wait[0 if wait[3] == ADMISSION else 1] += now - wait[2]
        wait[2], wait[3] = now, why
        return wait

    def leave(self, program_id: str) -> None:
        """Forget a program: its queued items, prefills and place."""
        self._announced.pop(program_id, None)
        self._permitted.discard(program_id)
        program = self._programs.pop(program_id, None)
        if program is not None and program.request_id is not None:
            self._requests.pop(program.request_id, None)
        for rid, (_, _, item) in list(self._queue.items()):
            if item.program_id == program_id:
                self.withdraw(rid)
        for rid, prefill in list(self._prefills.items()):
            if prefill.program == program_id:
                del self._prefills[rid]
        for rid, (program, _) in list(self._transfers.items()):
            if program == program_id:
                del self._transfers[rid]

    def pushed(self, engine: str, state: Mapping[str, Any] | None = None) -> None:
        """A push: intervals set `keep_busy_s`, and each new read times its move."""
        now = self._clock()
        last = self._pushed_at.get(engine)
        if last is not None:
            self._intervals.append(now - last)
        self._pushed_at[engine] = now
        if state is None or "reads" not in state:
            return
        self._reporting.add(engine)
        taken = self._reads_taken.get(engine, 0)
        for seq, request_id, start, end, tokens in state["reads"]:
            if seq > taken:
                self._read(engine, submitted_id(request_id), start, end, tokens)
                taken = seq
        self._reads_taken[engine] = taken

    def handed_off(self, request_id: str) -> None:
        """P has answered: its pulls have landed and its KV is on its way home."""
        prefill = self._prefills.pop(request_id, None)
        program_id, moves = self._transfers.get(request_id, ("", []))
        for move in moves:
            if not move.after:
                self._landed(move)
            elif move.since is None:
                move.since = self._clock()
                if prefill is not None:
                    # Held on P until its home reads it, in its reservation.
                    self.ledger.hold(program_id, prefill.engine, prefill.prompt)

    def first_token(self, request_id: str) -> None:
        """First token on the home: its KV has landed; a move releases its old home."""
        self._prefills.pop(request_id, None)
        _, moves = self._transfers.pop(request_id, ("", []))
        for move in moves:
            self._landed(move)
        program_id = self._requests.get(request_id)
        program = None if program_id is None else self._programs.get(program_id)
        if program_id is None or program is None or program.first_at is not None:
            return
        program.first_at = self._clock()
        # P's hold ends at the read.
        self.ledger.drop_holds(program_id)
        if program.moving_from is not None:
            self.ledger.finish_move(program_id)
            program.moving_from = None

    def finished(self, request_id: str) -> None:
        """The stage returned or failed: undo a move whose first token never landed."""
        self._prefills.pop(request_id, None)
        self._transfers.pop(request_id, None)
        self._untimed.pop(request_id, None)
        program_id = self._requests.pop(request_id, None)
        program = None if program_id is None else self._programs.get(program_id)
        if program_id is None or program is None:
            return
        program.over = True
        self.ledger.drop_holds(program_id)
        if program.moving_from is not None:
            self.ledger.abort_move(program_id)
            program.home, program.moving_from = program.moving_from, None
            program.moved = False
        reservation = self.ledger.reservations.get(program_id)
        if reservation is not None and self.roles[program.home] == ROLE_P:
            # A stage that pulled its history there holds it for P's lease;
            # one computed there, for the decoders' hold, as a D's does.
            reservation.hold_until_s = (
                self._clock() + self.lease_s if program.moved else None
            )

    def pump(
        self, views: Mapping[str, EngineView]
    ) -> list[tuple[WorkItem, Assignment | AdmissionError]]:
        """Place what can be placed now; items never admissible return their error."""
        snap = self._snapshot(views)
        tail = self._tail(snap)
        placed: list[tuple[WorkItem, Assignment | AdmissionError]] = []
        held_back: str | None = None
        for item in self._ordered(snap, tail):
            if item.cold and held_back:
                self._waited(item.request_id, held_back)
                continue
            try:
                assignment = self._place(item, snap, tail)
            except AdmissionError as exc:
                self.withdraw(item.request_id)
                placed.append((item, exc))
                continue
            if isinstance(assignment, Assignment):
                placed.append((item, assignment))
                continue
            self._waited(item.request_id, assignment)
            # One a pacing D keeps waiting holds back no later one, which can take
            # only routes it cannot.
            if item.cold and assignment != PACED:
                held_back = held_back or assignment
        return placed

    def _homes(self, role: str) -> list[str]:
        return [e for e in self.ledger.capacity if self.roles[e] == role]

    def _mean(self, index: int) -> float:
        count = self._generations[3]
        return self._generations[index] / count if count else 0.0

    def _row(
        self, program: _Program, view: EngineView, now: float
    ) -> tuple[int, int] | None:
        """A program's (output tokens left, context), or None with none left."""
        rest = sum(t for _, _, t in program.rest)
        if program.over:
            left, context = rest, program.prompt + program.max_tokens
        else:
            emitted = 0
            if program.first_at is not None:
                step = view.costs.decode_step_s(
                    max(1, view.state.get("decode_requests") or 1),
                    view.state.get("decode_context_tokens") or program.prompt,
                )
                if step > 0:
                    emitted = min(
                        program.max_tokens - 1,
                        int(
                            (now - program.first_at) * view.costs.accept_per_row / step
                        ),
                    )
            left = program.max_tokens - emitted + rest
            context = program.prompt + emitted
        return (left, context) if left > 0 else None

    def _link_s(self, engines: frozenset[str]) -> float:
        """Seconds until the KV placed stages move to or from `engines` has moved.

        Transfers run one after another, since priced alone, transfers placed together
        would each look far shorter than they take.
        """
        free = -math.inf
        for since, tokens, link in sorted(
            (
                move.since if move.read_at is None else move.read_at,
                move.tokens,
                move.engines,
            )
            for _, moves in self._transfers.values()
            for move in moves
            if move.since is not None and not move.landed and move.engines & engines
        ):
            free = max(free, since) + self._kv_s(tokens, link)
        return max(0.0, free - self._clock())

    def _kv_s(self, tokens: int, link: frozenset[str] | None = None) -> float:
        """Seconds to move `tokens` of KV on `link`, or the slowest link."""
        rates = [
            moved / seconds
            for k, (moved, seconds) in self._link_moved.items()
            if link in (None, k) and seconds > 0
        ]
        return tokens / min(rates) if rates else self.kv_transfer_s(tokens)

    def _landed(self, move: _Transfer) -> None:
        """`move` has landed: it no longer holds its link.

        Unreported (`_read`), it is timed from the first push that showed it loading.
        """
        if move.landed or move.since is None:
            return
        move.landed = True
        if move.read_at is not None and move.reader not in self._reporting:
            self._timed(move, move.read_at, self._clock(), move.tokens)

    def _read(
        self, engine: str, request_id: str, start: float, end: float, tokens: int
    ) -> None:
        """`engine` read `tokens` of KV for `request_id` from `start` to `end`.

        On its monotonic clock, shared on one host. The move lands when this push
        arrives, which may follow the stage's first token or P's answer.
        """
        moves = self._untimed.get(request_id) or []
        for move in moves:
            if move.reader == engine:
                moves.remove(move)
                move.landed = True
                self._timed(move, start, end, tokens or move.tokens)
                return

    def _timed(self, move: _Transfer, start: float, end: float, tokens: int) -> None:
        """`move`'s read ran from `start` to `end`: fold it into its link's rate.

        The link carried `tokens` since that start or its last timed read's end,
        whichever is later, so reads that end together share its busy time.
        """
        if move.measured is not None:
            if move.measured[move.index] is not None:
                return
            move.measured[move.index] = end - start
        link = move.engines
        landed = self._link_landed.get(link, -math.inf)
        seconds = max(0.0, end - max(start, landed))
        self._link_landed[link] = max(landed, end)
        last = self._link_moved.get(link)
        self._link_moved[link] = (
            (tokens, seconds)
            if last is None
            else (
                last[0] + LINK_RATE_WEIGHT * (tokens - last[0]),
                last[1] + LINK_RATE_WEIGHT * (seconds - last[1]),
            )
        )

    def _reads_on_device(self, engine: str) -> bool:
        """Whether KV is moving to or from another engine on `engine`'s device."""
        peers = {e for pair in self.shared_devices if engine in pair for e in pair}
        peers.discard(engine)
        return any(
            move.since is not None and not move.landed and move.engines & peers
            for _, moves in self._transfers.values()
            for move in moves
        )

    def _price(self, view: EngineView, decoding: bool, held: int, n: int) -> float:
        if n <= 0:
            return 0.0
        if decoding:
            return view.costs.local_prefill_s(held, n)
        return view.costs.prefill_s(held, n)

    def _snapshot(self, views: Mapping[str, EngineView]) -> _Snapshot:
        now = self._clock()
        rows: dict[str, list[tuple[int, int]]] = {e: [] for e in views}
        mine = {}
        for program_id, program in self._programs.items():
            row = self._row(program, views[program.home], now)
            if row is not None:
                rows[program.home].append(row)
                mine[program_id] = row
        decoding = {e: self._decoding(rows[e], v) for e, v in views.items()}
        listed = {
            e: {submitted_id(r[0]): r[1] for r in v.state.get("prefill_rows") or ()}
            for e, v in views.items()
        }
        holding = {
            e: {submitted_id(r) for r in v.state.get("request_ids") or ()}
            for e, v in views.items()
        }
        loading = set().union(
            *(map(submitted_id, v.state.get("loading") or ()) for v in views.values())
        )
        self._track_reads(loading, now)
        loads = dict.fromkeys(views, 0.0)
        backlog = dict.fromkeys(views, 0.0)
        warm = dict.fromkeys(views, 0.0)
        unstarted = dict.fromkeys(views, 0.0)
        generations = dict.fromkeys(views, 0)
        busy = {e: bool(rows[e]) for e in views}
        arrivals: dict[str, list[float]] = {e: [] for e in views}
        for rid, prefill in self._prefills.items():
            self._advance(
                rid,
                prefill,
                listed[prefill.engine].get(rid),
                rid in holding[prefill.engine],
                rid in loading,
            )
            engine = prefill.engine
            cost = self._price(
                views[engine],
                decoding[engine],
                prefill.done,
                prefill.prompt - prefill.done,
            )
            loads[engine] += cost
            backlog[engine] += cost
            if prefill.cold:
                generations[engine] += 1
            else:
                warm[engine] += cost
            busy[engine] = True
            if not prefill.started:
                unstarted[engine] += cost
        for rid, prefill in self._prefills.items():
            if prefill.home != prefill.engine:
                arrivals[prefill.home].append(backlog[prefill.engine])
        for e, view in views.items():
            loads[e] += drain_s(view.costs, rows[e])
        credit = {
            e: 0.0
            if busy[e] or views[e].state.get("running")
            else min(arrivals[e], default=math.inf)
            for e in views
        }
        keep_busy = self._keep_busy(views)
        # U: each queued item's cheapest prefill and each announced one's at the mean;
        # unplaced generation stages' decode loads the homes at their full batch.
        self._price_queue(views)
        announced = len(self._announced)
        mean_prompt, mean_tokens, mean_peak = (self._mean(i) for i in range(3))
        pool = sum(self._pool.values())
        if announced and mean_prompt:
            pool += announced * min(
                v.costs.prefill_s(0, int(mean_prompt)) for v in views.values()
            )
        cold = [item for _, _, item in self._queue.values() if item.cold]
        tokens = sum(item.decode_tokens for item in cold) + announced * mean_tokens
        homes = self._homes(ROLE_D)
        steady = self._steady(homes, mean_prompt, mean_tokens, mean_peak)
        total = sum(batch for batch, _ in steady.values())
        for e, (batch, context) in steady.items():
            loads[e] += steady_decode_s(
                views[e].costs, int(tokens * batch / total), batch, context
            )
        rates = {
            e: prefill_rate(
                self._delta(view), decoding[e] or (e in steady and tokens > 0)
            )
            for e, view in views.items()
        }
        return _Snapshot(
            views=views,
            rows=rows,
            mine=mine,
            loads=loads,
            backlog=backlog,
            warm=warm,
            unstarted=unstarted,
            generations=generations,
            credit=credit,
            keep_busy=keep_busy,
            steady=steady,
            rates=rates,
            pool_s=pool,
            cold_waiting=len(cold) + announced,
            unfinished=len(self._programs) + len(cold) + announced,
            d_full=self._d_full(homes, announced, mean_peak),
        )

    def _track_reads(self, loading: set[str], now: float) -> None:
        """Follow each move: on the wire while its reader loads it, then landed."""
        for rid, (_, moves) in self._transfers.items():
            for move in moves:
                if move.since is None or move.landed:
                    continue
                if rid in loading:
                    if move.read_at is None:
                        move.read_at = now
                elif move.read_at is not None:
                    self._landed(move)

    def _advance(
        self,
        rid: str,
        prefill: _Prefill,
        computed: int | None,
        holding: bool,
        loading: bool,
    ) -> None:
        """Update a placed prefill from what its engine pushed.

        `computed` is None when the engine does not list it; `loading` is any engine's.
        """
        if computed is not None:
            prefill.done = max(prefill.done, computed)
            prefill.started = prefill.started or computed > prefill.held
        elif holding:
            prefill.started = True
        prefill.prefilled = computed is None and prefill.started and not loading
        if prefill.started:
            # Computing past what it pulled: the pull has landed.
            for move in self._transfers.get(rid, ("", []))[1]:
                if not move.after and move.read_at is None:
                    self._landed(move)

    def _keep_busy(self, views: Mapping[str, EngineView]) -> dict[str, float]:
        """Prefill seconds that keep each engine busy across a long push gap."""
        interval = HEADROOM_REFRESH_S
        if self._intervals:
            ordered = sorted(self._intervals)
            interval = ordered[int(0.95 * (len(ordered) - 1))]
        return {
            e: max(interval, v.costs.prefill_s(0, v.costs.step.chunk_tokens))
            for e, v in views.items()
        }

    def _price_queue(self, views: Mapping[str, EngineView]) -> None:
        for rid, (_, _, item) in self._queue.items():
            if rid not in self._pool:
                hits = {e: min(item.hits.get(e, 0), item.prompt_tokens) for e in views}
                self._pool[rid] = min(
                    v.costs.prefill_s(hits[e], item.prompt_tokens - hits[e])
                    for e, v in views.items()
                )

    def _steady(
        self,
        homes: Sequence[str],
        mean_prompt: float,
        mean_tokens: float,
        mean_peak: float,
    ) -> dict[str, tuple[int, float]]:
        """Each home's batch once full of mean programs, and their mean context."""
        steady: dict[str, tuple[int, float]] = {}
        for e in homes:
            batch = (self.ledger.capacity[e] - self.ledger.watermark[e]) // max(
                1, math.ceil(mean_peak)
            )
            batch = min(batch, self.ledger.slots.get(e, batch))
            steady[e] = (max(1, batch), mean_prompt + mean_tokens / 2)
        return steady

    def _d_full(self, homes: Sequence[str], announced: int, mean_peak: float) -> bool:
        """Whether no D has room for the next generation stage, queued or announced."""
        head = min(
            ((seq, item) for seq, _, item in self._queue.values() if item.cold),
            key=lambda entry: entry[0],
            default=None,
        )
        need = head[1].peak_blocks if head else math.ceil(mean_peak) if announced else 0
        return need > 0 and all(self.ledger.room(e) < need for e in homes)

    @staticmethod
    def _decoding(rows: Sequence[tuple[int, int]], view: EngineView) -> bool:
        return bool(rows) or bool(view.state.get("decode_requests"))

    def _delta(self, view: EngineView) -> float | None:
        """The engine's TPOT budget: as it pushed it, else the run's."""
        budget = view.state.get("tpot_budget")
        return self.tpot_delta if budget is None else budget["delta"]

    def _stages(self, item: WorkItem) -> list[PlannedStage]:
        held = item.hits.get(item.home, 0) if item.home is not None else 0
        held = min(held, item.prompt_tokens)
        return [(held, item.prompt_tokens - held, item.max_tokens), *item.rest]

    def _critical_path(
        self, snap: _Snapshot, home: str, stages: Sequence[PlannedStage]
    ) -> float:
        """The least time the stages take in turn, each decoded alone.

        At the home's batch, rows alike would each read as long as its whole drain.
        """
        return critical_path_s(
            stages,
            {e: v.costs for e, v in snap.views.items()},
            home,
            (0, 0),
            self._kv_s,
        )

    def _item_path(self, snap: _Snapshot, item: WorkItem) -> float:
        path = self._paths.get(item.request_id)
        if path is None:
            home = item.home or min(
                self._homes(ROLE_D), key=lambda e: (-self.ledger.room(e), e)
            )
            path = self._critical_path(snap, home, self._stages(item))
            if item.request_id in self._queue:
                self._paths[item.request_id] = path
        return path

    def _tail(self, snap: _Snapshot) -> bool:
        """Whether the queue is in its tail, where it goes by critical path.

        Few programs are left, or one's remaining stages outlast Phi. Not the mean
        load, which averages a D's decode over Ps that cannot run it before the drain.
        """
        if snap.unfinished <= self.tail_programs * len(snap.views):
            return True
        phi = makespan_s(snap.loads, snap.pool_s, rates=snap.rates)
        for program_id, row in snap.mine.items():
            program = self._programs[program_id]
            current = row[0] - sum(t for _, _, t in program.rest)
            stages = [(row[1], 0, current), *program.rest]
            if self._critical_path(snap, program.home, stages) > phi:
                return True
        return any(
            self._item_path(snap, item) > phi for _, _, item in self._queue.values()
        )

    def _ordered(self, snap: _Snapshot, tail: bool) -> list[WorkItem]:
        """Later stages by program age, then generation stages in arrival order."""
        entries = list(self._queue.values())
        if tail:
            entries.sort(key=lambda entry: -self._item_path(snap, entry[2]))
            return [item for _, _, item in entries]

        def age(entry: tuple[int, float, WorkItem]) -> int:
            program = self._programs.get(entry[2].program_id)
            return entry[0] if program is None else program.seq

        warm = sorted((e for e in entries if not e[2].cold), key=age)
        cold = sorted((e for e in entries if e[2].cold), key=lambda e: e[0])
        return [item for _, _, item in (*warm, *cold)]

    @staticmethod
    def _held(item: WorkItem, engine: str, route: str) -> tuple[int, int]:
        """The prompt tokens `engine` starts from on `route`, and how many it pulls."""
        prompt = item.prompt_tokens
        hit = min(item.hits.get(engine, 0), prompt)
        pulled = 0
        if route != LOCAL and item.offered_tokens > hit:
            pulled = min(item.offered_tokens, prompt) - hit
        return hit + pulled, pulled

    def _option(
        self,
        item: WorkItem,
        snap: _Snapshot,
        engine: str,
        route: str,
        home: str,
    ) -> _Option:
        """Price one route: its prefill on `engine`, then its decode on `home`."""
        view = snap.views[engine]
        prompt = item.prompt_tokens
        ahead = (snap.backlog if item.cold else snap.warm)[engine]
        closed = None
        held, pulled = self._held(item, engine, route)
        read = prompt - min(item.hits.get(home, 0), prompt) if route == REMOTE_PD else 0
        moves = []
        if pulled:
            assert item.home is not None
            moves.append(_Transfer(frozenset((engine, item.home)), pulled, engine))
        if read:
            moves.append(_Transfer(frozenset((engine, home)), read, home, after=True))
        # Seconds before the link is free for them.
        link = self._link_s(frozenset().union(*(m.engines for m in moves)))
        if pulled and self._hold_lapses_first(item, max(ahead, link)):
            closed = HOLD_UNREADABLE
        decoding = self._decoding(snap.rows[engine], view)
        cost = self._price(view, decoding, held, prompt - held)
        delay = budget_delay_s(
            view.state.get("tpot_budget"),
            cost,
            decoding,
            owed_only=self.roles[engine] == ROLE_D and not item.cold,
            append=not item.cold,
        )
        if math.isinf(delay):
            closed = closed or TPOT_GUARD
        # Past the idle window, priced at the engine's rate as U is; at a budget of 0
        # a decoding engine paces it instead (`_place`).
        owed = max(0.0, cost - snap.credit[engine])
        rate = snap.rates[engine]
        delta = {engine: owed / rate if rate > 0 else owed}
        finish = ahead + (0.0 if closed else delay)
        if pulled:
            finish = max(finish, link) + self._kv_s(pulled, moves[0].engines)
        finish += cost
        if read:
            finish = max(finish, link) + self._kv_s(read, moves[-1].engines)
        row = (item.decode_tokens, prompt)
        rates: dict[str, float] = {}
        if item.cold:
            if not self.ledger.fits(home, item.peak_blocks):
                closed = closed or NO_KV_ROOM
            self._price_generation_home(item, snap, home, row, delta, rates)
        if route == REMOTE_HOME:
            why = self._price_move(item, snap, engine, row, delta, rates)
            closed = closed or why
        tpot = None
        if self.roles[home] == ROLE_P and home != item.home:
            tpot = self._p_home_tpot(item, snap, home, held)
            if tpot is not None and tpot["over"]:
                closed = closed or TPOT_GUARD
            if self._reads_on_device(home):
                # Its decode would slow the reads into a D on its device.
                closed = closed or READS_PENDING
        return _Option(
            Placement(
                engine,
                home,
                route,
                cost,
                finish,
                delta,
                on_d=self.roles[engine] == ROLE_D,
                closed=closed,
                rates=rates,
            ),
            held,
            pulled > 0,
            tuple(moves),
            tpot,
        )

    def _hold_lapses_first(self, item: WorkItem, wait_s: float) -> bool:
        """Whether the hold a stage pulls lapses before a read `wait_s` away can end."""
        remaining = item.hold_remaining_s
        reservation = self.ledger.reservations.get(item.program_id)
        if reservation is not None and reservation.hold_until_s is not None:
            lease = reservation.hold_until_s - self._clock()
            remaining = lease if remaining is None else min(remaining, lease)
        return remaining is not None and remaining - self.read_margin_s <= wait_s

    def _price_generation_home(
        self,
        item: WorkItem,
        snap: _Snapshot,
        home: str,
        row: tuple[int, int],
        delta: dict[str, float],
        rates: dict[str, float],
    ) -> None:
        """Add a generation stage's decode on `home` to its route's `delta`, `rates`."""
        if item.request_id in self._queue:
            # Its decode leaves the steady load it was put on: its home's,
            # or for a P home every D's share of it.
            total = sum(batch for batch, _ in snap.steady.values())
            shares = (
                {home: 1.0}
                if home in snap.steady
                else {e: b / total for e, (b, _) in snap.steady.items()}
            )
            for e, share in shares.items():
                batch, context = snap.steady[e]
                delta[e] = delta.get(e, 0.0) - steady_decode_s(
                    snap.views[e].costs,
                    int(item.decode_tokens * share),
                    batch,
                    context,
                )
        delta[home] = delta.get(home, 0.0) + marginal_drain_s(
            snap.views[home].costs, snap.rows[home], [row]
        )
        if self.roles[home] == ROLE_P:
            rates[home] = prefill_rate(self._delta(snap.views[home]), True)

    def _price_move(
        self,
        item: WorkItem,
        snap: _Snapshot,
        engine: str,
        row: tuple[int, int],
        delta: dict[str, float],
        rates: dict[str, float],
    ) -> str | None:
        """Add a move to `engine` to `delta` and `rates`; return why it is closed."""
        view = snap.views[engine]
        old = item.home
        assert old is not None
        closed = None
        reservation = self.ledger.reservations.get(item.program_id)
        room = reservation is not None and self.ledger.fits(
            engine, reservation.peak_blocks
        )
        if not room:
            closed = NO_KV_ROOM
        own = snap.rows[old]
        current = snap.mine.get(item.program_id)
        if current in own:
            without = list(own)
            without.remove(current)
            delta[old] = -(
                drain_s(snap.views[old].costs, own)
                - drain_s(snap.views[old].costs, without)
            )
        delta[engine] += marginal_drain_s(view.costs, snap.rows[engine], [row])
        rates[engine] = prefill_rate(self._delta(view), True)
        if item.rest and self.roles[engine] == ROLE_P:
            decode = self._decode_s(item, snap, engine)
            if decode >= self.lease_s - self.read_margin_s:
                closed = closed or HOLD_LAPSES
        return closed

    def _p_home_tpot(
        self, item: WorkItem, snap: _Snapshot, engine: str, held: int
    ) -> dict[str, Any] | None:
        """The windowed TPOT a stage homed on the P `engine` would see, and its bound.

        The P's unfinished prefills, and the stage's own from `held` while the P
        decodes other rows, run in its decode steps; None without a finite budget.
        """
        view = snap.views[engine]
        delta = self._delta(view)
        if delta is None or math.isinf(delta):
            return None
        budget = view.state.get("tpot_budget") or {}
        costs = view.costs
        rows = snap.rows[engine]
        prefills = [
            (p.done, p.prompt)
            for rid, p in self._prefills.items()
            if p.engine == engine and rid != item.request_id
        ]
        if rows:
            prefills.append((held, item.prompt_tokens))
        backlog = steps = 0.0
        for done, prompt in prefills:
            if prompt > done:
                backlog += costs.local_prefill_s(done, prompt - done)
                steps += math.ceil((prompt - done) / costs.step.chunk_tokens)
        chunk = backlog / steps if steps else 0.0
        if budget.get("step_cap_s") is not None:
            chunk = min(chunk, budget["step_cap_s"])
        pure = costs.decode_step_s(
            len(rows) + 1, sum(c for _, c in rows) + item.prompt_tokens
        )
        window = budget.get("window_s") or TPOT_WINDOW_S
        ratio = mixed_tpot_ratio(pure, chunk, backlog, window)
        token = pure / costs.accept_per_row
        return {
            "tpot_s": ratio * token,
            "bound_s": (1 + delta) * token,
            "ratio": ratio,
            "backlog_s": backlog,
            "over": ratio > 1 + delta,
        }

    def _options(self, item: WorkItem, snap: _Snapshot) -> list[_Option]:
        prefills = [e for e in snap.views if self.roles[e] == ROLE_P]
        if item.cold:
            return self._generation_options(item, snap, prefills)
        return self._later_options(item, snap, prefills)

    def _generation_options(
        self, item: WorkItem, snap: _Snapshot, prefills: Sequence[str]
    ) -> list[_Option]:
        """A generation stage's routes, which start from no engine's history."""
        homes = self._homes(ROLE_D)
        if not any(
            self.ledger.capacity[e] - self.ledger.watermark[e] >= item.peak_blocks
            for e in homes
        ):
            raise AdmissionError(
                f"{item.program_id} needs {item.peak_blocks} blocks at its "
                f"peak, more than any of {homes} has past its watermark"
            )
        # The P routes' home: the D with the most room of those with room and a
        # slot; by room alone, a D with blocks and no slot would close them all.
        star = min(
            homes,
            key=lambda e: (
                not self.ledger.fits(e, item.peak_blocks),
                -self.ledger.room(e),
                e,
            ),
        )
        options = [self._option(item, snap, p, REMOTE_PD, star) for p in prefills]
        if self.role_flex:
            options += [self._option(item, snap, d, LOCAL, d) for d in homes]
        # No D has room: the stage may be homed on a P that decodes less.
        if self.drain_on_p and not any(
            self.ledger.fits(e, item.peak_blocks) for e in homes
        ):
            top = max(snap.loads[e] for e in homes)
            options += [
                self._option(item, snap, p, LOCAL, p)
                for p in prefills
                if p in self.ledger and snap.loads[p] < top
            ]
        return options

    def _later_options(
        self, item: WorkItem, snap: _Snapshot, prefills: Sequence[str]
    ) -> list[_Option]:
        """A later stage's routes, the first its home's own."""
        home = item.home
        assert home is not None
        local = self._option(item, snap, home, LOCAL, home)
        on_p = self.roles[home] != ROLE_D
        if on_p and snap.cold_waiting:
            return [local]
        options = [local]
        if on_p:
            # Back to a D, which pulls the matched prefix from the P's hold.
            options += [
                self._option(item, snap, d, REMOTE_HOME, d) for d in self._homes(ROLE_D)
            ]
        else:
            options += [self._option(item, snap, p, REMOTE_PD, home) for p in prefills]
        # To a P that decodes less than its home while the Ds are full, or, once no
        # generation stage is left, to a P with none left to prefill.
        if self.drain_on_p and not on_p:
            options += [
                self._option(item, snap, p, REMOTE_HOME, p)
                for p in prefills
                if p in self.ledger
                and (
                    (snap.d_full and snap.loads[p] < snap.loads[home])
                    or (snap.cold_waiting == 0 and not snap.generations[p])
                )
            ]
        if local.placement.closed is not None:
            return options
        # Nor does it leave for a route that is later than its home by `_ends_s`.
        home_s = self._ends_s(item, snap, local.placement)
        return [
            local,
            *(
                option
                if option.placement.closed
                or self._ends_s(item, snap, option.placement) <= home_s
                else replace(
                    option,
                    placement=replace(option.placement, closed=SLOWER_THAN_HOME),
                )
                for option in options[1:]
            ),
        ]

    def _late_bound(self, item: WorkItem, snap: _Snapshot, option: _Option) -> _Option:
        """A generation stage's route, closed if its engine takes none now.

        `GATE` while the engine's unstarted prefill reaches `keep_busy_s`, `PACED`
        while its TPOT budget adds as much to the stages it took (`_paced_s`).
        """
        if option.placement.closed:
            return option
        engine = option.placement.engine
        if snap.unstarted[engine] >= snap.keep_busy[engine]:
            why = GATE
        elif self._paced_s(item, snap, engine) >= snap.keep_busy[engine]:
            why = PACED
        else:
            return option
        return replace(option, placement=replace(option.placement, closed=why))

    def _paced_s(self, item: WorkItem, snap: _Snapshot, engine: str) -> float:
        """What a D's TPOT budget adds to the prefill of the generation stages it took.

        Their seconds left times 1 / rate - 1 = 1 / delta: while the D decodes other
        rows its budget runs their chunks at that rate, and `item` would start behind
        them. 0 on a P, without a finite budget, while the D decodes only those, and
        for a stage with clips, which a media lane prepares outside the D's order.
        """
        if item.media or self.roles[engine] != ROLE_D:
            return 0.0
        ahead = {
            p.program: p
            for p in self._prefills.values()
            if p.engine == engine and p.cold and not p.prefilled
        }
        if not ahead:
            return 0.0
        view = snap.views[engine]
        others = bool(view.state.get("decode_requests")) or any(
            program.home == engine and program_id in snap.mine
            for program_id, program in self._programs.items()
            if program_id not in ahead
        )
        rate = prefill_rate(self._delta(view), others)
        left = math.fsum(
            view.costs.local_prefill_s(p.done, p.prompt - p.done)
            for p in ahead.values()
        )
        if rate >= 1 or left <= 0:
            return 0.0
        return left / rate - left if rate > 0 else math.inf

    def _decode_s(self, item: WorkItem, snap: _Snapshot, home: str) -> float:
        """Seconds to decode the stage's output on `home`, beside its other rows."""
        rows = list(snap.rows[home])
        current = snap.mine.get(item.program_id)
        if home == item.home and current in rows:
            rows.remove(current)
        costs = snap.views[home].costs
        step = costs.decode_step_s(
            len(rows) + 1, sum(c for _, c in rows) + item.prompt_tokens
        )
        return item.max_tokens / costs.accept_per_row * step

    def _ends_s(self, item: WorkItem, snap: _Snapshot, placement: Placement) -> float:
        """What routes are compared by: when the decode would start.

        For a later stage once no generation stage waits, when the stage would end.
        """
        if item.cold or snap.cold_waiting:
            return placement.finish_s
        return placement.finish_s + self._decode_s(item, snap, placement.home)

    def _choose(
        self, item: WorkItem, snap: _Snapshot, tail: bool
    ) -> tuple[Placement | None, list[_Option], float]:
        """The rule's pick, the item's routes, and the pool without the item."""
        options = self._options(item, snap)
        if item.cold:
            options = [self._late_bound(item, snap, o) for o in options]
        pool = snap.pool_s - self._pool.get(item.request_id, 0.0)
        placements = [option.placement for option in options]
        if tail:
            best = min(
                (p for p in placements if p.closed is None),
                key=lambda p: (self._ends_s(item, snap, p), p.on_d, p.engine),
                default=None,
            )
        else:
            best = decide_minmax(snap.loads, pool, placements, rates=snap.rates)
        return best, options, pool

    def _place(self, item: WorkItem, snap: _Snapshot, tail: bool) -> Assignment | str:
        """Where the item goes, or why it waits (`ADMISSION`, `GATE` or `PACED`)."""
        best, options, pool = self._choose(item, snap, tail)
        local = options[0]
        if (
            best is None
            and item.kind == LATER_STAGE
            and local.placement.closed == TPOT_GUARD
        ):
            # Closed everywhere, at home by the TPOT budget alone: the home paces
            # it, since its program's own decode there could hold it forever.
            local = replace(local, placement=replace(local.placement, closed=None))
            options[0], best = local, local.placement
        placements = [option.placement for option in options]
        if best is None:
            closed = {p.closed for p in placements}
            if item.cold and NO_KV_ROOM in closed and not closed & {GATE, PACED}:
                return ADMISSION
            return PACED if PACED in closed else GATE
        if (
            item.cold
            and not tail
            and self.roles[best.home] == ROLE_P
            and makespan_s(snap.loads, pool, best.delta, {**snap.rates, **best.rates})
            >= makespan_s(snap.loads, snap.pool_s, rates=snap.rates)
        ):
            # Homed on a P only if that shortens the pass; else it waits for room.
            return ADMISSION
        if item.cold and not tail and best.on_d:
            held = [
                replace(p, closed=None)
                for p in placements
                if p.closed == GATE and not p.on_d
            ]
            if held:
                rule = decide_minmax(
                    snap.loads,
                    pool,
                    [*held, *(p for p in placements if p.closed is None)],
                    rates=snap.rates,
                )
                if rule is not None and not rule.on_d:
                    # Wait for the P rather than take a D's KV and decode budget.
                    return GATE
        option = next(o for o in options if o.placement is best)
        return self._commit(item, option, snap, tail, pool, options)

    def _p_decode(
        self, item: WorkItem, option: _Option, snap: _Snapshot, local: Placement
    ) -> dict[str, Any] | None:
        """The record of a P made a home, or left for a D; None otherwise."""
        best = option.placement
        if best.home == item.home:
            return None
        if self.roles[best.home] == ROLE_P:
            return {
                "kind": "p_home_generations" if item.cold else "p_home_moves",
                "from": item.home,
                "to": best.home,
                **(option.tpot or {}),
            }
        if item.home is not None and self.roles[item.home] == ROLE_P:
            return {
                "kind": "p_to_d_moves",
                "from": item.home,
                "to": best.home,
                "ends_s": self._ends_s(item, snap, best),
                "home_ends_s": self._ends_s(item, snap, local),
            }
        return None

    def _commit(
        self,
        item: WorkItem,
        option: _Option,
        snap: _Snapshot,
        tail: bool,
        pool: float,
        options: Sequence[_Option],
    ) -> Assignment | str:
        best = option.placement
        placements = [o.placement for o in options]
        program_id = item.program_id
        program = self._programs.get(program_id)
        entry = self._queue.get(item.request_id)
        queued_at = self._clock() if entry is None else entry[1]
        if item.cold:
            if self.ledger.admit(program_id, item.peak_blocks, [best.home]) is None:
                return ADMISSION
            program = self._programs[program_id] = _Program(
                next(self._seq) if entry is None else entry[0], best.home
            )
        elif program is None:
            assert item.home is not None
            program = self._programs[program_id] = _Program(next(self._seq), item.home)
        if best.route == REMOTE_HOME:
            if not self.ledger.begin_move(program_id, best.engine):
                return GATE
            program.moving_from = program.home
        phase = self._phase(best, snap, tail)
        note = self._p_decode(item, option, snap, placements[0])
        before = dict(snap.loads)
        self._bind(item, program, option, snap)
        kv_move_s = self._start_moves(item, option)
        rates = self._charge(item, best, snap)
        admission_s, gate_s, _, _ = self._waited(item.request_id, None)
        self.withdraw(item.request_id)
        view = snap.views[best.engine]
        record = {
            "route": best.route,
            "engine": best.engine,
            "home": best.home,
            "pull": option.pull,
            "phase": phase,
            "cost_s": best.cost_s,
            "finish_s": best.finish_s,
            "makespan_s": {
                f"{p.route}:{p.engine}": None
                if p.closed
                else makespan_s(before, pool, p.delta, {**rates, **p.rates})
                for p in placements
            },
            "closed": {
                f"{p.route}:{p.engine}": p.closed for p in placements if p.closed
            },
            "loads_s": before,
            "pool_s": pool,
            "keep_busy_s": snap.keep_busy[best.engine],
            "tpot_budget": view.state.get("tpot_budget"),
            "home_tpot_budget": snap.views[best.home].state.get("tpot_budget"),
            "queued_s": self._clock() - queued_at,
            "admission_wait_s": admission_s,
            "gate_wait_s": gate_s,
            "prefill_rates": rates,
            "admission": self.ledger.record(best.home)
            if best.home in self.ledger
            else None,
            "d_room_blocks": {e: self.ledger.room(e) for e in self._homes(ROLE_D)},
        }
        tpot = {
            f"{o.placement.route}:{o.placement.engine}": o.tpot
            for o in options
            if o.tpot is not None
        }
        if tpot:
            record["p_home_tpot"] = tpot
        if kv_move_s is not None:
            record["kv_move_s"] = kv_move_s
        if note is not None:
            record["p_decode"] = note
        assignment = Assignment(best.route, best.engine, best.home, record)
        return assignment

    @staticmethod
    def _phase(best: Placement, snap: _Snapshot, tail: bool) -> str:
        return (
            TAIL
            if tail
            else DRAIN
            if best.route == REMOTE_HOME or snap.cold_waiting == 0
            else FILL
            if any(credit > 0 for credit in snap.credit.values())
            else STEADY
        )

    def _bind(
        self, item: WorkItem, program: _Program, option: _Option, snap: _Snapshot
    ) -> None:
        """Make `item` the program's running stage, on the engines of its route."""
        best = option.placement
        program_id = item.program_id
        row = (item.decode_tokens, item.prompt_tokens)
        current = snap.mine.get(program_id)
        if current in snap.rows[program.home]:
            snap.rows[program.home].remove(current)
        snap.rows[best.home].append(row)
        snap.mine[program_id] = row
        if program.request_id is not None:
            self._requests.pop(program.request_id, None)
        program.home = best.home
        program.request_id = item.request_id
        program.prompt = item.prompt_tokens
        program.max_tokens = item.max_tokens
        program.rest = item.rest
        program.first_at = None
        program.over = False
        program.moved = best.route == REMOTE_HOME
        self._requests[item.request_id] = program_id
        self._prefills[item.request_id] = _Prefill(
            program_id,
            best.engine,
            best.home,
            option.held,
            option.held,
            item.prompt_tokens,
            item.cold,
        )

    def _start_moves(self, item: WorkItem, option: _Option) -> dict[str, Any] | None:
        """Start the route's KV moves and return their record, None without moves."""
        if not option.moves:
            return None
        measured: list[float | None] = [None] * len(option.moves)
        kv_move_s = {
            "link_s": self._link_s(
                frozenset().union(*(m.engines for m in option.moves))
            ),
            "moves_s": [self._kv_s(m.tokens, m.engines) for m in option.moves],
            "measured_s": measured,
            "links": [
                f"{''.join(m.engines - {m.reader})}->{m.reader}" for m in option.moves
            ],
        }
        now = self._clock()
        moves = [
            replace(
                m,
                since=None if m.after else now,
                measured=measured,
                index=index,
            )
            for index, m in enumerate(option.moves)
        ]
        self._transfers[item.request_id] = (item.program_id, moves)
        self._untimed[item.request_id] = list(moves)
        return kv_move_s

    def _charge(
        self, item: WorkItem, best: Placement, snap: _Snapshot
    ) -> dict[str, float]:
        """Book a placement on the snapshot and return the prefill rates before it."""
        for engine, delta in best.delta.items():
            snap.loads[engine] += delta
        rates = dict(snap.rates)
        snap.rates.update(best.rates)
        snap.backlog[best.engine] += best.cost_s
        if not item.cold:
            snap.warm[best.engine] += best.cost_s
        snap.unstarted[best.engine] += best.cost_s
        snap.generations[best.engine] += int(item.cold)
        snap.credit[best.engine] = snap.credit[best.home] = 0.0
        snap.pool_s -= self._pool.get(item.request_id, 0.0)
        if item.cold:
            snap.cold_waiting -= 1
        return rates

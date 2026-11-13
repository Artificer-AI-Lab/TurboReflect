# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""Program-level KV admission: a ledger of reservations on decoding engines.

A program is admitted once its KV peak over every stage it has left is reserved on
its home, the engine that decodes its stages, before any engine prefills it. Later
stages then never ask for room: D holds each stage's history for the next, and a
home filled by first stages would leave later stages waiting on each other until
the holds lapse. The peak is an upper bound from the declared plan's `max_tokens`,
lowered as stages end.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class AdmissionError(Exception):
    """A program can never be admitted."""


@dataclass(frozen=True)
class StageShape:
    """How a later stage's prompt continues the stage before it, outputs aside.

    Attributes:
        template_tokens: Tokens past the generation prompt and carried outputs; < 0
            where it drops some of that prompt.
        released_tokens: Unmatched tokens of the previous prompt, held until its
            first token.
        matches_output: Whether the matched prefix runs through the previous
            output; if not, that output is appended again and held twice.
    """

    template_tokens: int
    released_tokens: int = 0
    matches_output: bool = True


@dataclass(frozen=True)
class StageBound:
    """What one stage holds at most while it runs, in tokens.

    `held_tokens` are the previous stage's tokens it does not match, still held.
    """

    prompt_tokens: int
    max_tokens: int
    held_tokens: int = 0


def stage_bounds(
    generation_tokens: int,
    max_tokens: Sequence[int],
    shapes: Sequence[StageShape],
) -> list[StageBound]:
    """Each planned stage's bound, generation stage first; `shapes` is one shorter.

    Stage k's prompt: at most the first prompt, its template and every earlier output.
    """
    bounds = [StageBound(generation_tokens, max_tokens[0])]
    for k, shape in enumerate(shapes, start=1):
        held = shape.released_tokens
        if not shape.matches_output:
            held += max_tokens[k - 1]
        bounds.append(
            StageBound(
                generation_tokens + shape.template_tokens + sum(max_tokens[:k]),
                max_tokens[k],
                held,
            )
        )
    return bounds


def peak_blocks(
    bounds: Iterable[StageBound],
    block: int,
    lookahead: int,
    state: tuple[int, int] = (0, 0),
    held_output: int = 0,
) -> int:
    """The most blocks of `block` tokens any of `bounds` holds.

    The stage's own sequence runs `lookahead` (one token and the drafts) past its
    output bound. `held_output` is the output bound of the stage before the first
    of `bounds`. A hybrid model's `state` (`decode_state`) is the recurrent-state
    blocks every sequence holds and those a hold pins per block boundary its stage's
    output crossed, plus one, for the boundary states a pull of D's history reads.
    """
    per_sequence, per_boundary = state
    bounds = list(bounds)
    outputs = [held_output, *(b.max_tokens for b in bounds)]

    def blocks(tokens: int, boundaries: int = 0) -> int:
        tokens = max(0, tokens)
        if not tokens:
            return 0
        return -(-tokens // block) + per_sequence + per_boundary * boundaries

    return max(
        (
            blocks(b.prompt_tokens + b.max_tokens + lookahead)
            + blocks(b.held_tokens, -(-outputs[k] // block) + 1)
            for k, b in enumerate(bounds)
        ),
        default=0,
    )


@dataclass
class Reservation:
    """One admitted program's blocks.

    Attributes:
        target: The home it moves to, counted on both until its first token there.
        hold_until_s: When the home's hold of its history lapses, if it does.
        holds: By engine, blocks a P holds of it until its home reads them.
    """

    program_id: str
    home: str
    peak_blocks: int
    target: str | None = None
    hold_until_s: float | None = None
    holds: dict[str, int] = field(default_factory=dict)


class Ledger:
    """Reservations against each decoding engine's KV pool, in blocks.

    `room(e)` = capacity - watermark - reservations on `e` - drift. Drift is what `e`
    reports in use past its reservations and holds for another home's reads (a
    leaked hold, untracked work); it closes room only past the watermark. KV a home
    reads or prefills for a program counts once, inside that program's reservation.
    """

    def __init__(
        self,
        capacity_blocks: Mapping[str, int],
        watermark_blocks: Mapping[str, int],
        slots: Mapping[str, int] | None = None,
        block_tokens: Mapping[str, int] | None = None,
    ) -> None:
        self.capacity = dict(capacity_blocks)
        self.block_tokens = dict(block_tokens or {})
        self.watermark = {e: watermark_blocks.get(e, 0) for e in self.capacity}
        self.slots = dict(slots or {})
        self.reservations: dict[str, Reservation] = {}
        self.drift = dict.fromkeys(self.capacity, 0)

    def __contains__(self, engine: str) -> bool:
        return engine in self.capacity

    def reserved(self, engine: str) -> int:
        """Every block reserved on `engine`: homes and move targets."""
        return sum(
            r.peak_blocks
            for r in self.reservations.values()
            if engine in (r.home, r.target)
        )

    def room(self, engine: str) -> int:
        drift = self.drift[engine]
        excess = drift if drift > self.watermark[engine] else 0
        return (
            self.capacity[engine]
            - self.watermark[engine]
            - self.reserved(engine)
            - excess
        )

    def _slot_free(self, engine: str) -> bool:
        if engine not in self.slots:
            return True
        used = sum(engine in (r.home, r.target) for r in self.reservations.values())
        return used < self.slots[engine]

    def fits(self, engine: str, blocks: int) -> bool:
        """Whether `engine` has room and a seq slot for `blocks` more now."""
        return self.room(engine) >= blocks and self._slot_free(engine)

    def admit(
        self, program_id: str, peak_blocks: int, homes: Iterable[str]
    ) -> str | None:
        """Reserve the program's peak on the home with the most room.

        Returns:
            The home, or None if none has room for it now.
        """
        open_ = [e for e in homes if self.fits(e, peak_blocks)]
        if not open_:
            return None
        home = min(open_, key=lambda e: (-self.room(e), e))
        self.reservations[program_id] = Reservation(program_id, home, peak_blocks)
        return home

    def resize(self, program_id: str, peak_blocks: int) -> None:
        """Lower the reservation to the stages left; never raise it."""
        r = self.reservations[program_id]
        r.peak_blocks = min(r.peak_blocks, peak_blocks)

    def begin_move(self, program_id: str, target: str) -> bool:
        """Also reserve the peak on `target`, or return False and change nothing.

        Both stay counted until `finish_move`: until the stage's first token there,
        the home's hold and the target's copy both hold the history.
        """
        r = self.reservations[program_id]
        if target == r.home or not self.fits(target, r.peak_blocks):
            return False
        r.target = target
        return True

    def finish_move(self, program_id: str, hold_until_s: float | None = None) -> None:
        """The stage has its first token on the target: release the old home.

        `hold_until_s` is when the new home's hold lapses; None if it lasts the pass.
        """
        r = self.reservations[program_id]
        r.home, r.target, r.hold_until_s = r.target, None, hold_until_s

    def abort_move(self, program_id: str) -> None:
        """The move was not made: release its target."""
        r = self.reservations.get(program_id)
        if r is not None:
            r.target = None

    def hold(self, program_id: str, engine: str, tokens: int) -> None:
        """`engine` holds `tokens` of the program's KV until its home reads them."""
        r = self.reservations.get(program_id)
        block = self.block_tokens.get(engine)
        if r is not None and block and engine != r.home:
            r.holds[engine] = -(-tokens // block)

    def drop_holds(self, program_id: str) -> None:
        r = self.reservations.get(program_id)
        if r is not None:
            r.holds.clear()

    def release(self, program_id: str) -> Reservation | None:
        """Free everything the program reserved; a released one is ignored."""
        return self.reservations.pop(program_id, None)

    def _accounted(self, engine: str) -> int:
        return self.reserved(engine) + sum(
            r.holds.get(engine, 0) for r in self.reservations.values()
        )

    def reconcile(self, engine: str, free_blocks: int, inflight_blocks: int) -> int:
        """Take an engine's report and return its drift.

        Free blocks include cached ones; in-flight blocks are what KV loads will take.
        """
        used = self.capacity[engine] - free_blocks + inflight_blocks
        drift = max(0, used - self._accounted(engine))
        self.drift[engine] = drift
        return drift

    def record(self, engine: str) -> dict[str, Any]:
        """The figures a decision on `engine` was made on."""
        return {
            "capacity_blocks": self.capacity[engine],
            "watermark_blocks": self.watermark[engine],
            "reserved_blocks": self.reserved(engine),
            "drift_blocks": self.drift[engine],
            "held_blocks": sum(
                r.holds.get(engine, 0) for r in self.reservations.values()
            ),
            "room_blocks": self.room(engine),
            "programs": sum(
                engine in (r.home, r.target) for r in self.reservations.values()
            ),
        }

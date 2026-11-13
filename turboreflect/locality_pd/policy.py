# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""The P/D cost model, the relative decode-latency budget and the min-max rule.

Deployment calibration (`pd_profile`) fits a step model (`StepModel`) and a
piecewise affine KV transfer cost over finite token domains; nothing is
extrapolated. Each engine corrects the prior's prices by its measured steps
(`EngineCosts`). A decoding engine's prefill chunks may add at most `delta` times
its pure decode seconds over a window (`TpotBudget`). The min-max rule places a
work item where the projected makespan grows least (`decide_minmax`).
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final


class PolicyInputError(ValueError):
    """A missing or ill-typed calibration or engine setting."""


class PolicyDomainError(PolicyInputError):
    """The profile has no prediction for this input or engine pair."""


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyInputError(f"{name} must be a number")
    return float(value)


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PolicyInputError(f"{name} must be an integer")
    if positive and value <= 0:
        raise PolicyInputError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class PrefillTask:
    history_tokens: int
    incremental_tokens: int


@dataclass(frozen=True)
class Observation:
    at_s: float
    value_s: float


@dataclass(frozen=True)
class _AffineSegment:
    bounds: tuple[tuple[int, int], ...]
    intercept_s: float
    coefficients_s: tuple[float, ...]


def _affine_segment(item: Any, features: tuple[str, ...]) -> _AffineSegment:
    if not isinstance(item, Mapping) or set(item) != {
        "bounds",
        "intercept_s",
        "coefficients_s",
    }:
        raise PolicyInputError("unknown or missing affine segment fields")
    if any(
        not isinstance(item[key], Mapping) or set(item[key]) != set(features)
        for key in ("bounds", "coefficients_s")
    ):
        raise PolicyInputError(f"segment features must be {features}")
    bounds = []
    for feature in features:
        pair = item["bounds"][feature]
        if not isinstance(pair, list) or len(pair) != 2:
            raise PolicyInputError("profile bounds must be [inclusive, exclusive]")
        low, high = (_integer(v, feature) for v in pair)
        bounds.append((low, high))
    values = (
        item["intercept_s"],
        *(item["coefficients_s"][f] for f in features),
    )
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        raise PolicyInputError("profile coefficients must be numbers")
    intercept, *slopes = map(float, values)
    coefficients = tuple(slopes)
    return _AffineSegment(tuple(bounds), intercept, coefficients)


class _PiecewiseAffine:
    def __init__(self, segments: Any, features: tuple[str, ...]):
        if not isinstance(segments, list) or not segments:
            raise PolicyInputError("a profile model needs a nonempty segments list")
        self.features = features
        self.segments = [_affine_segment(item, features) for item in segments]

    def __call__(self, *features: int) -> float:
        for segment in self.segments:
            if all(
                lo <= value < hi for value, (lo, hi) in zip(features, segment.bounds)
            ):
                return math.fsum(
                    (
                        segment.intercept_s,
                        *(v * c for v, c in zip(features, segment.coefficients_s)),
                    )
                )
        raise PolicyDomainError(
            f"no calibrated domain for {dict(zip(self.features, features))}"
        )


# The engine key a profile states for every engine it does not name.
ANY_WORKER = "*"


class ProfileCostModel:
    """Version-1 calibration schema: provenance and the KV transfer cost.

    `transfer` maps source to destination keys (`ANY_WORKER` for any unstated) to
    segments of half-open integer 'bounds', 'intercept_s' and 'coefficients_s' over
    `tokens`; `provenance` carries the fitted step model.
    """

    provenance: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProfileCostModel:
        obj = cls()
        obj.provenance = dict(payload["provenance"])
        obj._transfer = {
            (source, destination): _PiecewiseAffine(value, ("tokens",))
            for source, destinations in payload["transfer"].items()
            for destination, value in destinations.items()
        }
        return obj

    @staticmethod
    def _model(table: Mapping[Any, _PiecewiseAffine], key: Any, wildcard: Any):
        found = table.get(key) or table.get(wildcard)
        if found is None:
            raise PolicyDomainError(f"missing profile: {key}")
        return found

    def transfer(self, tokens: int, source: str, destination: str) -> float:
        if tokens == 0:
            return 0.0
        model = self._model(
            self._transfer, (source, destination), (ANY_WORKER, ANY_WORKER)
        )
        return model(tokens)


TPOT_GUARD = "tpot_guard"

# The step-model terms a profile must state; `prefill_context_tokens` defaults to 0.
STEP_TERMS = (
    "const",
    "prefill_tokens",
    "prefill_pairs",
    "prefill_square",
    "decode_requests",
    "decode_kv_tokens",
    "decode_query_tokens",
    "verify_pairs",
)


def chunk_seconds(terms: Mapping[str, Any], context: Any, tokens: Any) -> Any:
    """One prefill chunk's step time: `tokens` new tokens after `context` earlier ones.

    Every chunk repeats the context's per-token cost (`prefill_context_tokens`, such
    as MLA's re-expansion of cached latents). Takes numbers or numpy arrays.
    """
    return (
        terms["const"]
        + terms["prefill_tokens"] * tokens
        + terms["prefill_pairs"] * tokens * (context + tokens / 2)
        + terms["prefill_square"] * tokens**2
        + terms.get("prefill_context_tokens", 0.0) * context
    )


@dataclass(frozen=True)
class StepModel:
    """The step model a fitted profile states, and the times it prices.

    `terms` are seconds per unit of each term, `chunk_tokens` the prefill chunk size.
    """

    terms: Mapping[str, float]
    chunk_tokens: int

    @classmethod
    def from_profile(cls, model: ProfileCostModel) -> StepModel:
        """The step model in a profile's provenance; raises if it states none."""
        terms = model.provenance.get("step_terms")
        chunk = model.provenance.get("chunk_tokens")
        if not isinstance(terms, Mapping) or any(t not in terms for t in STEP_TERMS):
            raise PolicyInputError(
                "the dispatcher prices S(n) with the fitted step model, and "
                "this profile states no provenance.step_terms"
            )
        for name, value in terms.items():
            _number(value, f"step term {name}")
        return cls(dict(terms), _integer(chunk, "chunk_tokens", positive=True))

    def prefill_s(self, history: int, tokens: int) -> float:
        """S(n): `tokens` new tokens after `history`, alone, chunk by chunk."""
        return math.fsum(
            chunk_seconds(
                self.terms,
                history + start,
                min(self.chunk_tokens, tokens - start),
            )
            for start in range(0, tokens, self.chunk_tokens)
        )

    def decode_step_s(
        self, requests: int, context_tokens: float, query_per_request: float
    ) -> float:
        """A pure decode step: `requests` decoding over `context_tokens`.

        Each of `query_per_request` query tokens past the first is a draft verified
        against the request's context, a verify pair per context token.
        """
        query = query_per_request * requests
        verify = (query_per_request - 1) * context_tokens
        return (
            self.terms["const"]
            + self.terms["decode_requests"] * requests
            + self.terms["decode_query_tokens"] * query
            + self.terms["decode_kv_tokens"] * context_tokens
            + self.terms["verify_pairs"] * verify
        )

    def mix_extra_s(self, tokens: int, context: int) -> float:
        """What a chunk of `tokens` after `context` adds to a decode step."""
        return chunk_seconds(self.terms, context, tokens) - self.terms["const"]

    def local_prefill_s(self, history: int, tokens: int, chunk: int = 0) -> float:
        """E(h, n): what a prefill adds to the decode steps it rides in.

        One `chunk`-token chunk a step (the token budget when 0), each adding its
        prefill terms to a step whose constant the decodes already pay.
        """
        size = chunk or self.chunk_tokens
        return math.fsum(
            self.mix_extra_s(min(size, tokens - start), history + start)
            for start in range(0, tokens, size)
        )

    def price_s(self, features: Mapping[str, float]) -> float:
        """A step's seconds from its `step_features`."""
        return math.fsum(v * features.get(k, 0.0) for k, v in self.terms.items())


# Step classes calibrated apart: no row prefills, no row decodes, or both.
STEP_DEC: Final = "dec"
STEP_PRE: Final = "pre"
STEP_MIX: Final = "mix"
# Steps a class has measured before its ratio stands in for 1.
CALIBRATION_MIN_STEPS = 10
TPOT_WINDOW_S = 10.0
# The default share of `delta * window_s` all the chunks of one step may add.
TPOT_STEP_SHARE = 0.5
# `TpotBudget.state`'s rule: the per-step cap, or the floor rule.
TPOT_STEP_RULE: Final = "step_cap"
TPOT_FLOOR_RULE: Final = "floor"
# How a TPOT budget weighs a step: by its decoding rows, so `delta` bounds the
# token-weighted mean slowdown (mean TPOT), or by its seconds, bounding every row's.
TPOT_TOKENS: Final = "tokens"
TPOT_TIME: Final = "time"
TPOT_WEIGHTINGS = (TPOT_TOKENS, TPOT_TIME)
# `TpotBudget.counters`' decision counts: `capped` grants gave less than asked,
# `liveness_grants` ran one block past the cap (or the floor rule's floor), and
# `refunds` were given back and leave the other counts.
TPOT_COUNTS = (
    "grants",
    "capped",
    "first_chunk_caps",
    "liveness_grants",
    "step_cap_hits",
    "debt_holds",
    "exempt_rows",
    "refunds",
    "blocked",
)
# Its seconds; `fixed` is the per-chunk cost of the grants' contexts.
TPOT_SECONDS = ("granted", "fixed", "split", "live", "past_balance", "exempt")


def step_features(rows: Iterable[tuple[int, int, bool]]) -> dict[str, float]:
    """A step's step-model terms from its (scheduled, context, prefills) rows."""
    x = dict.fromkeys((*STEP_TERMS, "prefill_context_tokens"), 0.0)
    x["const"] = 1.0
    for scheduled, context, prefills in rows:
        if prefills:
            x["prefill_tokens"] += scheduled
            x["prefill_pairs"] += scheduled * (context + scheduled / 2)
            x["prefill_context_tokens"] += context
            x["prefill_square"] += scheduled**2
        else:
            x["decode_requests"] += 1
            x["decode_kv_tokens"] += context
            x["decode_query_tokens"] += scheduled
            x["verify_pairs"] += (scheduled - 1) * context
    return x


def step_class(rows: Iterable[tuple[int, int, bool]]) -> str:
    """`STEP_DEC`, `STEP_PRE` or `STEP_MIX` for a non-empty step's rows."""
    kinds = {prefills for _, _, prefills in rows}
    if kinds == {False}:
        return STEP_DEC
    return STEP_PRE if kinds == {True} else STEP_MIX


def calibrated_ratio(means: Mapping[str, Any] | None, step: StepModel) -> float:
    """rho: a step class's measured seconds over the model's price of its terms.

    `means` is a pushed `cost_classes` entry: means of terms `x` and seconds `t`
    over `n` steps. 1 until `n` reaches `CALIBRATION_MIN_STEPS`.
    """
    if not means or means["n"] < CALIBRATION_MIN_STEPS:
        return 1.0
    priced = step.price_s(means["x"])
    return means["t"] / priced if priced > 0 else 1.0


@dataclass
class TpotGrant:
    """One booked grant: what it added to the counters, and what it was priced at.

    `kind` becomes `refund` once given back; `shrink` reprices it from its inputs.
    """

    context: int
    kind: str
    counts: dict[str, int]
    seconds: dict[str, float]
    tokens: dict[str, int]
    weight: float = 1.0
    ratio: float = 1.0
    first: bool = False
    left: float = 0.0


class TpotBudget:
    """An engine's TPOT window budget for the prefill chunks of decode steps.

    Over the steps that ended in the last `window_s` seconds (caller's clock), what
    chunks added to decode steps (`extra`) stays within `delta` times the decode
    work alone (`pure`), appends' extra aside: a deficit bucket.

    With `step_share`, a step's chunks together cost at most
    `step_share * delta * window_s` (the step's cap), each the largest that fits.
    An append (a later stage, on its program's critical path) starts whatever the
    balance; any other chunk while it is not negative, going at most a cap into
    debt; a cold chunk (a generation stage on D, which a P could prefill) spends
    only the balance and waits until it covers its floor (c_min, or the cap if
    less), never running in chunks too small to pay for their context. If not even
    one block fits the cap, the step's first chunk takes one block (liveness).
    Appends aside, a window's extra is then at most `delta * pure` plus one cap.
    Without a step share (the floor rule), each chunk is the most the balance
    covers but at least its floor, and a waiting prefill's first chunk runs whole.

    Weighted by `tokens`, seconds are booked times the step's decoding rows (rows
    computing a first token excluded), so `delta` bounds the token-weighted mean
    slowdown; by `time` each weighs 1.
    """

    def __init__(
        self,
        delta: float,
        window_s: float,
        step: StepModel,
        block: int,
        step_share: float | None = None,
        weighting: str = TPOT_TIME,
    ):
        if not (delta >= 0 and window_s > 0 and block > 0):
            raise PolicyInputError("a TPOT budget needs delta >= 0 and a window")
        if step_share is not None and not 0 < step_share <= 1:
            raise PolicyInputError("a TPOT budget's step share is in (0, 1]")
        if weighting not in TPOT_WEIGHTINGS:
            raise PolicyInputError(f"a TPOT budget weighs by {list(TPOT_WEIGHTINGS)}")
        self.weighting = weighting
        # What a second of this step's chunks owes: the rows it decodes.
        self.weight = 1
        self.delta = delta
        self.window_s = window_s
        self.step = step
        self.block = block
        self.step_share = step_share
        self.step_cap_s = (
            None
            if step_share is None
            else math.inf
            if math.isinf(delta)
            else step_share * delta * window_s
        )
        self._steps: collections.deque[tuple[float, float, float]] = collections.deque()
        self.pure_s = 0.0
        self.extra_s = 0.0
        # Granted this step, not yet measured.
        self.pending_s = 0.0
        # This step's priced chunks, exempt rows included.
        self.step_s = 0.0
        self._step_grants = 0
        self.last_grant: TpotGrant | None = None
        self._counts = dict.fromkeys(TPOT_COUNTS, 0)
        self._seconds = dict.fromkeys(TPOT_SECONDS, 0.0)
        self._tokens = {"granted": 0, "wanted": 0}

    def left_s(self, now: float) -> float:
        if math.isinf(self.delta):
            return math.inf
        while self._steps and self._steps[0][0] <= now - self.window_s:
            _, pure, extra = self._steps.popleft()
            self.pure_s -= pure
            self.extra_s -= extra
        if not self._steps:
            self.pure_s = self.extra_s = 0.0
        return self.delta * self.pure_s - self.extra_s - self.pending_s

    def price_s(self, context: int, tokens: int, ratio: float = 1.0) -> float:
        """What a chunk of `tokens` after `context` adds to a decode step."""
        return ratio * self.step.mix_extra_s(tokens, context)

    def _weight(self, rows: int) -> int:
        return max(1, rows) if self.weighting == TPOT_TOKENS else 1

    def start_step(self, now: float, decoding: int = 1) -> None:
        """Start a step of `decoding` rows with its whole cap."""
        self.weight = self._weight(decoding)
        self.step_s = 0.0
        self._step_grants = 0
        self.left_s(now)

    def _fit(self, context: int, want: int, room_s: float, ratio: float) -> int:
        """The most of `want` tokens after `context` priced within `room_s`."""
        if self.price_s(context, want, ratio) <= room_s:
            return want
        low = context // self.block
        high = (context + want - 1) // self.block
        while low < high:
            mid = (low + high + 1) // 2
            if self.price_s(context, mid * self.block - context, ratio) <= room_s:
                low = mid
            else:
                high = mid - 1
        return max(0, low * self.block - context)

    def _one_block(self, context: int, want: int) -> int:
        """The least chunk: up to the next block boundary, or all of `want`."""
        return min(want, self.block - context % self.block)

    def _live(self, context: int, want: int, ratio: float) -> bool:
        """Whether no chunk ran this step and not even one block fits its cap."""
        assert self.step_cap_s is not None
        return (
            not self._step_grants
            and self.price_s(context, self._one_block(context, want), ratio)
            > self.step_cap_s
        )

    def grant(
        self,
        now: float,
        context: int,
        want: int,
        floor: float,
        ratio: float = 1.0,
        *,
        first: bool = False,
        cold: bool = False,
        append: bool = False,
    ) -> int:
        """The tokens of a `want`-token chunk after `context` to run now, 0 if held.

        Under the class's rules, with `floor` c_min, `ratio` rho, `first` marking a
        waiting prefill's first chunk and `cold` or `append` the chunk's kind. A cut
        chunk ends on a block boundary; the grant becomes `last_grant`.
        """
        self.last_grant = None
        if want <= 0:
            return 0
        left = self.left_s(now)
        if self.step_cap_s is None:
            if left < 0:
                self._hold("debt")
                return 0
            tokens = self._fit(context, want, left / self.weight, ratio)
            kind = "run"
            if tokens < min(want, floor):
                tokens, kind = int(min(want, floor)), "live"
        else:
            tokens, kind = 0, "debt"
            if left >= 0 or append:
                room = self.step_cap_s - self.step_s
                tokens = self._fit(context, want, room, ratio)
                kind = "run" if tokens else "cap"
                if tokens and cold:
                    tokens = self._cold_fit(context, tokens, floor, left, ratio)
                    kind = "run" if tokens else "debt"
                if not tokens and kind == "cap" and self._live(context, want, ratio):
                    tokens, kind = self._one_block(context, want), "live"
            if not tokens:
                self._hold(kind)
                return 0
        self._book_grant(context, want, tokens, ratio, first, kind, left)
        return tokens

    def may_start(
        self,
        now: float,
        context: int,
        want: int,
        ratio: float,
        *,
        cold: bool = False,
        append: bool = False,
        floor: float = 0,
    ) -> bool:
        """Whether a waiting prefill would be granted any tokens now; books nothing.

        Its first chunk is at most `want` tokens after `context`.
        """
        assert self.step_cap_s is not None
        left = self.left_s(now)
        if left < 0 and not append:
            self._hold("debt")
            return False
        room = self.step_cap_s - self.step_s
        if cold:
            fits = self._fit(context, want, room, ratio)
            if fits and self._cold_fit(context, fits, floor, left, ratio):
                return True
            if not fits and self._live(context, want, ratio):
                return True
            self._hold("debt" if fits else "cap")
            return False
        first = self._one_block(context, want)
        if self.price_s(context, first, ratio) <= room or self._live(
            context, want, ratio
        ):
            return True
        self._hold("cap")
        return False

    def _cold_fit(
        self,
        context: int,
        capped: int,
        floor: float,
        left: float,
        ratio: float,
    ) -> int:
        """The most of `capped` that `left` covers, or 0 below `floor` tokens."""
        need = min(capped, max(1, int(floor)))
        if self.price_s(context, need, ratio) * self.weight > left:
            return 0
        return max(need, self._fit(context, capped, left / self.weight, ratio))

    def admit(
        self,
        now: float,
        context: int,
        tokens: int,
        ratio: float = 1.0,
    ) -> bool:
        """Floor rule: whether a waiting prefill with first chunk `tokens` may start.

        It starts whole while the balance is not negative and owes its price at once.
        """
        left = self.left_s(now)
        if left < 0:
            self._hold("debt")
            return False
        self._book_grant(context, tokens, tokens, ratio, True, "first", left)
        return True

    def exempt(self, context: int, tokens: int, ratio: float = 1.0) -> None:
        """Price a row the budget does not hold into the step's cap, owing nothing."""
        price = self.price_s(context, tokens, ratio)
        self.step_s += price
        self._counts["exempt_rows"] += 1
        self._seconds["exempt"] += price

    def blocked(self) -> None:
        """Count a row the scheduler cannot run (encoder cache, Mamba alignment)."""
        self._counts["blocked"] += 1

    def refund(self, grant: TpotGrant) -> None:
        """Give back a grant of this step that did not run; idempotent."""
        if grant.kind == "refund":
            return
        self._take(grant, -1)
        self._step_grants -= 1
        self._counts["refunds"] += 1
        grant.kind = "refund"

    def shrink(self, grant: TpotGrant, tokens: int) -> None:
        """Cut this step's grant to the `tokens` that run, counts kept; 0 refunds it."""
        if not tokens:
            self.refund(grant)
            return
        if grant.kind == "refund":
            return
        if tokens >= grant.tokens["granted"]:
            return
        self._take(grant, -1)
        cut = self._priced(
            grant.context,
            grant.tokens["wanted"],
            tokens,
            grant.ratio,
            grant.first,
            grant.kind,
            grant.left,
        )
        grant.seconds, grant.tokens = cut.seconds, cut.tokens
        self._take(grant, 1)

    def _take(self, grant: TpotGrant, sign: int) -> None:
        """Owe and count a grant (`sign` 1), or take it back (-1)."""
        self.pending_s += sign * grant.seconds["granted"] * grant.weight
        self.step_s += sign * grant.seconds["granted"]
        self._count(grant, sign)

    def _count(self, grant: TpotGrant, sign: int) -> None:
        for totals, add in (
            (self._counts, grant.counts),
            (self._seconds, grant.seconds),
            (self._tokens, grant.tokens),
        ):
            for key, value in add.items():
                totals[key] += sign * value

    def _hold(self, kind: str) -> None:
        self._counts["debt_holds" if kind == "debt" else "step_cap_hits"] += 1

    def _book_grant(
        self,
        context: int,
        want: int,
        tokens: int,
        ratio: float,
        first: bool,
        kind: str,
        left: float,
    ) -> None:
        grant = self._priced(context, want, tokens, ratio, first, kind, left)
        self._take(grant, 1)
        self._step_grants += 1
        self.last_grant = grant

    def _priced(
        self,
        context: int,
        want: int,
        tokens: int,
        ratio: float,
        first: bool,
        kind: str,
        left: float,
    ) -> TpotGrant:
        price = self.price_s(context, tokens, ratio)
        cut = tokens < want
        whole = self.price_s(context, want, ratio)
        split = price - tokens / want * whole if cut else 0.0
        live = kind == "live"
        return TpotGrant(
            context=context,
            kind=kind,
            counts={
                "grants": 1,
                "capped": int(cut),
                "first_chunk_caps": int(first and cut),
                "liveness_grants": int(live),
            },
            seconds={
                "granted": price,
                "fixed": ratio
                * self.step.terms.get("prefill_context_tokens", 0.0)
                * context,
                "split": split,
                "live": price if live else 0.0,
                "past_balance": max(0.0, price - max(0.0, left) / self.weight),
            },
            tokens={"granted": tokens, "wanted": want},
            weight=self.weight,
            ratio=ratio,
            first=first,
            left=left,
        )

    def book(self, now: float, pure_s: float, extra_s: float, decoded: int = 1) -> None:
        """Book a step ended at `now`, its seconds weighed by its `decoded` rows."""
        weight = self._weight(decoded)
        self._steps.append((now, pure_s * weight, extra_s * weight))
        self.pure_s += pure_s * weight
        self.extra_s += extra_s * weight
        self.pending_s = 0.0
        self.left_s(now)

    def counters(self) -> dict[str, Any]:
        """`TPOT_COUNTS`, and the seconds and tokens of grants and exempt rows."""
        return {
            **self._counts,
            "seconds": dict(self._seconds),
            "tokens": dict(self._tokens),
        }

    def state(self, now: float) -> dict[str, Any]:
        """The driver's inputs for a D-local chunk's delay, in weighted seconds."""
        left = self.left_s(now)
        return {
            "delta": self.delta,
            "window_s": self.window_s,
            "weighting": self.weighting,
            "weight": self.weight,
            "pure_s": self.pure_s,
            "extra_s": self.extra_s,
            "left_s": left,
            "rule": TPOT_FLOOR_RULE if self.step_share is None else TPOT_STEP_RULE,
            "step_share": self.step_share,
            "step_cap_s": self.step_cap_s,
            **self.counters(),
        }


def budget_delay_s(
    budget: Mapping[str, float] | None,
    extra_s: float,
    decoding: bool,
    owed_only: bool = False,
    append: bool = False,
) -> float:
    """Seconds an engine's TPOT budget holds back `extra_s` of chunks.

    A chunk owes its seconds times the pushed `weight`; what the balance does not
    cover is repaid at `delta` per second of decode work, so over
    `(1 + delta) / delta` seconds of wall time each. `math.inf` where a decoding
    engine at `delta` 0 takes no chunks.

    Args:
        budget: The engine's pushed `TpotBudget.state`; None holds nothing back.
        owed_only: Under the floor rule, wait only for what is already owed.
        append: Under a step share, wait for nothing: an append starts regardless.
    """
    if budget is None or not decoding or math.isinf(budget["delta"]):
        return 0.0
    started = append and budget.get("step_share") is not None
    if (owed_only or started) and budget["delta"] == 0:
        return math.inf if extra_s > 0 else 0.0
    if started:
        return 0.0
    weight = budget.get("weight") or 1
    owed = (0.0 if owed_only else extra_s * weight) - budget["left_s"]
    if owed <= 0:
        return 0.0
    if budget["delta"] == 0:
        return math.inf
    return owed * (1 + budget["delta"]) / (budget["delta"] * weight)


def mixed_tpot_ratio(
    pure_step_s: float, chunk_extra_s: float, backlog_s: float, window_s: float
) -> float:
    """A decoding row's windowed TPOT over its pure decode step's.

    Each step of t = `pure_step_s` carries e = `chunk_extra_s` of chunk until
    `backlog_s` is spent, so the chunks add extra = min(backlog_s,
    window_s * e / (t + e)) to the window and the ratio is
    window_s / (window_s - extra); 1 with no backlog.
    """
    if backlog_s <= 0 or chunk_extra_s <= 0:
        return 1.0
    share = chunk_extra_s / (pure_step_s + chunk_extra_s)
    extra = min(backlog_s, window_s * share)
    return window_s / (window_s - extra)


@dataclass(frozen=True)
class EngineCosts:
    """One engine's prices: the prior's step model, corrected by its own steps.

    Each kind of work is priced at its step class's `ratio` (rho): a prefill alone
    at `STEP_PRE`'s, one riding in decode steps at `STEP_MIX`'s, a decode step at
    `STEP_DEC`'s. `accept_per_row` and `query_per_request` are a decoding row's
    measured tokens emitted and query tokens scheduled (drafts included) a step.
    """

    step: StepModel
    ratio: Mapping[str, float]
    accept_per_row: float = 1.0
    query_per_request: float = 1.0

    @classmethod
    def from_state(
        cls, step: StepModel, state: Mapping[str, Any] | None
    ) -> EngineCosts:
        """The prices an engine's last pushed state (`native_state`) corrects."""
        state = state or {}
        classes = state.get("cost_classes") or {}
        return cls(
            step,
            {
                kind: calibrated_ratio(classes.get(kind), step)
                for kind in (STEP_DEC, STEP_MIX, STEP_PRE)
            },
            state.get("accept_per_row") or 1.0,
            state.get("decode_query_per_request") or 1.0,
        )

    def prefill_s(self, history: int, tokens: int) -> float:
        """S(h, n): `tokens` after `history`, alone, chunk by chunk."""
        return self.ratio[STEP_PRE] * self.step.prefill_s(history, tokens)

    def local_prefill_s(self, history: int, tokens: int, chunk: int = 0) -> float:
        """E(h, n): what that prefill adds to the decode steps it rides in."""
        return self.ratio[STEP_MIX] * self.step.local_prefill_s(history, tokens, chunk)

    def decode_step_s(self, requests: int, context_tokens: float) -> float:
        """t_dec(B, sum ctx): a pure decode step of `requests` rows."""
        return self.ratio[STEP_DEC] * self.step.decode_step_s(
            requests, context_tokens, self.query_per_request
        )


def drain_s(costs: EngineCosts, rows: Iterable[tuple[int, int]]) -> float:
    """Seconds to decode `rows`, each (output tokens left, context), to their ends.

    The batch shrinks as rows end: sorted by tokens left m_u, the rows still running
    emit those between one end and the next in `(m_u - m_{u-1}) / a` steps
    (a = `accept_per_row`), priced at those rows and their context halfway through.
    """
    order = sorted(rows)
    context = math.fsum(start for _, start in order)
    emitted = 0
    seconds = []
    for index, (left, start) in enumerate(order):
        batch = len(order) - index
        span = left - emitted
        if span > 0:
            middle = context + batch * (emitted + span / 2)
            step = costs.decode_step_s(batch, middle)
            seconds.append(span / costs.accept_per_row * step)
            emitted = left
        context -= start
    return math.fsum(seconds)


def marginal_drain_s(
    costs: EngineCosts,
    rows: Sequence[tuple[int, int]],
    added: Sequence[tuple[int, int]],
) -> float:
    """What `added` rows add to the engine's drain of `rows`."""
    return drain_s(costs, [*rows, *added]) - drain_s(costs, rows)


def steady_decode_s(
    costs: EngineCosts, tokens: int, batch: int, context: float
) -> float:
    """F: decode work not yet admitted, at the steady batch it will join.

    `tokens` decoded `batch` rows a step over `context` each: `1 / (a * batch)` steps.
    """
    if tokens <= 0 or batch <= 0:
        return 0.0
    step = costs.decode_step_s(batch, batch * context)
    return tokens / (costs.accept_per_row * batch) * step


def critical_path_s(
    stages: Iterable[tuple[int, int, int]],
    engines: Mapping[str, EngineCosts],
    home: str,
    batch: tuple[int, float],
    kv_transfer_s: Callable[[int], float],
) -> float:
    """CP: the least time a program's stages not yet started take in turn.

    Each stage (matched, append, output bound) prefills its append where that ends
    first, at home or elsewhere with the matched prefix pulled and the append
    returned, and decodes on the home as one more row of `batch` (rows, context).
    """
    rows, context = batch
    home_costs = engines[home]
    seconds = []
    for matched, append, max_tokens in stages:
        moved = kv_transfer_s(matched) + kv_transfer_s(append)
        prefill = min(
            costs.prefill_s(matched, append) + (0.0 if name == home else moved)
            for name, costs in engines.items()
        )
        step = home_costs.decode_step_s(rows + 1, context + matched + append)
        seconds.append(prefill + max_tokens / home_costs.accept_per_row * step)
    return math.fsum(seconds)


@dataclass(frozen=True)
class CostPrior:
    """The fit a run's prices start from, before its engines correct them.

    `model` prices a transfer per KV token, at the calibrated engine's bytes a token.
    """

    model: ProfileCostModel
    step: StepModel
    path: str
    sha256: str


def load_prior(path: str | Path) -> CostPrior:
    """The run's own fit, and its file's hash."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    model = ProfileCostModel.from_dict(json.loads(raw))
    return CostPrior(model, StepModel.from_profile(model), str(path), digest)


@dataclass(frozen=True)
class Placement:
    """One route for a work item: prefill on `engine`, then decode on `home`.

    Attributes:
        finish_s: When it would be done: backlog or budget delay, cost, transfers.
        delta: Busy seconds it adds to each engine's load (negative: takes work off).
        on_d: Whether `engine` is a D, which a tie leaves to a P.
        closed: Why it may not be placed so; None if it may.
        rates: The new `prefill_rate` of each engine it makes decode.
    """

    engine: str
    home: str
    route: str
    cost_s: float
    finish_s: float
    delta: Mapping[str, float]
    on_d: bool = False
    closed: str | None = None
    rates: Mapping[str, float] = field(default_factory=dict)


def prefill_rate(delta: float | None, decoding: bool) -> float:
    """The share of its time an engine may spend on prefill under its TPOT budget."""
    if not decoding or delta is None or math.isinf(delta):
        return 1.0
    return delta / (1 + delta)


def _absorbed_by(
    loads: Mapping[str, float], pool_s: float, rates: Mapping[str, float]
) -> float:
    """The water level L with sum_i rate_i * max(0, L - T_i) = U (`pool_s`)."""
    if pool_s <= 0:
        return -math.inf
    order = sorted(loads.items(), key=lambda item: item[1])
    rate = weighted = 0.0
    for index, (name, load) in enumerate(order):
        rate += rates.get(name, 1.0)
        weighted += rates.get(name, 1.0) * load
        if rate <= 0:
            continue
        level = (pool_s + weighted) / rate
        if index + 1 == len(order) or level <= order[index + 1][1]:
            return level
    return math.inf


def makespan_s(
    loads: Mapping[str, float],
    pool_s: float,
    delta: Mapping[str, float] | None = None,
    rates: Mapping[str, float] | None = None,
) -> float:
    """Phi: the projected makespan, max(max_i T_i, L).

    T_i are `loads` plus a placement's `delta`, and L is when U = `pool_s` is
    taken on, each engine at its `rates` share (`prefill_rate`, 1 if absent) once
    its own load is done: (sum_i T_i + U) / G when every rate is 1.
    """
    new = {name: load + (delta or {}).get(name, 0.0) for name, load in loads.items()}
    return max(max(new.values()), _absorbed_by(new, pool_s, rates or {}))


def decide_minmax(
    loads: Mapping[str, float],
    pool_s: float,
    candidates: Iterable[Placement],
    tie_s: float = 1e-3,
    rates: Mapping[str, float] | None = None,
) -> Placement | None:
    """The min-max rule: the open placement of least Phi; None if all are closed.

    Ties within `tie_s` go to the one done first, then one on a P, sparing D's
    decode steps: while D is busiest what can run elsewhere does, and while it is
    not no placement moves Phi and an append stays where it is done first.
    """
    priced = [
        (makespan_s(loads, pool_s, c.delta, {**(rates or {}), **c.rates}), c)
        for c in candidates
        if c.closed is None
    ]
    if not priced:
        return None
    best = min(phi for phi, _ in priced)
    return min(
        (c for phi, c in priced if phi <= best + tie_s),
        key=lambda c: (c.finish_s, c.on_d, c.engine),
    )

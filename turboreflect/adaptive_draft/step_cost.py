# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors
"""The two-knee step cost model adaptive draft length prices a step with.

A step with B generating rows (row r schedules n_r query tokens, 1 + its drafts,
against ctx_r cached tokens), N scheduled tokens and prefill chunks costs

    T = c0 + c_row * B + [verifies] * c_ver + T_lin(N)
        + sum_r T_attn(ctx_r, n_r) + P

- `T_lin(N) = max(w_dense + w_moe * u(N), alpha * N)`: rows share one weight read; a
  MoE reads the `u(N) = 1 - (1 - p)^N` experts some token routes to, p = top_k /
  experts, and `alpha` meets the weight term at the knee `lin_knee`.
- `T_attn(ctx, n) = beta * ctx * r(n)`, `r(n) = max(1, n / attn_knee)`: each row
  reads its own KV.
- `c_ver`: paid once by a verifying step without prefill, however long its drafts,
  for leaving the uniform decode CUDA graph for the ragged verify path.
- P: the prefill chunks' attention (pairs, chunk squares, context MLA re-expands).
- `switch_tax(B)` per context token of each one-row request in a verifying step: a
  backend running every row on its verify kernel once any row drafts (Triton MLA)
  reads their KV at its cost.
- `launch_s`: in a pure step (no prefill) whose longest row passes a launch edge,
  where the longest query sets the kernel launch, each row up to that edge pays the
  measured jump per context token; none where only the request's tile opens.

With deployment calibration (`turboreflect.calibration.step_time`) a decode or verify
step costs `L(B, D) + sum_r ctx_r * A(B, n_r)`, both measured: L the part no context
moves (weights, experts, graph mode, sampling) at D drafts, so `c_ver` is L's jump
from D = 0 to 1, and A a request's attention per context token at n rows. The last
row count is a uniform batch on another CUDA graph and prices only a step whose rows
all have it. A row raising A by more than `TILE_STEP` opens a tile. Steps with prefill
use the two-knee constants (calibration states them 0, so only generating rows
count), and every price is corrected online per step class (`ClassRatios`).
"""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# (scheduled tokens, context tokens, prefills)
Row = tuple[int, int, bool]
# (batch, ((key, seconds), ...)), both ascending: a calibrated table.
Table = tuple[tuple[int, tuple[tuple[int, float], ...]], ...]
# (batch, ((rows, jump), ...)), both ascending: row counts whose next row opens a
# tile, and the jump per context token every row up to it pays once the longest row
# passes it (0 where only its own tile opens).
Edges = tuple[tuple[int, tuple[tuple[int, float], ...]], ...]

FORM = "two_knee_v1"
# 1 + the decode-row time share a draft may add and stay free (`FREE_SHARE`).
FREE_TOLERANCE = 1.03
TILE_STEP = 1.10
# Values of `KneeStepModel.staircase`.
STAIRCASE_LAUNCH = "launch"
STAIRCASE_REQUEST = "request"


class StepCostError(ValueError):
    """A step model file or adaptive draft length settings this code cannot use."""


@dataclass(frozen=True)
class KneeStepModel:
    """The constants of the two-knee step model (module docstring)."""

    c0: float
    c_row: float
    w_dense: float
    w_moe: float
    moe_p: float
    lin_knee: float
    beta: float
    attn_knee: float
    prefill_pairs: float = 0.0
    prefill_square: float = 0.0
    prefill_context: float = 0.0
    c_ver: float = 0.0
    # L(B, drafts) and A(B, rows) (module docstring).
    calibrated_l: Table = ()
    calibrated_a: Table = ()
    # (batch, switch tax per context token), batches ascending.
    calibrated_tax: tuple[tuple[int, float], ...] = ()
    calibrated_edges: Edges = ()
    # Step class -> initial measured/priced ratio (`ClassRatios`); not priced.
    ratio_priors: dict[str, float] = field(default_factory=dict, compare=False)

    @property
    def alpha(self) -> float:
        """Seconds per scheduled token once linear layers are compute bound."""
        return self.weights_s(self.lin_knee) / self.lin_knee

    def weights_s(self, tokens: float) -> float:
        return self.w_dense + self.w_moe * _active(self.moe_p, tokens)

    def lin_s(self, tokens: float) -> float:
        return max(self.weights_s(tokens), self.alpha * tokens)

    def attn_ratio(self, rows: float, batch: int = 1) -> float:
        """r(rows): a row's attention at `rows` query rows over one, at `batch` rows."""
        if self.calibrated_a:
            one = table_value(self.calibrated_a, batch, 1)
            at = table_value(self.calibrated_a, batch, rows, staircase=self.staircase)
            return at / one if one > 0 else 1.0
        return max(1.0, rows / self.attn_knee)

    def attn_s(
        self, context: float, rows: float, batch: int = 1, uniform: bool = False
    ) -> float:
        if self.calibrated_a:
            return context * table_value(
                self.calibrated_a, batch, rows, uniform, self.staircase
            )
        return self.beta * context * self.attn_ratio(rows, batch)

    @property
    def staircase(self) -> str | None:
        """How a ragged row past the last ragged count is priced (`table_value`).

        `STAIRCASE_LAUNCH` with a launch edge (the slope across it is no row's own),
        `STAIRCASE_REQUEST` with only per-request tiles, None without edges.
        """
        if not self.calibrated_edges:
            return None
        if any(j > 0 for _, edges in self.calibrated_edges for _, j in edges):
            return STAIRCASE_LAUNCH
        return STAIRCASE_REQUEST

    def step_s(
        self, batch: int, tokens: int, prefills: bool = False, uniform: bool = False
    ) -> float:
        """Cost besides attention and prefill: L, or c0 + c_row * B + T_lin + c_ver."""
        if self.calibrated_l and not prefills:
            return table_value(self.calibrated_l, batch, tokens - batch, uniform)
        verify = self.c_ver if tokens > batch and not prefills else 0.0
        return self.c0 + self.c_row * batch + self.lin_s(tokens) + verify

    def switch_tax(self, batch: float) -> float:
        """Seconds per context token a one-row request adds once its step verifies."""
        if not self.calibrated_tax:
            return 0.0
        batches = [b for b, _ in self.calibrated_tax]
        i = min(bisect.bisect_left(batches, batch), len(batches) - 1)
        return self.calibrated_tax[i][1]

    def edges(self, batch: float) -> tuple[tuple[int, float], ...]:
        """(rows, launch jump) pairs at the next classified batch up, or the last."""
        if not self.calibrated_edges:
            return ()
        batches = [b for b, _ in self.calibrated_edges]
        i = min(bisect.bisect_left(batches, batch), len(batches) - 1)
        return self.calibrated_edges[i][1]

    def launch_tax(self, batch: float, rows: float) -> float:
        """Per context token, the launch jumps a longest row of `rows` rows passes."""
        if not self.calibrated_edges:
            return 0.0
        lo, hi, w = _bracket([b for b, _ in self.calibrated_edges], batch)
        low, high = (
            sum(j for e, j in self.calibrated_edges[i][1] if e < rows) for i in (lo, hi)
        )
        return low + w * (high - low)

    def launch_s(self, rows: Sequence[Row]) -> float:
        """Launch-edge seconds a pure step's rows pay; 0 for a step with prefill."""
        if not self.calibrated_edges or not rows or any(p for _, _, p in rows):
            return 0.0
        batch = len(rows)
        top = self.launch_tax(batch, max(n for n, _, _ in rows))
        return sum(c * (top - self.launch_tax(batch, n)) for n, c, _ in rows)

    def launch_ends(self, rows: int, batch: int) -> list[int]:
        """The row counts up to `rows` whose next row passes a launch edge."""
        taxes = [self.launch_tax(batch, n) for n in range(1, rows + 2)]
        return [n for n in range(1, rows + 1) if taxes[n] > taxes[n - 1]]

    def bands(self, batch: float) -> tuple[int, ...]:
        """Each verify band's first row count: launch-edge tiles, or `VERIFY_BANDS`."""
        if not any(j > 0 for _, edges in self.calibrated_edges for _, j in edges):
            return VERIFY_BANDS
        return (2, *(e + 1 for e, j in self.edges(batch) if j > 0))

    def step_class(self, rows: Sequence[Row]) -> str:
        """`step_class` in this model's verify bands at the step's batch."""
        return step_class(rows, self.bands(sum(1 for _, _, p in rows if not p)))

    def uniform(self, rows: Sequence[int], batch: int) -> bool:
        """Whether every row is at the last measured row count (the uniform batch)."""
        if not self.calibrated_a or not rows:
            return False
        last = table_rows(self.calibrated_a, batch, uniform=True)[-1][0]
        return all(n == last for n in rows)

    def tile_ends(self, rows: int, batch: int) -> list[int]:
        """The row counts up to `rows` whose next row opens an attention tile."""
        step = TILE_STEP if self.calibrated_a else 1.0
        ratios = [self.attn_ratio(n, batch) for n in range(1, rows + 2)]
        return [n for n in range(1, rows + 1) if ratios[n] > step * ratios[n - 1]]

    def prefill_s(self, tokens: int, context: int) -> float:
        """A prefill chunk's attention (its linear work is in `lin_s`)."""
        return (
            self.prefill_pairs * tokens * (context + tokens / 2)
            + self.prefill_square * tokens**2
            + self.prefill_context * context
        )

    def price(self, rows: Iterable[Row]) -> float:
        """The seconds of one step of these rows."""
        rows = list(rows)
        counts = [n for n, _, prefills in rows if not prefills]
        generating = len(counts)
        prefilling = generating < len(rows)
        uniform = not prefilling and self.uniform(counts, generating)
        tokens = 0
        seconds = 0.0
        for scheduled, context, prefills in rows:
            tokens += scheduled
            if prefills:
                seconds += self.prefill_s(scheduled, context)
            else:
                seconds += self.attn_s(context, scheduled, generating, uniform)
        if not tokens:
            return 0.0
        step = self.step_s(generating, tokens, prefilling, uniform)
        if any(n > 1 for n in counts):
            seconds += self.switch_tax(generating) * sum(
                c for n, c, p in rows if not p and n <= 1
            )
            if not prefilling:
                # A verifying step never costs less than decode, whatever L measured.
                step = max(step, self.step_s(generating, generating))
                seconds += self.launch_s(rows)
        return step + seconds

    def to_dict(self) -> dict[str, Any]:
        return {"form": FORM, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> KneeStepModel:
        if payload.get("form") != FORM:
            raise StepCostError(f"not a {FORM} step model")
        values = {k: v for k, v in payload.items() if k != "form"}
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise StepCostError(f"unknown step model fields {sorted(unknown)}")
        for name in ("calibrated_l", "calibrated_a", "calibrated_edges"):
            if name in values:
                values[name] = _by_batch(values[name])
        if "calibrated_tax" in values:
            values["calibrated_tax"] = tuple(
                (int(b), float(v)) for b, v in values["calibrated_tax"]
            )
        if "ratio_priors" in values:
            values["ratio_priors"] = {
                k: float(v) for k, v in values["ratio_priors"].items()
            }
        return cls(**values)


def _by_batch(points: Any) -> Table:
    """A JSON `[[batch, [[key, value], ...]], ...]` as tuples (`Table`)."""
    return tuple(
        (int(b), tuple((int(k), float(v)) for k, v in rows)) for b, rows in points
    )


def table_rows(
    table: Table, batch: float, uniform: bool = False
) -> tuple[tuple[int, float], ...]:
    """The calibrated rows at the next measured batch up, or the last.

    The last key, the uniform batch, is included only for a `uniform` step.
    """
    i = min(bisect.bisect_left([b for b, _ in table], batch), len(table) - 1)
    rows = table[i][1]
    return rows if uniform or len(rows) <= 2 else rows[:-1]


def table_value(
    table: Table,
    batch: float,
    key: float,
    uniform: bool = False,
    staircase: str | None = None,
) -> float:
    """The calibrated value at the next measured key up, linear in batch.

    Past the last key, the last per-key slope (at least 0). A non-uniform key past the
    last but one takes the larger of the last two values with `STAIRCASE_LAUNCH`, and
    at least that with `STAIRCASE_REQUEST`.
    """
    lo, hi, w = _bracket([b for b, _ in table], batch)
    low = _rows_value(table[lo][1], key, uniform, staircase)
    if not w:
        return low
    return low + w * (_rows_value(table[hi][1], key, uniform, staircase) - low)


def _rows_value(
    rows: Sequence[tuple[int, float]],
    key: float,
    uniform: bool,
    staircase: str | None,
) -> float:
    step = 0.0
    if not uniform and len(rows) > 2:
        if staircase and rows[-2][0] < key <= rows[-1][0]:
            step = max(rows[-2][1], rows[-1][1])
            if staircase == STAIRCASE_LAUNCH:
                return step
        rows = rows[:-1]
    j = bisect.bisect_left(rows, (math.ceil(key), -math.inf))
    if j < len(rows):
        return rows[j][1]
    if len(rows) == 1:
        return rows[0][1]
    (k0, v0), (k1, v1) = rows[-2], rows[-1]
    return max(step, v1 + max(0.0, (v1 - v0) / (k1 - k0)) * (key - k1))


def _bracket(batches: Sequence[float], batch: float) -> tuple[int, int, float]:
    """(lower, upper, weight of upper) of `batch` among measured batches, clamped."""
    if len(batches) == 1:
        return 0, 0, 0.0
    b = min(max(batch, batches[0]), batches[-1])
    hi = min(max(bisect.bisect_left(batches, b), 1), len(batches) - 1)
    return hi - 1, hi, (b - batches[hi - 1]) / (batches[hi] - batches[hi - 1])


def _active(p: float, tokens: float) -> float:
    """The expected share of experts some of `tokens` tokens routes to."""
    if p <= 0:
        return 0.0
    return 1.0 - (1.0 - p) ** tokens


def load(path: str | Path) -> KneeStepModel:
    """The step model in a file's `verify_cost` (`calibration.step_time`)."""
    model = json.loads(Path(path).read_bytes()).get("verify_cost")
    if not isinstance(model, Mapping):
        raise StepCostError(f"{path} carries no verify_cost step model")
    return KneeStepModel.from_dict(model)


# Step classes the online correction keeps apart: decode; verify by the widest row's
# band, all bands pooled in `STEP_VERIFY`; mixed; prefill only.
STEP_DECODE = "dec"
STEP_VERIFY = "ver"
STEP_MIXED = "mix"
STEP_PREFILL = "pre"
VERIFY_BANDS = (2, 4, 8, 16)


def step_class(rows: Sequence[Row], bands: Sequence[int] = VERIFY_BANDS) -> str:
    """`dec`, `ver<band>` (the widest generating row's band), `mix` or `pre`."""
    generating = [n for n, _, prefills in rows if not prefills]
    if len(generating) < len(rows):
        return STEP_MIXED if generating else STEP_PREFILL
    widest = max(generating, default=1)
    if widest <= 1:
        return STEP_DECODE
    band = max(b for b in bands if b <= widest)
    return f"ver{band}"


def coarse_class(kind: str) -> str:
    return STEP_VERIFY if kind.startswith(STEP_VERIFY) else kind


class ClassRatios:
    """Measured over priced seconds per step class, an online correction.

    Running means weighted toward about the last `1 / weight` steps. A class with fewer
    than `min_steps` borrows its coarse class's ratio (bands pooled), or 1. A prior
    (`KneeStepModel.ratio_priors`) holds until a class's first step, then counts as one
    earlier step, weighing `(1 - weight) ** n` after n. `signed` admits measured seconds
    <= 0 (an increment over a measured base); the ratio stays within `bounds`.
    """

    def __init__(
        self,
        weight: float = 0.1,
        min_steps: int = 10,
        priors: Mapping[str, float] | None = None,
        signed: bool = False,
        bounds: tuple[float, float] = (0.0, math.inf),
    ) -> None:
        self.weight = weight
        self.min_steps = min_steps
        self.priors = dict(priors or {})
        self.signed = signed
        self.bounds = bounds
        self.means: dict[str, list[float]] = {}

    def _update(self, kind: str, seconds: float, priced: float) -> None:
        mean = self.means.get(kind)
        if mean is None:
            if kind not in self.priors:
                self.means[kind] = [seconds, priced, 1]
                return
            mean = self.means[kind] = [self.priors[kind] * priced, priced, 0]
        mean[0] += self.weight * (seconds - mean[0])
        mean[1] += self.weight * (priced - mean[1])
        mean[2] += 1

    def observe(self, kind: str, seconds: float, priced: float) -> None:
        if not (priced > 0 and (seconds > 0 or self.signed and math.isfinite(seconds))):
            return
        self._update(kind, seconds, priced)
        if coarse_class(kind) != kind:
            self._update(coarse_class(kind), seconds, priced)

    def ratio(self, kind: str) -> float:
        lo, hi = self.bounds
        for key in (kind, coarse_class(kind)):
            mean = self.means.get(key)
            if mean is None:
                if key in self.priors:
                    return min(hi, max(lo, self.priors[key]))
            elif (key in self.priors or mean[2] >= self.min_steps) and mean[1] > 0:
                return min(hi, max(lo, mean[0] / mean[1]))
        return 1.0

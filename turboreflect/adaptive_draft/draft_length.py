# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors
"""Adaptive draft length: which suffix drafts a step sends to verification.

SuffixDecoding sizes a draft by how well it matched, blind to what verifying costs.
Once the native schedule has run, each step cuts (never lengthens) every generating
request's draft by its cost at the step's batch B and the request's own context, under
the shared step model (`step_cost.KneeStepModel`).

An accepted draft is worth 1 / lambda seconds, and drafts are kept where their expected
tokens beat lambda x the seconds they add. lambda = rho * B / T_dec: T_dec is the
batch's decode step price at the measured ratio (`decode_ratio`), and rho >= 1 the
achieved speedup over decoding (`verify_speedup`), 1 until `RATE_MIN_STEPS` steps. A
cut draft is not lost: the proposer continues it the next step.

- Free range: a draft adding at most `FREE_SHARE` of a decode row's time to the
  request's attention is verified whatever its odds; the KV read hides it.
- Tiles: past it, a request stops at the tile edge (`KneeStepModel.tile_ends`) or
  proposal maximizing sum_k a(k) - lambda x its attention seconds, a(k) being the
  probability that the first k drafts are all accepted (`ScoreCalibration`).
- Once per step: the verify seconds no context moves (`KneeStepModel.step_s`) and the
  switch tax of one-row requests (`KneeStepModel.switch_tax`) are weighed once; the
  step verifies only if its requests' net gains cover them.
- Launch edges: where the longest row sets every row's launch (`launch_s`), every
  request capped at each launch edge is priced too, so a draft passes an edge only if
  it pays the whole batch's jump.

Priced verify seconds are corrected online per verify band (`ClassRatios`). The first
`warmup_steps` drafted steps verify as proposed; every `explore_every`-th verifying
step extends its most cut request so each score bin and position stays measured; and
`RealizedNetGuard` scales the cost side up while verifying steps lose. A placeholder
draft (-1) is no proposal: never cut or calibrated, and its step skips the guard.
Verification is decode work, never charged to a prefill budget.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from turboreflect.adaptive_draft.step_cost import (
    FREE_TOLERANCE,
    STEP_DECODE,
    STEP_VERIFY,
    VERIFY_BANDS,
    ClassRatios,
    KneeStepModel,
    Row,
    StepCostError,
    load,
)

# Score bins of the suffix score, the sum of the draft's path probabilities: about
# the tokens the suffix tree expects to be accepted.
SCORE_EDGES = (1.0, 2.0, 4.0, 8.0, 16.0)
FREE_SHARE = FREE_TOLERANCE - 1.0
# rho averages about the last 1 / RATE_WEIGHT steps without prefill.
RATE_WEIGHT = 1 / 32
RATE_MIN_STEPS = 10
# A step without prefill over both STALL_S and STALL_FACTOR x its price stalled on
# something outside its rows (a remote KV read under P/D) and is not learned from.
STALL_S = 1.0
STALL_FACTOR = 20.0
# Only a step without prefill whose drafts are priced at least this share of its
# price corrects the verify price; below it, decode-ratio drift swamps the increment.
INCREMENT_SHARE = 1 / 3
INCREMENT_BOUNDS = (0.5, 2.0)
# The realized-net guard (`RealizedNetGuard`).
GUARD_WINDOW = 32
GUARD_MIN_STEPS = 8
# A window loses only past this share of its decode seconds (step-time noise).
GUARD_NOISE = 0.05
GUARD_STEP = 1.1
GUARD_LIMIT = 64.0
# Decode-seconds share charged a verifying step per unit of guard factor past 1, so
# the guard reaches no drafts even where verification is priced at about nothing.
GUARD_SHARE = 1 / 16


def real_drafts(drafts: Sequence[int]) -> int:
    """Drafts before the first placeholder (-1), where the scheduler holds a place."""
    for i, token in enumerate(drafts):
        if token < 0:
            return i
    return len(drafts)


def score_bin(score: float | None) -> int:
    """The bin of a match score; -1 when the proposer gave none."""
    if score is None or not math.isfinite(score):
        return -1
    return sum(score >= edge for edge in SCORE_EDGES)


class ScoreCalibration:
    """P(the first k drafts are all accepted | score bin), from outcomes.

    A verified draft of n tokens, m accepted, counts at each position k <= n, accepted
    if m >= k. A bin's estimate at k shrinks with `bin_weight` pseudo-observations
    toward its own at k - 1 (deep positions are seen mostly in exploration), position
    1 toward the pool, and the pool toward `prior` with `pool_weight`; it never rises
    along k. A bin with no outcome keeps `score_gains`, or a copying bin first seen
    after warmup would never verify. A draft with no score uses the pool.
    """

    def __init__(
        self,
        positions: int,
        prior: float = 0.5,
        pool_weight: float = 2.0,
        bin_weight: float = 4.0,
    ) -> None:
        self.positions = positions
        self.prior = prior
        self.pool_weight = pool_weight
        self.bin_weight = bin_weight
        bins = len(SCORE_EDGES) + 1
        self.reached = [[0] * positions for _ in range(bins)]
        self.hit = [[0] * positions for _ in range(bins)]
        self.pool_reached = [0] * positions
        self.pool_hit = [0] * positions

    def predict(self, score: float | None, drafts: int) -> list[float]:
        b = score_bin(score)
        if b >= 0 and not self.reached[b][0]:
            return score_gains(score, min(drafts, self.positions))
        hit, reached = (
            (self.hit[b], self.reached[b])
            if b >= 0
            else (self.pool_hit, self.pool_reached)
        )
        toward = (self.pool_hit[0] + self.pool_weight * self.prior) / (
            self.pool_reached[0] + self.pool_weight
        )
        out: list[float] = []
        for k in range(min(drafts, self.positions)):
            p = (hit[k] + self.bin_weight * toward) / (reached[k] + self.bin_weight)
            toward = min(out[-1], p) if out else p
            out.append(toward)
        return out

    def observe(self, score: float | None, verified: int, accepted: int) -> None:
        b = score_bin(score)
        verified = min(verified, self.positions)
        for k in range(verified):
            hit = accepted > k
            self.pool_reached[k] += 1
            self.pool_hit[k] += hit
            if b >= 0:
                self.reached[b][k] += 1
                self.hit[b][k] += hit


def score_gains(score: float | None, drafts: int) -> list[float]:
    """a(1 .. drafts) before any outcome, spending the score from the first draft."""
    left = max(score or 0.0, 0.0)
    out = []
    for _ in range(drafts):
        out.append(min(1.0, left))
        left = max(0.0, left - 1.0)
    return out


@dataclass
class Decision:
    """One step's draft cuts."""

    caps: list[int]
    # The kept drafts' net gains missed the step's verify cost, so none verify.
    below_step: bool
    prefill: bool
    # 1 / lambda: seconds an accepted draft of each request is worth.
    values: list[float]


def allocate(
    model: KneeStepModel,
    generating: Sequence[tuple[int, int, Sequence[float]]],
    fixed: Sequence[Row] = (),
    scale: float = 1.0,
    value_scale: float = 1.0,
    guard: float = 1.0,
    speedup: float | None = None,
    band_scale: Callable[[int], float] | None = None,
) -> Decision:
    """How many of its drafts each generating request verifies this step.

    Args:
        generating: (context, proposed drafts, a(1..drafts)) per drafting request.
        fixed: The step's other rows (prefill chunks, rows without drafts).
        scale: Online correction of verification's priced seconds.
        value_scale: Online correction of decode's priced seconds, at which a draft's
            worth is priced.
        guard: `RealizedNetGuard.factor`: it multiplies verification's priced seconds,
            and a verifying step is charged (guard - 1) x `GUARD_SHARE` of its decode
            seconds.
        speedup: rho (at least 1; None is 1).
        band_scale: Correction by the step's widest row count (its verify band),
            pricing the step as a whole; `scale` where None.
    """
    drafts = [min(d, len(a)) for _, d, a in generating]
    # Prefill chunks are the step's own: drafting changes nothing in them.
    decoding = [(1, c, False) for _, c, p in fixed if not p]
    prefill = len(decoding) < len(fixed)
    batch = len(generating) + len(decoding)
    decode_s = model.price([(1, c, False) for c, _, _ in generating] + decoding)
    base_s = model.step_s(batch, batch)
    worth = value_scale * decode_s / batch / max(1.0, speedup or 1.0)
    values = [worth] * len(generating)
    free_s = FREE_SHARE * decode_s / batch
    tax = guard * scale * model.switch_tax(batch)

    def attn_s(i: int, start: int, end: int, s: float = guard * scale) -> float:
        context = generating[i][0]
        return s * (
            model.attn_s(context, end, batch) - model.attn_s(context, start, batch)
        )

    def net(i: int, start: int, end: int, s: float = guard * scale) -> float:
        gain = values[i] * sum(generating[i][2][start - 1 : end - 1])
        return gain - attn_s(i, start, end, s)

    free_rows = [1] * len(generating)
    for i, d in enumerate(drafts):
        # The kernel's own staircase, against the price of a decode row.
        while (
            free_rows[i] <= d
            and attn_s(i, free_rows[i], free_rows[i] + 1, 1.0) <= free_s
        ):
            free_rows[i] += 1
    most = 1 + max(drafts, default=0)
    ends = model.tile_ends(most, batch)

    def launch_tax(rows: int) -> float:
        return model.launch_tax(batch, rows)

    def choose(limit: int) -> list[int]:
        """Each request's best tile edge as drafts, with the longest row at `limit`."""
        caps = []
        for i, d in enumerate(drafts):
            top = min(1 + d, limit)
            start = min(free_rows[i], top)
            # Verifying, a request with no draft pays the tax its drafts save.
            saved = tax * generating[i][0] if start == 1 else 0.0
            edges = [e for e in ends if start < e < top] + [top]
            # A row the step's top passes the edge with pays its jump anyway.
            lift = 0.0 if prefill else guard * scale * generating[i][0]
            best = max(
                [start, *edges],
                key=lambda e: (
                    net(i, start, e)
                    + lift * (launch_tax(e) - launch_tax(start))
                    + saved * (e > start),
                    -e,
                ),
            )
            caps.append(best - 1)
        return caps

    def uniform(caps: list[int]) -> bool:
        return not prefill and model.uniform(
            [1 + c for c in caps] + [1] * len(decoding), batch
        )

    def priced(caps: list[int]) -> tuple[float, float]:
        """`(net gain, step-wide seconds added incl. switch and launch tax)` of caps."""
        band = band_scale(1 + max(caps, default=0)) if band_scale else scale
        s = guard * band
        gain = sum(net(i, 1, 1 + c, s) for i, c in enumerate(caps))
        if not any(caps):
            return gain, 0.0
        step = max(
            0.0,
            model.step_s(batch, batch + sum(caps), uniform=uniform(caps)) - base_s,
        )
        ones = sum(c for _, c, _ in decoding) + sum(
            generating[i][0] for i, c in enumerate(caps) if not c
        )
        switch = s * model.switch_tax(batch) * ones
        launch = 0.0
        if not prefill:
            launch = s * model.launch_s(
                [(1 + c, generating[i][0], False) for i, c in enumerate(caps)]
                + decoding
            )
        charge = (guard - 1.0) * GUARD_SHARE * value_scale * decode_s
        return gain, s * step + switch + launch + charge

    def surplus(caps: list[int]) -> float:
        gain, step = priced(caps)
        return gain - step

    def off_uniform(caps: list[int]) -> list[int]:
        if any(caps) and uniform(caps):
            # Every row at the calibrated last count runs another graph; one
            # draft fewer on the row whose last draft is worth least avoids it.
            j = min(range(len(caps)), key=lambda i: generating[i][2][caps[i] - 1])
            trimmed = [c - (i == j) for i, c in enumerate(caps)]
            if surplus(trimmed) > surplus(caps):
                return trimmed
        return caps

    limits = [most]
    if not prefill:
        limits += [e for e in model.launch_ends(most, batch) if e > 1]
    caps = max((off_uniform(choose(n)) for n in limits), key=surplus)
    if not any(caps):
        return Decision(caps, False, prefill, values)
    gain, step = priced(caps)
    below = gain <= step
    if below:
        caps = [0] * len(caps)
    return Decision(caps, below, prefill, values)


class RealizedNetGuard:
    """Scale the cost side by what verifying steps realized.

    Each pure step (no prefill) chosen to verify files its accepted drafts at
    1 / lambda less what it took over its decode price. While the last `window` lose
    more than `noise` of their decode seconds, `factor` grows by `step` per step up to
    `limit` (verifying nothing), else decays toward 1. While engaged, every
    `explore_every`-th drafted step is decided at factor 1, so evidence keeps coming.
    """

    def __init__(
        self,
        window: int = GUARD_WINDOW,
        min_steps: int = GUARD_MIN_STEPS,
        step: float = GUARD_STEP,
        limit: float = GUARD_LIMIT,
        noise: float = GUARD_NOISE,
    ) -> None:
        # (realized, decode) seconds per step.
        self.recent: deque[tuple[float, float]] = deque(maxlen=window)
        self.min_steps = min_steps
        self.noise = noise
        self.step = step
        self.limit = limit
        self.factor = 1.0

    def observe(self, realized_s: float, decode_s: float) -> None:
        self.recent.append((realized_s, decode_s))
        if len(self.recent) < self.min_steps:
            return
        realized = sum(r for r, _ in self.recent)
        if realized < -self.noise * sum(d for _, d in self.recent):
            self.factor = min(self.limit, self.factor * self.step)
        else:
            self.factor = max(1.0, self.factor / self.step)


@dataclass(frozen=True)
class AdaptiveDraftSettings:
    """`additional_config.adaptive_draft`."""

    step_model_path: str
    explore_every: int = 16
    warmup_steps: int = 64

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> AdaptiveDraftSettings:
        unknown = set(config) - set(cls.__dataclass_fields__)
        if unknown:
            raise StepCostError(f"unknown adaptive_draft fields {sorted(unknown)}")
        settings = cls(**config)
        if settings.explore_every < 0 or settings.warmup_steps < 0:
            raise StepCostError("adaptive_draft exploration counts must be >= 0")
        return settings


class AdaptiveDraftMixin:
    """A scheduler that cuts suffix drafts by adaptive draft length.

    Mixed in ahead of a `Scheduler` subclass; active when `additional_config` has an
    `adaptive_draft` block and drafts are on. Reads each draft's score
    (`DraftTokenIds.draft_scores`), cuts drafts after the native schedule, times the
    step from schedule to update, and learns from what verification accepted.
    """

    _draft_clock = staticmethod(time.monotonic)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._draft_scores: dict[str, float] = {}
        self._timed_step: dict[str, Any] | None = None
        self._drafted_steps = 0
        self._verifying_steps = 0
        self._net_guard = RealizedNetGuard()
        config = (self.vllm_config.additional_config or {}).get("adaptive_draft")
        self._draft_settings = None
        if not config or not self.num_spec_tokens:
            return
        self._draft_settings = AdaptiveDraftSettings.from_config(config)
        self._step_model = load(self._draft_settings.step_model_path)
        self._class_ratios = ClassRatios(priors=self._step_model.ratio_priors)
        # Measured over priced seconds verification adds, per verify class.
        self._increment_ratios = ClassRatios(signed=True, bounds=INCREMENT_BOUNDS)
        self._score_calibration = ScoreCalibration(self.num_spec_tokens)
        # Running means behind rho (`verify_speedup`).
        self._mean_emitted = self._mean_decoded = 0.0
        self._rate_steps = 0

    @property
    def adaptive_draft_on(self) -> bool:
        return self._draft_settings is not None

    def update_draft_token_ids(self, draft_token_ids) -> None:
        super().update_draft_token_ids(draft_token_ids)
        scores = getattr(draft_token_ids, "draft_scores", None)
        if self._draft_settings is not None and scores:
            for request_id, score in zip(draft_token_ids.req_ids, scores):
                self._draft_scores[request_id] = score

    def _drafts_warming(self) -> bool:
        return self._drafted_steps <= self._draft_settings.warmup_steps

    def _draft_gains(self, request_id: str, drafts: int) -> list[float]:
        score = self._draft_scores.get(request_id)
        if not self._score_calibration.pool_reached[0]:
            return score_gains(score, drafts)
        return self._score_calibration.predict(score, drafts)

    @staticmethod
    def _trim_drafts(output, request, keep: int) -> None:
        """Verify only the first `keep` of the drafts `output` scheduled for `request`.

        The step, the request's progress and its in-flight count drop the rest.
        """
        rid = request.request_id
        drafts = output.scheduled_spec_decode_tokens[rid]
        cut = len(drafts) - keep
        if keep:
            output.scheduled_spec_decode_tokens[rid] = drafts[:keep]
        else:
            del output.scheduled_spec_decode_tokens[rid]
        output.num_scheduled_tokens[rid] -= cut
        output.total_num_scheduled_tokens -= cut
        request.num_computed_tokens -= cut
        request.num_in_flight_tokens -= cut

    def _cut_drafts(self, output) -> dict[str, Any] | None:
        drafts = output.scheduled_spec_decode_tokens
        cuttable, fixed = [], []
        held = False
        for rid, n in output.num_scheduled_tokens.items():
            request = self.requests[rid]
            context = request.num_computed_tokens - n
            real = real_drafts(drafts.get(rid, ()))
            if real:
                cuttable.append((request, context, real))
            else:
                held = held or bool(drafts.get(rid))
                prefills = (
                    context < request.num_tokens - 1 or not request.output_token_ids
                )
                fixed.append((n, context, prefills))
        if not cuttable:
            return None
        self._drafted_steps += 1
        gains = {
            r.request_id: self._draft_gains(r.request_id, d) for r, _, d in cuttable
        }
        batch = len(cuttable) + sum(1 for _, _, p in fixed if not p)
        decision = None
        s = self._draft_settings
        guard = self._net_guard.factor
        probing = (
            guard > 1.0
            and bool(s.explore_every)
            and self._drafted_steps % s.explore_every == 0
        )
        if self._drafts_warming():
            mode, caps = "warmup", [d for _, _, d in cuttable]
        else:
            model = self._step_model
            band_scale = None
            if model.bands(batch) != VERIFY_BANDS:
                bands = model.bands(batch)

                def band_scale(rows: int) -> float:
                    band = max(b for b in bands if b <= max(rows, 2))
                    return self._increment_ratios.ratio(f"ver{band}")

            decision = allocate(
                model,
                [(c, d, gains[r.request_id]) for r, c, d in cuttable],
                fixed,
                self._increment_ratios.ratio(STEP_VERIFY),
                self.decode_ratio(),
                1.0 if probing else guard,
                self.verify_speedup(),
                band_scale,
            )
            caps = list(decision.caps)
            if decision.below_step:
                mode = "below_step"
            elif not any(caps):
                mode = "none"
            else:
                mode = "guard_probe" if probing else "verify"
                self._verifying_steps += 1
                if s.explore_every and self._verifying_steps % s.explore_every == 0:
                    # The most cut request verifies its whole proposal, short of a
                    # launch edge the step does not pass already.
                    j = max(range(len(caps)), key=lambda i: cuttable[i][2] - caps[i])
                    d = cuttable[j][2]
                    if not decision.prefill:
                        top = 1 + max(caps)
                        ends = model.launch_ends(1 + d, batch)
                        d = min([d] + [e - 1 for e in ends if e >= top])
                    if caps[j] < d:
                        caps[j] = d
                        mode = "explore"
        for (request, _, _), cap in zip(cuttable, caps):
            if cap < len(drafts[request.request_id]):
                self._trim_drafts(output, request, cap)
        out: dict[str, Any] = {}
        if decision is not None:
            out.update(
                taus={
                    r.request_id: v for (r, _, _), v in zip(cuttable, decision.values)
                },
                guarded=mode in ("verify", "guard_probe")
                and not decision.prefill
                and not held,
            )
        return out

    def schedule(self, *args, **kwargs):
        if self._draft_settings is None:
            return super().schedule(*args, **kwargs)
        started = self._draft_clock()
        output = super().schedule(*args, **kwargs)
        decision = self._cut_drafts(output)
        rows: list[Row] = []
        verified: dict[str, tuple[float | None, int]] = {}
        taus = (decision or {}).pop("taus", {})
        guarded = (decision or {}).pop("guarded", False)
        for request_id, n in output.num_scheduled_tokens.items():
            request = self.requests[request_id]
            context = request.num_computed_tokens - n
            prefills = context < request.num_tokens - 1 or not request.output_token_ids
            rows.append((n, context, prefills))
            drafts = real_drafts(
                output.scheduled_spec_decode_tokens.get(request_id, ())
            )
            if drafts:
                verified[request_id] = (self._draft_scores.get(request_id), drafts)
        self._timed_step = (
            {
                "started": started,
                "rows": rows,
                "verified": verified,
                "taus": taus,
                "guarded": guarded and bool(verified),
            }
            if rows
            else None
        )
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        step, self._timed_step = self._timed_step, None
        if self._draft_settings is None or step is None:
            return outputs
        seconds = self._draft_clock() - step["started"]
        rows = step["rows"]
        decoding = self._step_model.price([(1, c, False) for _, c, _ in rows])
        decode_s = self.decode_ratio() * decoding
        kind = self._step_model.step_class(rows)
        priced = self._step_model.price(rows)
        prefill = any(p for _, _, p in rows)
        expected = self._class_ratios.ratio(kind) * priced
        timed = prefill or seconds <= max(STALL_S, STALL_FACTOR * expected)
        if timed:
            self._class_ratios.observe(kind, seconds, priced)
        sampled = model_runner_output.sampled_token_ids
        pure = timed and not prefill
        if pure and priced - decoding >= INCREMENT_SHARE * priced:
            self._increment_ratios.observe(kind, seconds - decode_s, priced - decoding)
        if seconds > 0 and decode_s > 0 and pure:
            means = (
                sum(len(tokens) for tokens in sampled or ()),
                len(rows) * seconds / decode_s,
            )
            if self._rate_steps:
                old = (self._mean_emitted, self._mean_decoded)
                means = tuple(o + RATE_WEIGHT * (m - o) for o, m in zip(old, means))
            self._mean_emitted, self._mean_decoded = means
            self._rate_steps += 1
        saved_s = 0.0
        if sampled:
            for i, request_id in enumerate(model_runner_output.req_ids):
                entry = step["verified"].get(request_id)
                if entry is None or i >= len(sampled):
                    continue
                score, drafts = entry
                accepted = max(0, len(sampled[i]) - 1)
                self._score_calibration.observe(score, drafts, accepted)
                saved_s += accepted * step["taus"].get(request_id, 0.0)
        if step["guarded"] and seconds > 0 and timed:
            self._net_guard.observe(saved_s - (seconds - decode_s), decode_s)
        return outputs

    def _free_request(self, request, delay_free_blocks: bool = False):
        self._draft_scores.pop(request.request_id, None)
        return super()._free_request(request, delay_free_blocks)

    def decode_ratio(self) -> float:
        """Measured over priced decode seconds: the lower of decode and verify ratios.

        Decode-only steps are few where nearly every step drafts; the lower ratio errs
        toward pricing verification dearer.
        """
        return min(
            self._class_ratios.ratio(STEP_DECODE), self._class_ratios.ratio(STEP_VERIFY)
        )

    def verify_speedup(self) -> float | None:
        """rho over recent steps without prefill; None before `RATE_MIN_STEPS`."""
        if self._rate_steps < RATE_MIN_STEPS or not self._mean_decoded > 0:
            return None
        return self._mean_emitted / self._mean_decoded

    def verify_decode_s(self, rows: Sequence[Row]) -> float:
        """Generating rows' seconds alone: what a TPOT budget books as decode."""
        kind = self._step_model.step_class(rows)
        return self._class_ratios.ratio(kind) * self._step_model.price(rows)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""Worker scheduler for locality-aware P/D: native chunked prefill, decodes first.

A prefill routed to D shares each step's token budget with D's decodes. On top of
native chunked prefill:

- Decodes first: `running` is sorted decodes, then prefill chunks, each by
  priority and admission, so the preemption victim is the last prefill, not a decode.
- In-flight reservation: a local prefill's admission leaves free the blocks
  in-flight prefills still need, as native admission does only for async KV
  loads; under `scheduler_reserve_full_isl` its whole sequence must also fit.
- Priority on P: later stages (class 0) precede cold ones (class 1). A waiting
  later stage is admitted ahead of a running cold chunk, and no cold one joins
  it that step, so the decodes after it keep their tokens.
- TPOT budget: with `tpot_delta`, a decoding step's prefill chunks are held to
  `policy.TpotBudget`, each priced at its own context by the step model times
  this engine's measured mixed-step ratio.

With `tpot_step_share` one step's chunks share its cap: a later stage starts on
it whatever the balance, a P's generation stage once the balance is not negative,
and a cold chunk on D spends only the balance it finds and waits for its floor
c_min (or less, if that finishes its held encoder input). A held request is passed
over for the step, and later local prefills of that step wait behind it; one vLLM
runs nothing of (encoder cache, Mamba alignment) is granted nothing. Without a
step share, running chunks are held to the balance and a waiting prefill's first
chunk is admitted whole. Requests computing nothing yet, session segments and rows
with at most a decode's tokens left are never held.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from typing import Any

from turboreflect.adaptive_draft.draft_length import AdaptiveDraftMixin
from turboreflect.context_reuse.admission import STAGE_CLASS_STRIDE
from turboreflect.locality_pd.decode_scheduler import (
    ENGINE_FIELDS,
    DecodeCostScheduler,
    load_profile,
    native_state,
)
from turboreflect.locality_pd.policy import (
    STEP_DEC,
    STEP_MIX,
    TPOT_TIME,
    TPOT_WINDOW_S,
    PolicyInputError,
    StepModel,
    TpotBudget,
    TpotGrant,
    calibrated_ratio,
    step_features,
)
from turboreflect.metrics.trace import EngineTrace
from vllm.v1.request import Request, RequestStatus

# The share of a segment's own cost its fixed cost may reach.
SEGMENT_OVERHEAD_EPS = 0.25


def whole_blocks(tokens: int, block: int) -> int:
    return tokens // block * block


def min_segment_tokens(
    terms: Mapping[str, float],
    context: int,
    block: int,
    eps: float,
) -> float:
    """c_min(H): the fewest tokens, in whole blocks, whose segment pays for itself.

    A segment of c tokens after H = `context` pays for itself when
    `f*H <= eps * (b*c + pairs*c*(H + c/2) + square*c^2)`, with b, pairs, square
    and f the step model's `prefill_tokens`, `prefill_pairs`, `prefill_square` and
    `prefill_context_tokens` (`policy.chunk_seconds`); f*H is what every segment
    repeats over its context, such as MLA's re-expansion of cached latents. It
    rides in a decode step, so it pays no step constant. Infinity if none pays.
    """
    fixed = terms.get("prefill_context_tokens", 0.0) * context

    def own(c: int) -> float:
        return (
            terms["prefill_tokens"] * c
            + terms["prefill_pairs"] * c * (context + c / 2)
            + terms["prefill_square"] * c * c
        )

    c = block
    while fixed > eps * own(c):
        if c > 1 << 22:
            return math.inf
        c *= 2
    low = c // 2
    while c - low > block:
        mid = whole_blocks((low + c) // 2, block)
        if mid <= low:
            break
        if fixed <= eps * own(mid):
            c = mid
        else:
            low = mid
    return c


def _cold(request: Request) -> bool:
    """A request of priority class 1, a program's first stage."""
    return request.priority >= STAGE_CLASS_STRIDE


class AppendScheduler(DecodeCostScheduler):
    """D's or P's worker scheduler: chunked, decodes first."""

    def __init__(self, vllm_config, *args, **kwargs):
        config = (vllm_config.additional_config or {}).get("locality_pd")
        if not isinstance(config, Mapping):
            raise PolicyInputError("additional_config.locality_pd is required")
        if set(config) - ENGINE_FIELDS:
            raise PolicyInputError("unknown additional_config.locality_pd fields")
        self.pd_role = config.get("role")
        # Without a TPOT budget D prices nothing with the calibration.
        delta = config.get("tpot_delta")
        if delta is not None:
            self._tpot = TpotBudget(
                delta,
                config.get("tpot_window_s") or TPOT_WINDOW_S,
                StepModel.from_profile(load_profile(config)),
                vllm_config.cache_config.block_size,
                config.get("tpot_step_share"),
                config.get("tpot_weighting", TPOT_TIME),
            )
            # This step's grants, refunded on preemption or a failed admission.
            self._step_grants: dict[str, TpotGrant] = {}
            self._waiting_grant: tuple[str, TpotGrant] | None = None
        # Each running request's place in native `running`: its admission.
        self._admissions = itertools.count()
        self._admitted: dict[str, int] = {}
        # Admitted this step, so scheduled before `running` was sorted.
        self._step_admitted: set[str] = set()
        self._decoding = False
        self._waiting_first = False
        self._now = 0.0
        # The last refusal was the step cap's; a local prefill was held this step,
        # so later ones wait too.
        self._held = False
        self._held_local = False
        super().__init__(vllm_config, *args, **kwargs)

    def _cost_step(self) -> StepModel | None:
        return None if self._tpot is None else self._tpot.step

    def _order(self, request: Request) -> tuple[bool, int, int]:
        return (
            request.is_prefill_chunk,
            request.priority,
            self._admitted[request.request_id],
        )

    def schedule(self, throttle_prefills: bool = False):
        self.running.sort(key=self._order)
        self._step_admitted.clear()
        self._decoding = any(not r.is_prefill_chunk for r in self.running)
        if self._tpot is None:
            return super().schedule(throttle_prefills=throttle_prefills)
        self._now = self._decode_clock()
        self._held_local = False
        self._tpot.start_step(
            self._now, sum(not r.is_prefill_chunk for r in self.running)
        )
        output = super().schedule(throttle_prefills=throttle_prefills)
        self._step_grants.clear()
        self._waiting_grant = None
        # Rows computing only a prompt's last token (D's first step after P's
        # prefill): the budget cannot defer or shrink them, so charges none.
        self._step_tails = [
            not request.output_token_ids
            and request.num_computed_tokens - n >= request.num_tokens - 1
            for request, n in (
                (self.requests[rid], n)
                for rid, n in output.num_scheduled_tokens.items()
            )
        ]
        return output

    def _ready(self, request: Request) -> bool:
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            return request.request_id in self.finished_recving_kv_req_ids
        return not self._is_blocked_waiting_status(request.status)

    def _waiting_before_running(self) -> bool:
        self._waiting_first = any(
            r.is_prefill_chunk and _cold(r) for r in self.running
        ) and any(
            not _cold(r) and self._ready(r)
            for queue in (
                self.async_load_waiting,
                self.skipped_waiting,
                self.waiting,
            )
            for r in queue
        )
        return self._waiting_first

    @staticmethod
    def _computes_nothing_local(request: Request) -> bool:
        """Back from (or starting) a KV load, or a session's next segment."""
        params = request.kv_transfer_params or {}
        return request.num_computed_tokens > 0 or bool(params.get("do_remote_prefill"))

    def _step_capped(self) -> bool:
        return (
            self._tpot is not None
            and self._tpot.step_cap_s is not None
            and self._decoding
        )

    def _loaded_append(self, request: Request) -> bool:
        """Whether a request is back from its KV load with more than a decode left."""
        computed = request.num_computed_tokens
        return (
            computed > 0
            and not request.resumable
            and self._ready(request)
            and not self._last_tokens(request, computed)
        )

    def _can_schedule_waiting_request(self, request, scheduled_tokens) -> bool:
        self._held = False
        if not super()._can_schedule_waiting_request(request, scheduled_tokens):
            return False
        capped = self._step_capped()
        loaded = capped and self._loaded_append(request)
        if self._computes_nothing_local(request) and not loaded:
            return True
        if self._waiting_first and _cold(request) and not loaded:
            return False
        if self._tpot is None or not self._decoding:
            return True
        chunk = self.max_num_scheduled_tokens - sum(scheduled_tokens.values())
        threshold = self.scheduler_config.long_prefill_token_threshold
        if threshold > 0:
            chunk = min(chunk, threshold)
        extra = (
            (request.sampling_params.extra_args or {})
            if request.sampling_params
            else {}
        )
        # Its load's end, else the driver's history hint, else the dearest place.
        context = request.num_computed_tokens or (
            (extra.get("locality_pd_prefill") or {}).get("history_tokens")
        )
        if context is None:
            tokens = max(1, min(chunk, request.num_tokens))
            context = request.num_tokens - tokens
        else:
            tokens = max(1, min(chunk, request.num_tokens - context))
        ratio = self._ratio(STEP_MIX)
        if not capped:
            return self._tpot.admit(self._now, context, tokens, ratio)
        if self._last_tokens(request, request.num_computed_tokens):
            return True
        cold = self._borrowed_cold(request)
        if self._held_local or not self._tpot.may_start(
            self._now,
            context,
            tokens,
            ratio,
            cold=cold,
            append=not _cold(request),
            floor=self._floor(context) if cold else 0,
        ):
            self._held = self._held_local = True
            return False
        return True

    def _borrowed_cold(self, request: Request) -> bool:
        """A program's first stage on D, which a P could prefill instead."""
        return self.pd_role == "d" and _cold(request)

    def _passes_over_waiting_request(self, request: Request) -> bool:
        return self._held

    def _last_tokens(self, request: Request, computed: int) -> bool:
        """No more than a decode's tokens left past `computed`."""
        return request.num_tokens - computed <= 1 + self.num_spec_tokens

    def _floor(self, context: int) -> float:
        """c_min at `context`: the least chunk worth running."""
        assert self._tpot is not None
        return min_segment_tokens(
            self._tpot.step.terms,
            context,
            self._tpot.block,
            SEGMENT_OVERHEAD_EPS,
        )

    def _tokens_to_free_encoder_cache(self, request: Request) -> float:
        """Tokens until `request`'s first held encoder input frees, rounded to a
        chunk Mamba alignment runs whole; inf if it holds none.
        """
        held = self.encoder_cache_manager.get_cached_input_ids(request)
        if not held:
            return math.inf
        assert request.mm_features is not None
        lookahead = 1 if self.use_eagle else 0
        computed = request.num_computed_tokens
        tokens = min(
            request.mm_features[i].mm_position.offset
            + request.mm_features[i].mm_position.length
            + lookahead
            - computed
            for i in held
        )
        if (
            self.need_mamba_block_aligned_split
            and self._mamba_block_aligned_split(request, tokens) < tokens
        ):
            # A chunk ending mid-block before the prompt's last one runs only
            # up to that block's start: run on to its end.
            block = self.cache_config.block_size
            tokens = -(-(computed + tokens) // block) * block - computed
        return tokens

    def _blocked(self, request: Request, computed: int, tokens: int) -> bool:
        """Whether vLLM runs nothing of a `tokens` chunk after `computed` this step.

        So when Mamba alignment leaves it no block, or its next encoder input is
        uncached and exceeds what the encoder cache can free
        (`EncoderCacheManager.can_allocate`, without evicting) or a step's budget.
        """
        if self.need_mamba_block_aligned_split and not (
            self._mamba_block_aligned_split(
                request, tokens, computed - request.num_computed_tokens
            )
        ):
            return True
        if (
            not request.has_encoder_inputs
            or self.is_encoder_decoder
            or self.ec_connector is not None
        ):
            return False
        assert request.mm_features is not None
        cache = self.encoder_cache_manager
        position = computed + (1 if self.use_eagle else 0)
        room = min(self.max_num_encoder_input_tokens, cache.num_freeable_slots)
        for i, feature in enumerate(request.mm_features):
            span = feature.mm_position
            if span.offset + span.length <= computed:
                continue
            if feature.identifier in cache.cached:
                continue
            return span.offset <= position and request.get_num_encoder_embeds(i) > room
        return False

    def _cap_waiting_tokens(
        self, request: Request, num_computed_tokens: int, num_new_tokens: int
    ) -> int:
        tokens = super()._cap_waiting_tokens(
            request, num_computed_tokens, num_new_tokens
        )
        if not self._step_capped():
            return tokens
        assert self._tpot is not None
        ratio, rid = self._ratio(STEP_MIX), request.request_id
        if request.resumable:
            self._tpot.exempt(num_computed_tokens, tokens, ratio)
            return tokens
        if self._last_tokens(request, num_computed_tokens):
            return tokens
        if self._blocked(request, num_computed_tokens, tokens):
            self._tpot.blocked()
            self._waiting_grant = None
            self._held_local = True
            return 0
        # Its context after the prefix-cache lookup may price its first
        # block past what `may_start` saw: then it is passed over.
        cold = self._borrowed_cold(request)
        granted = self._tpot.grant(
            self._now,
            num_computed_tokens,
            tokens,
            self._floor(num_computed_tokens) if cold else 0,
            ratio,
            first=True,
            cold=cold,
            append=not _cold(request),
        )
        grant = self._tpot.last_grant
        self._waiting_grant = None if grant is None else (rid, grant)
        self._held_local = self._held_local or not granted
        return granted

    def _ratio(self, kind: str) -> float:
        assert self._tpot is not None
        return calibrated_ratio(self._cost_classes.get(kind), self._tpot.step)

    def _cap_running_tokens(self, request: Request, num_new_tokens: int) -> int:
        tokens = super()._cap_running_tokens(request, num_new_tokens)
        if (
            self._tpot is None
            or not self._decoding
            or not request.is_prefill_chunk
            or not tokens
        ):
            return tokens
        context, ratio = request.num_computed_tokens, self._ratio(STEP_MIX)
        rid = request.request_id
        if request.resumable:
            self._tpot.exempt(context, tokens, ratio)
            return tokens
        if self._last_tokens(request, context):
            return tokens
        if self._blocked(request, context, tokens):
            self._tpot.blocked()
            return 0
        capped = self._tpot.step_cap_s is not None
        cold = capped and self._borrowed_cold(request)
        floor = self._floor(context) if cold or not capped else 0
        if cold:
            floor = min(floor, self._tokens_to_free_encoder_cache(request))
        granted = self._tpot.grant(
            self._now,
            context,
            tokens,
            floor,
            ratio,
            cold=cold,
            append=not _cold(request),
        )
        if granted and capped:
            assert self._tpot.last_grant is not None
            self._step_grants[rid] = self._tpot.last_grant
        return granted

    def _settle_capped_tokens(self, request: Request, num_new_tokens: int) -> None:
        if self._tpot is None:
            return
        rid = request.request_id
        if self._waiting_grant is not None and self._waiting_grant[0] == rid:
            grant = self._waiting_grant[1]
            if not num_new_tokens:
                self._waiting_grant = None
        elif num_new_tokens:
            grant = self._step_grants.get(rid)
        else:
            grant = self._step_grants.pop(rid, None)
        if grant is not None:
            self._tpot.shrink(grant, num_new_tokens)

    def _on_waiting_allocation_failure(self) -> None:
        if self._tpot is not None and self._waiting_grant is not None:
            self._tpot.refund(self._waiting_grant[1])
            self._waiting_grant = None
        super()._on_waiting_allocation_failure()

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        if self._tpot is not None:
            grant = self._step_grants.pop(request.request_id, None)
            if grant is not None:
                self._tpot.refund(grant)
        super()._preempt_request(request, timestamp)

    def _select_preemption_victim(self) -> Request | None:
        victim = max(
            (r for r in self.running if r.request_id not in self._step_admitted),
            key=self._order,
            default=None,
        )
        return super()._select_preemption_victim() if victim is None else victim

    def _on_waiting_request_scheduled(self, request: Request) -> None:
        self._admitted[request.request_id] = next(self._admissions)
        self._step_admitted.add(request.request_id)
        if self._tpot is not None and self._waiting_grant is not None:
            rid, grant = self._waiting_grant
            self._step_grants[rid] = grant
            self._waiting_grant = None
        super()._on_waiting_request_scheduled(request)

    def update_from_output(self, scheduler_output, model_runner_output):
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        if self._tpot is not None and self._last_step is not None:
            rows, seconds, ended = self._last_step
            tails = getattr(self, "_step_tails", None) or [False] * len(rows)
            decodes = [
                (n, context, False) if tail else (n, context, prefills)
                for (n, context, prefills), tail in zip(rows, tails)
                if tail or not prefills
            ]
            pure = extra = 0.0
            if len(decodes) == len(rows):
                pure = seconds
            elif decodes:
                pure = self._pure_decode_s(decodes)
                extra = max(0.0, seconds - pure)
            self._tpot.book(ended, pure, extra, sum(not p for _, _, p in rows))
        return outputs

    def _pure_decode_s(self, decodes: list[tuple[int, int, bool]]) -> float:
        """A mixed step's generating rows alone, verification included; uncharged."""
        assert self._tpot is not None
        return self._ratio(STEP_DEC) * self._tpot.step.price_s(step_features(decodes))

    def _free_request(self, request: Request, delay_free_blocks: bool = False):
        self._admitted.pop(request.request_id, None)
        return super()._free_request(request, delay_free_blocks)

    def _admission_reserved_blocks(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int,
        load_kv_async: bool,
    ) -> int:
        # Native admission's sign of a request back from its KV load.
        if load_kv_async or request.num_computed_tokens:
            return super()._admission_reserved_blocks(
                request, num_new_tokens, num_new_local_computed_tokens, load_kv_async
            )
        reserved = sum(
            self._request_remaining_blocks(other)
            for other in self._inflight_prefills
            if other is not request
        )
        if not self._waiting_full_sequence_must_fit(request):
            return reserved
        computed = request.num_computed_tokens + num_new_local_computed_tokens
        end = computed + num_new_tokens + self.num_lookahead_tokens
        return reserved + self._blocks_after_chunk(request, computed, end)

    def _blocks_after_chunk(self, request: Request, computed: int, end: int) -> int:
        """Blocks `request`'s whole sequence needs past its slots up to `end`, under
        the full-sequence admission check's cap.
        """
        full = min(request.num_tokens, self.max_model_len)

        def blocks(tokens: int) -> int:
            return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=tokens,
                new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks,
                num_encoder_tokens=0,
                total_computed_tokens=computed,
                num_local_computed_tokens=computed,
                num_tokens_main_model=tokens,
                apply_admission_cap=True,
            )

        return blocks(full) - blocks(min(end, full))

    def get_locality_pd_state(self) -> dict[str, Any]:
        """What the driver routes on, read between two steps."""
        return native_state(self)


class AppendTraceScheduler(EngineTrace, AppendScheduler):
    """`AppendScheduler` with the engine trace."""


class AppendAdaptiveDraftScheduler(AdaptiveDraftMixin, AppendScheduler):
    """`AppendScheduler` with adaptive draft length cutting suffix drafts.

    Generating rows are booked as decode work at adaptive draft length's price, so
    verification never spends the TPOT budget; only rows the cut leaves count.
    """

    def schedule(self, throttle_prefills: bool = False):
        output = super().schedule(throttle_prefills=throttle_prefills)
        if self.adaptive_draft_on:
            self._take_step_rows(output)
        return output

    def _pure_decode_s(self, decodes: list[tuple[int, int, bool]]) -> float:
        if self.adaptive_draft_on:
            return self.verify_decode_s(decodes)
        return super()._pure_decode_s(decodes)


class AppendAdaptiveDraftTraceScheduler(EngineTrace, AppendAdaptiveDraftScheduler):
    """`AppendAdaptiveDraftScheduler` with the engine trace."""

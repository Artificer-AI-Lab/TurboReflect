# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""The worker scheduler's decode-cost measurements and native state."""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from turboreflect.locality_pd.policy import (
    STEP_DEC,
    ProfileCostModel,
    StepModel,
    TpotBudget,
    calibrated_ratio,
    step_class,
    step_features,
)
from vllm.v1.core.kv_cache_utils import kv_pool_layout
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

# `additional_config.locality_pd`'s fields (`PolicyConfig.engine_extra_config`).
ENGINE_FIELDS = frozenset(
    {
        "role",
        "profile_path",
        "tpot_delta",
        "tpot_window_s",
        "tpot_step_share",
        "tpot_weighting",
    }
)


# A step's weight in the running averages: about the last ten steps.
DECODE_COST_WEIGHT = 0.1
# A step stalled on what its rows do not cost (a KV pull) stays out of the means:
# one longer than `DECODE_STALL_S` and `DECODE_STALL_FACTOR` times `_expected_s`,
# or, before its class has a mean, than `DECODE_STALL_S` while remote KV arrives.
DECODE_STALL_S = 1.0
DECODE_STALL_FACTOR = 20.0
# Finished remote KV reads an engine's state lists: far more than end between polls.
READ_LOG = 32


def _averaged(last: float | None, value: float) -> float:
    return value if last is None else last + DECODE_COST_WEIGHT * (value - last)


def _class_means(
    means: dict[str, Any] | None, features: dict[str, float], seconds: float
) -> dict[str, Any]:
    if means is None:
        return {"x": features, "t": seconds, "n": 1}
    return {
        "x": {k: _averaged(means["x"][k], v) for k, v in features.items()},
        "t": _averaged(means["t"], seconds),
        "n": means["n"] + 1,
    }


def load_profile(config: Mapping[str, Any]) -> ProfileCostModel:
    """The profile `additional_config.locality_pd` names."""
    return ProfileCostModel.from_dict(
        json.loads(Path(config["profile_path"]).read_bytes())
    )


def native_state(scheduler: DecodeCostScheduler) -> dict[str, Any]:
    """The native queues and KV headroom of a worker's `get_locality_pd_state`.

    `paused_sessions` are streaming sessions holding a slot and blocks between
    segments, `prefill_rows` are [id, computed, prompt, priority], `loading` the
    requests whose KV is on the wire, and `reads` the last `READ_LOG` finished
    reads as [sequence number, id, start, end, tokens], which time the links.
    """
    decoding = [r for r in scheduler.running if r.num_output_tokens]
    paused = scheduler.num_waiting_for_streaming_input
    prefilling = [r for r in scheduler.running if r.is_prefill_chunk] + [
        r
        for queue in (scheduler.waiting, scheduler.skipped_waiting)
        for r in queue
        if not scheduler._is_blocked_waiting_status(r.status)
    ]
    budget = scheduler._tpot
    # In the ledger's blocks: a sequence takes one per token group.
    groups = kv_pool_layout(scheduler.kv_cache_config)["token_groups"]
    return {
        "running": len(scheduler.running),
        "waiting": len(scheduler.waiting)
        + len(scheduler.skipped_waiting)
        + len(scheduler.async_load_waiting)
        - paused,
        "paused_sessions": paused,
        "preempted_total": scheduler._preempted_total,
        "kv_cache_usage": scheduler.kv_cache_manager.usage,
        "free_blocks": scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        // groups,
        "inflight_reserved_blocks": scheduler._inflight_prefill_reserved_blocks()
        // groups,
        "request_ids": list(scheduler.requests),
        "loading": [
            r.request_id
            for r in scheduler.async_load_waiting
            if r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            and r.request_id not in scheduler.finished_recving_kv_req_ids
        ],
        "reads": list(scheduler._reads),
        "decode_requests": len(decoding),
        "decode_context_tokens": sum(r.num_computed_tokens for r in decoding),
        "decode_s_per_token": scheduler._decode_s_per_token,
        "decode_query_per_request": scheduler._decode_query_per_request,
        "cost_classes": dict(scheduler._cost_classes),
        "accept_per_row": scheduler._accept_per_row,
        "prefill_rows": [
            [r.request_id, r.num_computed_tokens, r.num_prompt_tokens, r.priority]
            for r in prefilling
        ],
        "tpot_budget": None
        if budget is None
        else budget.state(scheduler._decode_clock()),
    }


class DecodeCostScheduler(Scheduler):
    """A `Scheduler` that measures what its steps cost and decides nothing.

    Each step is timed from its schedule to the end of its update and averaged
    into its `step_class`'s means; each remote KV read, from the schedule that
    starts it to the update that reports it finished.
    """

    _decode_clock = staticmethod(time.monotonic)
    _decode_s_per_token: float | None = None
    _decode_query_per_request: float | None = None
    _accept_per_row: float | None = None
    _step_started: float | None = None
    _step_reading = False
    # The rows, seconds and end of the step last updated.
    _last_step: tuple[list[tuple[int, int, bool]], float, float] | None = None
    _decode_query: float = 0.0
    _preempted_total = 0
    _tpot: TpotBudget | None = None
    _reads_done = 0

    def __init__(self, *args, **kwargs):
        self._cost_classes: dict[str, dict[str, Any]] = {}
        self._step_rows: list[tuple[int, int, bool]] = []
        # Reads on the wire, (start, tokens) by request id; done ones as `reads`.
        self._reading: dict[str, tuple[float, int]] = {}
        self._reads: deque[list[Any]] = deque(maxlen=READ_LOG)
        super().__init__(*args, **kwargs)

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        self._preempted_total += 1
        super()._preempt_request(request, timestamp)

    def schedule(self, *args, **kwargs):
        started = self._decode_clock()
        output = super().schedule(*args, **kwargs)
        self._take_step_rows(output)
        self._step_started = started if self._step_rows else None
        self._step_reading = any(
            r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            for r in self.async_load_waiting
        )
        for r in self.async_load_waiting:
            if (
                r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
                and r.request_id not in self._reading
                and r.request_id not in self.finished_recving_kv_req_ids
            ):
                stats = r.prefill_stats
                tokens = 0 if stats is None else stats.num_external_cached_tokens
                self._reading[r.request_id] = (started, tokens)
        return output

    def _take_step_rows(self, output) -> None:
        """Record `output`'s rows, and a pure decode step's query tokens per row."""
        rows = []
        for rid, n in output.num_scheduled_tokens.items():
            request = self.requests[rid]
            context = request.num_computed_tokens - n
            prefills = context < request.num_tokens - 1 or not request.output_token_ids
            rows.append((n, context, prefills))
        self._step_rows = rows
        if rows and step_class(rows) == STEP_DEC:
            self._decode_query = output.total_num_scheduled_tokens / len(rows)

    def update_from_output(self, scheduler_output, model_runner_output):
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        self._last_step = None
        if self._step_started is None and not self._reading:
            return outputs
        ended = self._decode_clock()
        if self._reading:
            self._reads_ended(ended)
        if self._step_started is None:
            return outputs
        seconds = ended - self._step_started
        rows = self._step_rows
        kind = step_class(rows)
        self._last_step = (rows, seconds, ended)
        self._step_started = None
        means = self._cost_classes.get(kind)
        if means is None:
            if self._step_reading and seconds > DECODE_STALL_S:
                return outputs
        elif seconds > max(
            DECODE_STALL_S, DECODE_STALL_FACTOR * self._expected_s(rows, means)
        ):
            return outputs
        self._cost_classes[kind] = _class_means(means, step_features(rows), seconds)
        tokens = sum(
            len(output.new_token_ids)
            for batch in outputs.values()
            for output in batch.outputs
        )
        if kind == STEP_DEC and tokens:
            self._decode_s_per_token = _averaged(
                self._decode_s_per_token, seconds / tokens
            )
            self._decode_query_per_request = _averaged(
                self._decode_query_per_request, self._decode_query
            )
            self._accept_per_row = _averaged(self._accept_per_row, tokens / len(rows))
        return outputs

    def _reads_ended(self, now: float) -> None:
        """Log the reads this update finished at `now`; a failed read times no link."""
        for rid in [
            rid
            for rid in self._reading
            if rid in self.finished_recving_kv_req_ids or rid not in self.requests
        ]:
            start, tokens = self._reading.pop(rid)
            if (
                rid in self.finished_recving_kv_req_ids
                and rid not in self.failed_recving_kv_req_ids
            ):
                self._reads_done += 1
                self._reads.append([self._reads_done, rid, start, now, tokens])

    def _cost_step(self) -> StepModel | None:
        """The fitted step model this scheduler prices steps with, if any."""
        return None

    def _expected_s(
        self, rows: list[tuple[int, int, bool]], means: dict[str, Any]
    ) -> float:
        """What these rows should take: their price at the class ratio, else its mean.

        Priced, since a chunk's cost grows with its context far past the mean.
        """
        step = self._cost_step()
        if step is None:
            return means["t"]
        return calibrated_ratio(means, step) * step.price_s(step_features(rows))

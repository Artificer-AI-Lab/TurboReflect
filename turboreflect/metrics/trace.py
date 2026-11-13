# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors
"""Per-step and per-request engine records around the unchanged scheduler.

Always on, and read in the engine core process, not on the frontend's event loop.
Step rows go to `STEPS_FILE` every `FLUSH_EVERY` steps, when no traced request is
left and at shutdown; a request's row goes to `REQUESTS_FILE` when it finishes, under
vLLM's internal id. Every `t*` field is the engine core's `time.monotonic()`, the
clock `RequestStateStats` uses.
"""

import json
import os
import time
from pathlib import Path

from turboreflect.adaptive_draft.draft_length import AdaptiveDraftMixin
from turboreflect.context_reuse.scheduler import (
    ContextReuseScheduler,
    ReserveOnArrivalScheduler,
)
from vllm.v1.core.sched.scheduler import Scheduler

TRACE_DIR_ENV = "TURBOREFLECT_ENGINE_TRACE_DIR"
STEPS_FILE = "engine_steps_core.jsonl"
REQUESTS_FILE = "engine_requests_core.jsonl"
FLUSH_EVERY = 500

# One entry of a step row's `reqs`.
STEP_REQUEST_FIELDS = (
    "request_id",
    "scheduled_tokens",
    "is_prefill",
    "context_tokens",
    "emitted_tokens",
)


class EngineTrace:
    """Record steps and requests without changing a scheduling decision.

    Mixed in ahead of a `Scheduler`. Needs sync scheduling, so that the request
    state read here belongs to the step just run.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.scheduler_config.async_scheduling:
            raise RuntimeError(
                "the engine trace needs async_scheduling=false: under async "
                "scheduling a step's output lands after the next schedule"
            )
        directory = os.environ.get(TRACE_DIR_ENV)
        if not directory:
            raise RuntimeError(
                f"{TRACE_DIR_ENV} is not set; the driver sets it to the run "
                "directory before building the engine"
            )
        spec = self.vllm_config.speculative_config
        self._start_trace(Path(directory), None if spec is None else spec.method)

    def _start_trace(self, directory, spec_method):
        directory.mkdir(parents=True, exist_ok=True)
        header = {"kind": "header", "spec_method": spec_method}
        self._steps_file = (directory / STEPS_FILE).open("a", encoding="utf-8")
        self._requests_file = (directory / REQUESTS_FILE).open("a", encoding="utf-8")
        steps_header = {**header, "request_fields": STEP_REQUEST_FIELDS}
        for handle, line in (
            (self._steps_file, steps_header),
            (self._requests_file, header),
        ):
            handle.write(json.dumps(line) + "\n")
            handle.flush()
        self._trace_rows = {}
        self._trace_step = 0
        self._step_lines = []
        self._scheduled = None

    def schedule(self, *args, **kwargs):
        start = time.monotonic()
        output = super().schedule(*args, **kwargs)
        end = time.monotonic()
        reqs = {}
        for rid, n in output.num_scheduled_tokens.items():
            req = self.requests[rid]
            context = req.num_computed_tokens - n
            prefill = context < req.num_tokens - 1 or not req.output_token_ids
            reqs[rid] = (context, prefill)
        self._scheduled = {"t_schedule": [start, end], "reqs": reqs}
        return output

    def _request_row(self, rid):
        row = self._trace_rows.get(rid)
        if row is None:
            row = self._trace_rows[rid] = {
                "request_id": rid,
                "verification_steps": 0,
                "proposed_tokens": 0,
                "accepted_committed_tokens": 0,
            }
        return row

    def update_from_output(self, scheduler_output, model_runner_output):
        start = time.monotonic()
        self._trace_step += 1
        scheduled = self._scheduled or {"reqs": {}}
        self._scheduled = None
        snapshots = []
        for rid, n in scheduler_output.num_scheduled_tokens.items():
            req = self.requests.get(rid)
            if req is None or req.is_finished():
                continue
            idx = model_runner_output.req_id_to_index[rid]
            raw = (
                len(model_runner_output.sampled_token_ids[idx])
                if model_runner_output.sampled_token_ids
                else 0
            )
            drafts = len(scheduler_output.scheduled_spec_decode_tokens.get(rid, []))
            invalid = (scheduler_output.num_invalid_spec_tokens or {}).get(rid, 0)
            snapshots.append(
                (rid, req, n, len(req.output_token_ids), raw, drafts, invalid)
            )
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        end = time.monotonic()
        step_reqs = []
        for rid, req, scheduled_n, before, raw, drafts, invalid in snapshots:
            emitted = len(req.output_token_ids) - before
            row = self._request_row(rid)
            row["verification_steps"] += int(drafts > 0)
            row["proposed_tokens"] += drafts - invalid
            if drafts:
                row["accepted_committed_tokens"] += min(
                    max(raw - self.num_sampled_tokens_per_step, 0), emitted
                )
            context, prefill = scheduled["reqs"].get(rid, (None, before == 0))
            step_reqs.append([rid, scheduled_n, prefill, context, emitted])
            if req.is_finished():
                self._write_request(rid)
        if step_reqs:
            self._step_lines.append(
                json.dumps(
                    {
                        "step": self._trace_step,
                        "t_schedule": scheduled.get("t_schedule"),
                        "t_update": [start, end],
                        "reqs": step_reqs,
                    }
                )
            )
        if len(self._step_lines) >= FLUSH_EVERY or not self._trace_rows:
            self._flush_steps()
        return outputs

    def _write_request(self, rid):
        self._requests_file.write(json.dumps(self._trace_rows.pop(rid)) + "\n")
        self._requests_file.flush()

    def _flush_steps(self):
        if not self._step_lines:
            return
        self._step_lines.append("")
        self._steps_file.write("\n".join(self._step_lines))
        self._steps_file.flush()
        self._step_lines.clear()

    def shutdown(self):
        if not self._steps_file.closed:
            self._flush_steps()
            for rid in list(self._trace_rows):
                self._write_request(rid)
            self._steps_file.close()
            self._requests_file.close()
        super().shutdown()


class SpecTraceScheduler(EngineTrace, Scheduler):
    """vLLM's scheduler, traced."""


class ReserveOnArrivalTraceScheduler(EngineTrace, ReserveOnArrivalScheduler):
    """Reserve-on-arrival, traced."""


class ContextReuseTraceScheduler(EngineTrace, ContextReuseScheduler):
    """The context reuse scheduler, traced."""


class AdaptiveDraftTraceScheduler(EngineTrace, AdaptiveDraftMixin, Scheduler):
    """vLLM's scheduler with adaptive draft length on suffix drafts, traced."""


class ContextReuseAdaptiveDraftTraceScheduler(
    EngineTrace, AdaptiveDraftMixin, ContextReuseScheduler
):
    """The context reuse scheduler with adaptive draft length, traced."""

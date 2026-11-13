# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""KV use and per-request transfers, from inside the engine.

`kv_cache_usage` and the NIXL telemetry appear only on the scheduler stats each
iteration hands a stat logger. Only this process knows when a request began, so each
request's transfer span is opened and closed here and travels back on its output.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from turboreflect.metrics.metrics import (
    TransferSpans,
    TransferWindow,
    requests_in_flight,
)
from vllm.v1.metrics.loggers import StatLoggerBase


@dataclass
class StepTotals:
    """Steps and KV use over some span of a run."""

    role: str = ""
    steps: int = 0
    peak_kv_cache_usage: float = 0.0
    # KV usage integrated over time, each step's state weighted until the next;
    # intervals with no request running or waiting are skipped.
    kv_usage_s: float = 0.0

    def observe(self, usage: float) -> None:
        self.steps += 1
        self.peak_kv_cache_usage = max(self.peak_kv_cache_usage, usage)

    def integrate(self, running: int, waiting: int, usage: float, dt: float) -> None:
        """Weight one held state by how long it held."""
        if dt <= 0 or (running == 0 and waiting == 0):
            return
        self.kv_usage_s += usage * dt

    def as_dict(self) -> dict[str, Any]:
        """These totals for the summary, labelled only when there is a side."""
        payload = asdict(self)
        if not self.role:
            del payload["role"]
        return payload


class StepLogger(StatLoggerBase):
    """The totals the summary needs, and each request's transfers.

    vLLM builds stat loggers from a factory, so `as_factory` hands back this
    instance and the driver keeps a handle on it.
    """

    def __init__(
        self, vllm_config: Any = None, engine_index: int = 0, role: str = ""
    ) -> None:
        self.vllm_config = vllm_config
        self.engine_index = engine_index
        self.role = role
        self.run = StepTotals(role=role)
        self.window = StepTotals(role=role)
        self.spans = TransferSpans()
        self._last_step_t: float | None = None
        # (running, waiting, kv usage) held since the last step; None at a window start.
        self._held: tuple[int, int, float] | None = None

    def as_factory(self) -> Callable[[Any, int], StepLogger]:
        """A `StatLoggerFactory` that yields this instance."""

        def make(vllm_config: Any, engine_index: int = 0) -> StepLogger:
            self.vllm_config = vllm_config
            self.engine_index = engine_index
            return self

        return make

    def open_window(self, label: str) -> None:
        """Start counting a fresh pass, without disturbing the run totals.

        The gap to the pass's first record is weighted by that record's state,
        not the warmup's.
        """
        self.window = StepTotals(role=self.role)
        self._held = None
        self._last_step_t = time.monotonic()

    def reset_run(self) -> None:
        """Re-base the run totals after warmup; open request spans are kept."""
        self.run = StepTotals(role=self.role)

    def open_span(self, name: str) -> None:
        """Start counting one request's transfers."""
        self.spans.start(name)

    def close_span(self, name: str) -> TransferWindow | None:
        """What the connector moved while one request was open, or `None`."""
        return self.spans.finish(name)

    def record(
        self,
        scheduler_stats: Any,
        iteration_stats: Any,
        mm_cache_stats: Any = None,
        engine_idx: int = 0,
    ) -> None:
        if scheduler_stats is None:
            return
        running = scheduler_stats.num_running_reqs
        waiting = scheduler_stats.num_waiting_reqs
        usage = scheduler_stats.kv_cache_usage
        now = time.monotonic()
        if self._last_step_t is not None:
            held = self._held or (running, waiting, usage)
            dt = now - self._last_step_t
            self.run.integrate(*held, dt)
            self.window.integrate(*held, dt)
        self._held = (running, waiting, usage)
        self.run.observe(usage)
        self.window.observe(usage)
        self.spans.observe(
            scheduler_stats.kv_connector_stats, requests_in_flight(scheduler_stats)
        )
        self._last_step_t = now

    def log_engine_initialized(self) -> None:
        pass

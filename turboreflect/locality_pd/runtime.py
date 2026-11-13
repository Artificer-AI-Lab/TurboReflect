# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""Locality-aware P/D runtime settings and live latency windows, no CUDA imports."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from turboreflect.locality_pd.policy import (
    CostPrior,
    Observation,
)


@dataclass(frozen=True)
class LocalityPDSettings:
    """Locality-aware P/D settings; the flags are `dispatch.Dispatcher`'s arguments.

    Attributes:
        prior: The fit prices start from, which the engines load too.
        tpot_delta: Each engine's relative decode-latency budget delta.
    """

    prior: CostPrior
    tpot_delta: float
    tail_programs_per_engine: int = 2
    drain_on_p: bool = True
    role_flex: bool = True


@dataclass
class PolicyRuntime:
    name: str
    settings: LocalityPDSettings
    # `admission.StageShape` by declared later stage, measured once a pass.
    shapes: dict[Any, Any] = field(default_factory=dict, init=False, repr=False)


class DecodeWindowStats:
    """Keep individual TTFT/ITL samples from IterationStats for `window_s` seconds.

    Timed on the callback's monotonic clock, not the wall-clock iteration_timestamp.
    """

    def __init__(self, window_s: float = 10.0):
        self.window_s = window_s
        self.ttft: deque[Observation] = deque()
        self.itl: deque[Observation] = deque()

    def clear(self) -> None:
        self.ttft.clear()
        self.itl.clear()

    def record(self, iteration_stats: Any) -> None:
        if iteration_stats is None:
            return
        now = time.monotonic()
        for name, source in (
            ("ttft", "time_to_first_tokens_iter"),
            ("itl", "inter_token_latencies_iter"),
        ):
            samples = getattr(self, name)
            samples.extend(
                Observation(now, value) for value in getattr(iteration_stats, source)
            )
        self._expire(now)

    def _expire(self, now_s: float) -> None:
        for samples in (self.ttft, self.itl):
            while samples and samples[0].at_s < now_s - self.window_s:
                samples.popleft()

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        self._expire(now)
        return {
            name: [
                {"at_s": sample.at_s, "value_s": sample.value_s}
                for sample in getattr(self, name)
            ]
            for name in ("ttft", "itl")
        }

    def as_factory(self):
        """The stat logger class, importing vLLM only after the worker pins its GPU."""
        from vllm.v1.metrics.loggers import StatLoggerBase

        window = self

        class Logger(StatLoggerBase):
            def __init__(self, vllm_config, engine_index=0):
                pass

            def record(
                self,
                scheduler_stats,
                iteration_stats,
                mm_cache_stats=None,
                engine_idx=0,
            ):
                window.record(iteration_stats)

            def log_engine_initialized(self):
                pass

        return Logger

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""Per-stage timing and reuse counts, taken from the engine's own request stats.

Every duration is a difference of engine-core monotonic timestamps (`queued_ts`,
`scheduled_ts`, `first_token_ts`, `last_token_ts`) or the frontend's own
`first_token_latency`. A stage split across P and D records which engine timed it,
and intervals spanning both use the driver's clock. The handoff,
`handoff_to_first_token_wall_s`, is not a transfer time: it holds the driver's
submission, D's queueing, the wait for the KV, the last prompt token's forward,
sampling, and the trip back. `prefill_s` spans every chunked-prefill step, so under
concurrency it is a residency time, not a compute time.

Missing engine stats are errors. Connector telemetry carries no request id, so its
per-request fields are `None` unless the span held one request.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

# The `engine` label of a stage one engine ran end to end.
ONE_ENGINE = ""

# The label of a stage prefilled on P and decoded on D.
SPLIT_ENGINE = "pd"


def split_engine(decode_engine: str) -> str:
    """The label of a split stage decoded on this engine: `pd:<name>`."""
    return f"{SPLIT_ENGINE}:{decode_engine}"


# The P/D fields of a stage's record, written (nulls included) only when split.
SPLIT_FIELDS = (
    "engine",
    "p_prefill_s",
    "d_decode_s_per_token",
    "handoff_to_first_token_wall_s",
    "nixl_xfer_s",
    "nixl_bytes",
    "nixl_descriptors",
    "d_external_tokens",
    "remote_block_ids_len",
    "d_ttft_s",
    "ttft_wall_s",
    "ttft_engine_s",
    "d_e2e_s",
    "p_cached_tokens",
    "p_pulled_tokens",
)

# `NixlKVConnectorStats.data` lists and the field each sums into (seconds).
_TRANSFER_SOURCES = (
    ("transfer_duration", "xfer_s"),
    ("bytes_transferred", "bytes_transferred"),
    ("num_descriptors", "descriptors"),
)


class MetricsError(Exception):
    """The engine did not report the per-request stats a run collects."""


def requests_in_flight(scheduler_stats: Any) -> int:
    """Running, waiting and skipped-waiting (remote KV not landed) requests."""
    return (
        scheduler_stats.num_running_reqs
        + scheduler_stats.num_waiting_reqs
        + scheduler_stats.num_skipped_waiting_reqs
    )


def count_remote_blocks(remote_block_ids: Any) -> int:
    """Blocks P offered, summed over KV cache groups; `len` would count groups."""
    return sum(
        1 if isinstance(group, int) else len(group) for group in remote_block_ids
    )


@dataclass
class TransferWindow:
    """NIXL telemetry over one span, and how crowded that span was.

    Attributes:
        steps: Scheduler iterations fed; zero means unfed, not "no transfer".
    """

    steps: int = 0
    peak_in_flight: int = 0
    xfer_s: float = 0.0
    bytes_transferred: int = 0
    descriptors: int = 0

    def observe(
        self, data: Mapping[str, Sequence[float]] | None, in_flight: int
    ) -> None:
        """Fold one step's `SchedulerStats.kv_connector_stats` into this window."""
        self.steps += 1
        self.peak_in_flight = max(self.peak_in_flight, in_flight)
        if not data:
            return
        for source, name in _TRANSFER_SOURCES:
            total = sum(data.get(source) or ())
            setattr(self, name, getattr(self, name) + total)

    @property
    def attributable(self) -> bool:
        """Whether these numbers can be read as one request's."""
        return self.steps > 0 and self.peak_in_flight <= 1

    def per_request(self) -> tuple[float | None, int | None, int | None]:
        """`(nixl_xfer_s, nixl_bytes, nixl_descriptors)`, all None if crowded."""
        if not self.attributable:
            return None, None, None
        return self.xfer_s, int(self.bytes_transferred), int(self.descriptors)


@dataclass
class TransferSpans:
    """One open `TransferWindow` per request the engine holds, by request id."""

    windows: dict[str, TransferWindow] = field(default_factory=dict)

    def start(self, name: str) -> None:
        """Begin counting under this name, from nothing."""
        self.windows[name] = TransferWindow()

    def finish(self, name: str) -> TransferWindow | None:
        """Stop counting under this name; None for a name never opened."""
        return self.windows.pop(name, None)

    def observe(
        self, data: Mapping[str, Sequence[float]] | None, in_flight: int
    ) -> None:
        """Fold one scheduler iteration into every span still open."""
        for window in self.windows.values():
            window.observe(data, in_flight)


@dataclass(frozen=True)
class Handoff:
    """P's half of one stage, as the driver saw it.

    Fields feed the `StageMetrics` P/D fields of the same name;
    `to_first_token_wall_s` becomes `handoff_to_first_token_wall_s`, and
    `transfer` is D's connector telemetry over the stage's span.
    """

    engine: str = SPLIT_ENGINE
    p_prefill_s: float | None = None
    to_first_token_wall_s: float | None = None
    remote_block_ids_len: int | None = None
    transfer: TransferWindow | None = None
    p_queued_ts: float | None = None
    ttft_wall_s: float | None = None
    p_cached_tokens: int | None = None
    p_pulled_tokens: int | None = None

    @classmethod
    def from_prefill(
        cls,
        out: Any,
        *,
        to_first_token_wall_s: float,
        ttft_wall_s: float,
        transfer: TransferWindow | None = None,
        engine: str = SPLIT_ENGINE,
    ) -> Handoff:
        """Read P's side of one stage off what P returned.

        Args:
            out: P's `RequestOutput`, or its projection across the worker pipe;
                only attribute names are read.

        Raises:
            MetricsError: If P reported no stats or no `kv_transfer_params`,
                without which D would prefill the prompt itself.
        """
        stats = out.metrics
        if stats is None:
            raise MetricsError(
                "the prefill engine returned no metrics -- it was built with "
                "log stats disabled, so nothing in this run says what P cost"
            )
        params = out.kv_transfer_params
        if not params:
            raise MetricsError(
                "the prefill engine returned no kv_transfer_params, so there "
                "is nothing for the decode engine to pull and it would prefill "
                "the prompt itself and answer correctly"
            )
        return cls(
            engine=engine,
            p_prefill_s=stats.first_token_ts - stats.scheduled_ts,
            to_first_token_wall_s=to_first_token_wall_s,
            remote_block_ids_len=count_remote_blocks(params["remote_block_ids"]),
            transfer=transfer,
            p_queued_ts=stats.queued_ts,
            ttft_wall_s=ttft_wall_s,
            p_cached_tokens=out.num_cached_tokens,
            p_pulled_tokens=getattr(out, "num_external_cached_tokens", None),
        )


@dataclass(frozen=True)
class StageMetrics:
    """What one stage cost. A None field was not measured."""

    stage: str
    request_id: str
    prompt_tokens: int
    cached_tokens: int
    cache_creation_tokens: int
    computed_tokens: int
    gen_tokens: int
    ttft_s: float
    queue_s: float
    prefill_s: float
    prefill_tokens_per_s: float
    decode_s_per_token: float
    wall_s: float
    finish_reason: str
    # Raw engine-core timestamps, comparable across one engine's requests.
    queued_ts: float = 0.0
    scheduled_ts: float = 0.0
    first_token_ts: float = 0.0
    last_token_ts: float = 0.0
    # Driver `perf_counter` from the previous stage's output to this stage's
    # submit; no running request protects the program's blocks meanwhile.
    host_gap_s: float | None = None
    # The part of that gap spent building and tokenizing the prompt.
    render_s: float | None = None
    # The part of that gap blocked in admission before submission; on a first
    # stage, the wait before the program started (outside `e2e_seconds`).
    admission_wait_s: float | None = None
    # Split stage: time queued in `Dispatcher.pump` before `wall_s` starts.
    gate_wait_s: float | None = None
    # The part of `cached_tokens` a KV connector (the P/D NIXL transfer) supplied.
    external_cached_tokens: int | None = None
    # Which engine's clock the timestamps came from; empty for one engine.
    engine: str = ONE_ENGINE
    # On a split stage the unprefixed timings are D's: `prefill_s` is D's wait for
    # the KV plus one recomputed token, and `computed_tokens` is zero because the
    # scheduler records the local and external hits as the whole prompt.
    p_prefill_s: float | None = None
    d_decode_s_per_token: float | None = None
    handoff_to_first_token_wall_s: float | None = None
    # Attributable only while one request was in flight.
    nixl_xfer_s: float | None = None
    nixl_bytes: int | None = None
    nixl_descriptors: int | None = None
    # Tokens D received over NIXL: the tail past D's local hit.
    d_external_tokens: int | None = None
    remote_block_ids_len: int | None = None
    # On a split stage `ttft_s` is `ttft_wall_s` (driver clock, P being sent the
    # request to D's first token); D's own `d_ttft_s` misses P's queue and prefill.
    # `ttft_engine_s` is D's `first_token_ts` less P's `queued_ts`, valid because
    # both engine cores stamp `time.monotonic()` on one host. `e2e_s` replaces
    # D's own `d_e2e_s` for the same reason.
    d_ttft_s: float | None = None
    ttft_wall_s: float | None = None
    ttft_engine_s: float | None = None
    d_e2e_s: float | None = None
    # P's `num_cached_tokens`, of which `p_pulled_tokens` were pulled from a D.
    p_cached_tokens: int | None = None
    p_pulled_tokens: int | None = None

    @classmethod
    def from_output(
        cls,
        stage: str,
        out: Any,
        wall_s: float,
        handoff: Handoff | None = None,
        host_gap_s: float | None = None,
        render_s: float | None = None,
        admission_wait_s: float | None = None,
        gate_wait_s: float | None = None,
    ) -> StageMetrics:
        """Extract a stage's timings and reuse counts from its `RequestOutput`.

        Args:
            out: What the engine returned; on a split stage, D's answer.
            wall_s: The driver's own clock across the whole stage.
            handoff: P's half, when two engines ran this stage.

        Raises:
            MetricsError: If the engine reported no stats or no reuse counts.
        """
        stats = out.metrics
        if stats is None:
            raise MetricsError(
                f"stage {stage!r} came back without metrics -- the engine was "
                "built with log stats disabled"
            )
        cached = out.num_cached_tokens
        created = out.num_cache_creation_tokens
        for name, value in (
            ("num_cached_tokens", cached),
            ("num_cache_creation_tokens", created),
        ):
            if value is None:
                raise MetricsError(
                    f"stage {stage!r} reported {name}=None -- that is a broken "
                    "collection path, not a hit count of zero"
                )
        completion = out.outputs[0]
        n_gen = stats.num_generation_tokens or len(completion.token_ids)
        # An output that crossed the P/D pipe carries the prompt's length only.
        prompt_len = getattr(out, "prompt_len", None)
        prompt_tokens = (
            len(out.prompt_token_ids or ()) if prompt_len is None else prompt_len
        )
        computed = prompt_tokens - cached
        # Decode latency per token: the decode span over the gaps between tokens.
        decode_span = stats.last_token_ts - stats.first_token_ts
        decode_s_per_token = decode_span / max(1, n_gen - 1)
        prefill_s = stats.first_token_ts - stats.scheduled_ts
        external = getattr(out, "num_external_cached_tokens", None)
        ttft_s = stats.first_token_latency
        if handoff is not None:
            ttft_s = handoff.ttft_wall_s
        return cls(
            stage=stage,
            request_id=out.request_id,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached,
            cache_creation_tokens=created,
            computed_tokens=computed,
            gen_tokens=n_gen,
            ttft_s=ttft_s,
            queue_s=stats.scheduled_ts - stats.queued_ts,
            prefill_s=prefill_s,
            prefill_tokens_per_s=computed / prefill_s if prefill_s > 0 else 0.0,
            decode_s_per_token=decode_s_per_token,
            wall_s=wall_s,
            finish_reason=completion.finish_reason or "unknown",
            queued_ts=stats.queued_ts,
            scheduled_ts=stats.scheduled_ts,
            first_token_ts=stats.first_token_ts,
            last_token_ts=stats.last_token_ts,
            host_gap_s=host_gap_s,
            render_s=render_s,
            admission_wait_s=admission_wait_s,
            gate_wait_s=gate_wait_s,
            external_cached_tokens=external,
            **_split_fields(handoff, stats, external, decode_s_per_token),
        )

    @property
    def hit_ratio(self) -> float:
        """Prefix hit ratio: the fraction of this prompt served from cache."""
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def e2e_s(self) -> float:
        """From the request's submission to its last token.

        On a split stage, `ttft_wall_s` plus D's own decode span, so nothing is
        subtracted across engine clocks.
        """
        if self.ttft_wall_s is None:
            return self.last_token_ts - self.queued_ts
        return self.ttft_wall_s + (self.last_token_ts - self.first_token_ts)

    def as_dict(self) -> dict[str, Any]:
        """This stage's result row, minus unmeasured gaps and unused `SPLIT_FIELDS`."""
        payload = asdict(self)
        if self.host_gap_s is None:
            del payload["host_gap_s"]
        if self.render_s is None:
            del payload["render_s"]
        if self.admission_wait_s is None:
            del payload["admission_wait_s"]
        if self.gate_wait_s is None:
            del payload["gate_wait_s"]
        if self.external_cached_tokens is None:
            del payload["external_cached_tokens"]
        if self.engine:
            return payload
        for name in SPLIT_FIELDS:
            del payload[name]
        return payload

    def line(self) -> str:
        row = (
            f"{self.stage:<10s} prompt {self.prompt_tokens:>7d}  "
            f"cached {self.cached_tokens:>7d}  "
            f"hit {self.hit_ratio:6.1%}  gen {self.gen_tokens:>5d}  "
            f"ttft {self.ttft_s:7.3f}s  prefill {self.prefill_s:7.3f}s  "
            f"{self.prefill_tokens_per_s:>8.0f} tok/s  "
            f"decode {self.decode_s_per_token * 1000:6.2f}ms/tok  "
            f"[{self.finish_reason}]"
        )
        return f"{row}  {self.split_line()}" if self.engine else row

    def split_line(self) -> str:
        """The P/D half of a stage's log row; no transfer time is "not measured"."""
        parts = [f"engine {self.engine}"]
        if self.p_prefill_s is not None:
            parts.append(f"p_prefill {self.p_prefill_s:7.3f}s")
        if self.handoff_to_first_token_wall_s is not None:
            parts.append(f"handoff {self.handoff_to_first_token_wall_s:7.3f}s")
        if self.d_external_tokens is not None:
            parts.append(f"external {self.d_external_tokens:>7d}")
        parts.append(
            f"nixl {self.nixl_xfer_s:7.3f}s"
            if self.nixl_xfer_s is not None
            else "nixl not measured"
        )
        return "  ".join(parts)


def _split_fields(
    handoff: Handoff | None,
    stats: Any,
    external: int | None,
    decode_s_per_token: float,
) -> dict[str, Any]:
    """The P/D fields of a stage's record, or `{}` without a handoff."""
    if handoff is None:
        return {}
    xfer_s, xfer_bytes, descriptors = (
        handoff.transfer.per_request()
        if handoff.transfer is not None
        else (None, None, None)
    )
    ttft_engine_s = (
        stats.first_token_ts - handoff.p_queued_ts
        if handoff.p_queued_ts is not None
        else None
    )
    return {
        "engine": handoff.engine,
        "p_prefill_s": handoff.p_prefill_s,
        "d_decode_s_per_token": decode_s_per_token,
        "handoff_to_first_token_wall_s": handoff.to_first_token_wall_s,
        "nixl_xfer_s": xfer_s,
        "nixl_bytes": xfer_bytes,
        "nixl_descriptors": descriptors,
        "d_external_tokens": external,
        "remote_block_ids_len": handoff.remote_block_ids_len,
        "d_ttft_s": stats.first_token_latency,
        "ttft_wall_s": handoff.ttft_wall_s,
        "ttft_engine_s": ttft_engine_s,
        "d_e2e_s": stats.last_token_ts - stats.queued_ts,
        "p_cached_tokens": handoff.p_cached_tokens,
        "p_pulled_tokens": handoff.p_pulled_tokens,
    }

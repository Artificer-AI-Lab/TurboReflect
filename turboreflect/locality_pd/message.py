# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""What crosses the pipe between the driver and the engine workers.

The driver, parent of every worker, must not import vLLM before they pin their
devices, so a `RequestOutput` cannot cross. Each worker projects it onto the
records here under the names `StageMetrics.from_output` reads, so a stage run on
two engines is measured by the code that measures one. Message kinds are shared
strings, not objects: a mistyped kind is a message silently never handled.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any

import regex as re

# Driver -> worker. `generate` carries `request_id`, `prompt_token_ids`,
# `cache_salt`, `sampling`, `kv_transfer_params`, `priority` (an `add_request`
# argument), `stream_first_token` and, with clips, `MEDIA`. `abort` carries a
# `request_id`, `reject` that and the transfer params. `release_hold` names a
# finished request whose blocks a D holds for a pull no stage will make, and is
# not answered (`EngineCore.release_kv_hold`).
GENERATE = "generate"
ABORT = "abort"
REJECT = "reject"
RESET_PREFIX_CACHE = "reset_prefix_cache"
OPEN_WINDOW = "open_window"
RESET_RUN = "reset_run"
STEPS = "steps"
RELEASE_HOLD = "release_hold"
SHUTDOWN = "shutdown"

TO_WORKER = (
    GENERATE,
    ABORT,
    REJECT,
    RESET_PREFIX_CACHE,
    OPEN_WINDOW,
    RESET_RUN,
    STEPS,
    RELEASE_HOLD,
    SHUTDOWN,
)

# Worker -> driver. `admitted` tells a request D never admitted from one it holds
# open awaiting KV, which need opposite cleanups; `headroom` is pushed unasked.
READY = "ready"
FAILED = "failed"
ADMITTED = "admitted"
FIRST_TOKEN = "first_token"
OUTPUT = "output"
ERROR = "error"
RESET = "reset"
STEP_TOTALS = "step_totals"
HEADROOM = "headroom"

TO_DRIVER = (
    READY,
    FAILED,
    ADMITTED,
    FIRST_TOKEN,
    OUTPUT,
    ERROR,
    RESET,
    STEP_TOTALS,
    HEADROOM,
)

# A worker reads its engine core every poll and pushes `headroom` when that
# changed, else a refresh after its last push.
HEADROOM_POLL_S = 0.05
HEADROOM_REFRESH_S = 1.0

# The `output` key for the request's `TransferWindow`, counted in the worker, whose
# stat logger alone sees transfer telemetry; a typo silently empties NIXL fields.
TRANSFER = "transfer"

# The `generate` key for its clips' paths; each worker decodes them at startup.
MEDIA = "media"


@dataclass(frozen=True)
class EngineFacts:
    """The `facts` of a worker's `ready`; `block_size` is tokens per KV block."""

    block_size: int


# The `ready` key for each sample's clip lines as the engine's processor expands
# them (`program.media.ClipLayout`), by clip paths.
CLIP_LAYOUTS = "clip_layouts"

# The key `vllm/v1/request.py` reads; under any other D prefills the whole prompt.
KV_TRANSFER_PARAMS_KEY = "kv_transfer_params"

# Set False in P's params for a D whose blocks no later stage will pull, which
# under `bidirectional_kv_xfer` it would otherwise hold for its whole TTL.
DECODER_KV_HOLD_KEY = "decoder_kv_blocks_hold"

SCHEDULER_STATE_UTILITY = "get_locality_pd_scheduler_state"
RELEASE_HOLD_UTILITY = "release_kv_hold"
PREFILL_TAG = "locality_pd_prefill"

# The engine core appends `-<8 hex>` to the driver's id
# (`vllm/v1/engine/input_processor.py`).
_INTERNAL_SUFFIX = re.compile(r"-[0-9a-f]{8}$")


def submitted_id(request_id: str) -> str:
    """The id the driver submitted, from an engine core's own request id."""
    return _INTERNAL_SUFFIX.sub("", request_id)


# `RequestOutputKind` by name: the driver has no vLLM enum; the worker resolves it.
DELTA = "delta"
FINAL_ONLY = "final_only"


@dataclass(frozen=True)
class _PipeEnd:
    error: BaseException


class PipeReader:
    """Every message on one end of a pipe, read by its own thread as it lands.

    Reading only while the event loop is free would deadlock two ends that each
    block writing a message larger than the pipe holds.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self._queue: asyncio.Queue | None = None

    async def recv(self) -> Any:
        """The next message, in the order sent.

        Raises:
            EOFError: Or whatever the read raised, on this and every later call.
        """
        if self._queue is None:
            self._queue = asyncio.Queue()
            threading.Thread(
                target=self._read,
                args=(asyncio.get_running_loop(), self._queue),
                name="pd-pipe",
                daemon=True,
            ).start()
        item = await self._queue.get()
        if isinstance(item, _PipeEnd):
            self._queue.put_nowait(item)
            raise item.error
        return item

    def _read(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        while True:
            try:
                item = self._conn.recv()
            except BaseException as exc:  # noqa: BLE001 - handed to the reader
                item = _PipeEnd(exc)
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:
                return  # the loop has closed: nobody is reading
            if isinstance(item, _PipeEnd):
                return


@dataclass(frozen=True)
class RemoteStats:
    """`RequestStateStats` as it survives the pipe.

    Timestamps are the engine core's `time.monotonic()`, comparable only on one host.
    """

    queued_ts: float
    scheduled_ts: float
    first_token_ts: float
    last_token_ts: float
    first_token_latency: float
    num_generation_tokens: int


@dataclass(frozen=True)
class RemoteCompletion:
    """`CompletionOutput` as it survives the pipe."""

    token_ids: tuple[int, ...]
    text: str
    finish_reason: str | None


@dataclass(frozen=True)
class RemoteOutput:
    """`RequestOutput` as it survives the pipe; the prompt crosses as `prompt_len`."""

    request_id: str
    prompt_len: int
    outputs: tuple[RemoteCompletion, ...]
    metrics: RemoteStats | None
    num_cached_tokens: int | None
    num_cache_creation_tokens: int | None
    # The part of `num_cached_tokens` the connector supplied.
    num_external_cached_tokens: int | None
    kv_transfer_params: dict[str, Any] | None


@dataclass(frozen=True)
class ExpandedOutput(RemoteOutput):
    """A `RemoteOutput` with the prompt the engine ran, its clips expanded.

    The driver reads the stage's expansion off it (`program.media.Expansion`).
    """

    prompt_token_ids: tuple[int, ...] = ()


def project(out: Any, expanded: bool = False) -> RemoteOutput:
    """Project a vLLM `RequestOutput` onto the records that cross the pipe.

    `expanded` marks a request with clips, whose expanded prompt crosses too.
    """
    stats = out.metrics
    record = ExpandedOutput if expanded else RemoteOutput
    extra = {"prompt_token_ids": tuple(out.prompt_token_ids)} if expanded else {}
    return record(
        request_id=out.request_id,
        prompt_len=len(out.prompt_token_ids or ()),
        outputs=tuple(
            RemoteCompletion(
                token_ids=tuple(completion.token_ids),
                text=completion.text,
                finish_reason=completion.finish_reason,
            )
            for completion in out.outputs
        ),
        metrics=None
        if stats is None
        else RemoteStats(
            queued_ts=stats.queued_ts,
            scheduled_ts=stats.scheduled_ts,
            first_token_ts=stats.first_token_ts,
            last_token_ts=stats.last_token_ts,
            first_token_latency=stats.first_token_latency,
            num_generation_tokens=stats.num_generation_tokens,
        ),
        num_cached_tokens=out.num_cached_tokens,
        num_cache_creation_tokens=out.num_cache_creation_tokens,
        num_external_cached_tokens=out.num_external_cached_tokens,
        # Carried as P returned it: `on_new_request` silently skips a dict missing
        # a field, and D then never renews P's lease.
        kv_transfer_params=out.kv_transfer_params,
        **extra,
    )

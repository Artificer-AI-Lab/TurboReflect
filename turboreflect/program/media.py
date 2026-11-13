# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""A task's video clips: decoded once, carried unchanged by every stage.

The driver submits ids with one placeholder per clip and the clip beside them
(`multi_modal_data`); vLLM expands each placeholder on its side. Hence:

1. Every stage carries the clips. A repeated placeholder without them runs as text:
   its blocks lack the clip's key and everything from the clip on is prefilled again,
   silently. With the same data, uuids and processor kwargs, a later stage's prefix hit
   runs through the clip and skips the encoder.
2. The engine runs more ids than are submitted. Matched prefix and append count engine
   ids (`Expansion`, `ClipLayout`); expanded ids are never submitted, since vLLM would
   expand them again.
3. A clip's uuid digests its bytes, its decode and processor settings and, under
   program isolation, the cache salt: programs share no processor or encoder cache
   entry, and a program's stages share all of them.

Kimi K2.6 (selected by checkpoint architecture) takes a clip as the `vision_chunk`
items its checkpoint's `split_video_chunks` cuts, each one `<|media_pad|>` after its
first frame's time. vLLM does not cut them for this model, so this module does.

A split run sends clip paths, not frames (`clip_ref`), and each worker decodes every
sample's clips at startup (`clip_table`), so P and D derive the same uuids.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import fcntl
import functools
import hashlib
import json
import os
import time
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from turboreflect.config import (
    MULTIMODAL_FIELDS,
    RunConfig,
    checkpoint_architectures,
)
from turboreflect.locality_pd.cache_hits import lcp
from turboreflect.program.task import Task, file_sha256, load_task, video_paths

VIDEO_MODALITY = "video"
# Checkpoints whose clips go in as `vision_chunk` items (module docstring).
KIMI_ARCHITECTURES = ("KimiK25ForConditionalGeneration",)
KIMI_MODALITY = "vision_chunk"
# The text Kimi's processor renders per chunk (`make_chunk_prompt`).
KIMI_CHUNK_PROMPT = (
    "{timestamp}<|media_begin|>video<|media_content|><|media_pad|><|media_end|>"
)
# The `media_proc_cfg` entries that decide a chunk's frames and its text.
KIMI_SAMPLING = ("sample_fps", "temporal_merge_kernel_size", "timestamp_mode")
# A node-local directory a node's decoding processes share (`stored_clip`).
FRAME_STORE_ENV = "TURBOREFLECT_FRAME_STORE"


class MediaError(Exception):
    """A clip cannot be carried the way this run needs it."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def video_placeholders(model: str, count: int) -> list[str]:
    """Each clip's placeholder text, from vLLM's model class (`get_placeholder_str`).

    Raises:
        MediaError: If no class loads for the checkpoint or it has no video placeholder.
    """
    from vllm.model_executor.models.registry import ModelRegistry

    for architecture in checkpoint_architectures(model) or ():
        model_cls = ModelRegistry._try_load_model_cls(architecture)
        if model_cls is not None:
            break
    else:
        raise MediaError(
            f"vLLM loads no model class for the architectures in {model}'s config.json"
        )
    ask = getattr(model_cls, "get_placeholder_str", None)
    try:
        lines = [ask(VIDEO_MODALITY, i) if ask else None for i in range(1, count + 1)]
    except ValueError:
        lines = [None]
    if not all(lines):
        raise MediaError(f"vLLM's {model_cls.__name__} gives no video placeholder")
    return lines


def bind(task: Task, cfg: RunConfig, media: SampleMedia | None = None) -> Task:
    """`task` with one line per clip: `media`'s text, else the model placeholder."""
    if not task.videos:
        return task
    lines = (media.placeholders if media is not None else ()) or (
        video_placeholders(cfg.model, len(task.videos))
    )
    return dataclasses.replace(task, media_text="\n".join(lines))


@dataclass(frozen=True)
class SampleMedia:
    """One sample's clips, decoded once and shared by every program over it.

    Attributes:
        data: The `multi_modal_data` every stage request carries.
        digests: Per item, its clip's bytes and settings (and a chunk's place in the
            clip); `request_fields` adds the program's salt for the uuid.
        processor_kwargs: Every request's `mm_processor_kwargs`; None for the engine's.
        modality: `video` (an item per clip) or Kimi K2.6's `vision_chunk` (per chunk).
        placeholders: Per clip, its prompt text; empty means the model's placeholder.
    """

    data: dict[str, list[Any]]
    digests: tuple[str, ...]
    processor_kwargs: dict[str, Any] | None = None
    modality: str = VIDEO_MODALITY
    placeholders: tuple[str, ...] = ()

    def request_fields(self, salt: str | None) -> dict[str, Any]:
        """The media fields of every stage request of a program under `salt`."""
        uuids = [
            digest if salt is None else _sha256(f"{digest}\nsalt:{salt}")
            for digest in self.digests
        ]
        fields: dict[str, Any] = {
            "multi_modal_data": self.data,
            "multi_modal_uuids": {self.modality: uuids},
        }
        if self.processor_kwargs:
            fields["mm_processor_kwargs"] = self.processor_kwargs
        return fields


@dataclass(frozen=True)
class Expansion:
    """The generation stage's ids through its last clip, as submitted and as run.

    The engine expands each placeholder (Kimi: each `<|media_pad|>`) and leaves every
    other id, so a later stage, which repeats these ids, runs `served` then its own.
    """

    submitted: list[int]
    served: list[int]

    @classmethod
    def from_generation(
        cls, submitted: Sequence[int], served: Sequence[int]
    ) -> Expansion:
        """Read the expansion off the generation stage's two id lists."""
        submitted, served = list(submitted), list(served)
        head = lcp(submitted, served)
        tail = min(lcp(submitted[::-1], served[::-1]), len(submitted) - head)
        return cls(submitted[: len(submitted) - tail], served[: len(served) - tail])

    def engine_ids(self, submitted: Sequence[int]) -> list[int]:
        """The ids the engine runs for a later stage submitting `submitted`."""
        return [*self.served, *submitted[len(self.submitted) :]]


@dataclass(frozen=True)
class ClipLayout:
    """A sample's clip lines (`bind`) as submitted and as the engine expands them.

    Read before any program runs (`expand_lines`), so a split run prices and cuts
    prompts by engine position and never ends a request inside the clips.
    """

    submitted: tuple[int, ...]
    served: tuple[int, ...]

    def expansion(self, prompt: Sequence[int]) -> Expansion:
        """The expansion of a prompt that carries the clip lines.

        Raises:
            MediaError: If the prompt carries them other than once.
        """
        lines, n = list(self.submitted), len(self.submitted)
        found = [
            i
            for i in range(len(prompt) - n + 1)
            if prompt[i] == lines[0] and list(prompt[i : i + n]) == lines
        ]
        if len(found) != 1:
            raise MediaError(
                f"the prompt carries its {n} clip-line ids {len(found)} times, "
                "not once, so the engine's positions cannot be predicted"
            )
        [i] = found
        return Expansion(list(prompt[: i + n]), [*prompt[:i], *self.served])


def _is_kimi(model_config: Any) -> bool:
    return bool(set(model_config.architectures or ()) & set(KIMI_ARCHITECTURES))


def clip_digest(
    path: str,
    processor_kwargs: dict[str, Any] | None,
    video_processor: str | None,
    sampling: Mapping[str, Any] | None = None,
) -> str:
    """Digest of a clip's bytes and of its decode and processor settings."""
    settings: dict[str, Any] = {
        "io": {},
        "processor": processor_kwargs or {},
        "video_processor": video_processor,
    }
    if sampling is not None:
        settings["sampling"] = dict(sampling)
    text = json.dumps(settings, sort_keys=True, default=str)
    return _sha256(f"{file_sha256(path)}\n{text}")


def decode_clip(
    path: str,
    video_processor: str | None,
    media_root: str | None,
) -> Any:
    """`(frames, metadata)` for one clip, by vLLM's loader for this processor."""
    from vllm.multimodal.media import MediaConnector

    connector = MediaConnector(
        media_io_kwargs={"video": {}},
        allowed_local_media_path=media_root or "/",
    )
    return connector.fetch_video(
        Path(path).resolve().as_uri(), video_processor=video_processor
    )


@contextlib.contextmanager
def _replacing(path: Path) -> Iterator[Any]:
    """A binary file that takes `path`'s place whole once written."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as out:
        yield out
    os.replace(tmp, path)


def stored_clip(
    store: str,
    processor_kwargs: dict[str, Any] | None,
    path: str,
    video_processor: str | None,
    media_root: str | None,
) -> Any:
    """`decode_clip`'s result, decoded once per node through the frame store `store`.

    The first process to lock a clip's digest decodes and files it; the others wait.
    Each holds the frames as a copy-on-write map sharing the page cache.
    """
    import numpy as np

    root = Path(store)
    root.mkdir(parents=True, exist_ok=True)
    digest = clip_digest(path, processor_kwargs, video_processor)
    frames_path, meta_path = root / f"{digest}.npy", root / f"{digest}.json"
    with open(root / f"{digest}.lock", "wb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        done = "loaded"
        if not meta_path.exists():
            frames, metadata = decode_clip(path, video_processor, media_root)
            text = json.dumps(metadata)
            with _replacing(frames_path) as out:
                np.save(out, frames, allow_pickle=False)
            del frames
            with _replacing(meta_path) as out:
                out.write(text.encode())
            done = "decoded"
        print(f"pid {os.getpid()} {done} {path} as {frames_path}", flush=True)
        return np.load(frames_path, mmap_mode="c"), json.loads(meta_path.read_text())


def kimi_frame_indices(total: int, fps: float, sample_fps: float) -> list[int]:
    """Frames Kimi's `split_video_chunks` takes: evenly spread, `sample_fps` at most."""
    import numpy as np

    rate = min(sample_fps, fps)
    count = max(round(total * rate / fps), 1)
    return np.linspace(0, total - 1, count).round().astype(int).tolist()


def kimi_timestamp(seconds: float, mode: str) -> str:
    """Kimi's `timestamp_as_str`, which truncates the milliseconds."""
    clock = datetime.fromtimestamp(seconds, tz=timezone.utc)
    millis = f".{int((seconds % 1) * 1000):03d}"
    if mode == "hh:mm:ss.fff":
        return clock.strftime("%H:%M:%S") + millis
    if mode == "mm:ss.fff":
        return clock.strftime("%M:%S") + millis
    if mode == "mm:ss":
        return clock.strftime("%M:%S")
    raise MediaError(f"Kimi's processor knows no timestamp_mode {mode!r}")


def kimi_chunks(
    frames: Sequence[Any],
    indices: Sequence[int],
    fps: float,
    sampling: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    """A clip's `vision_chunk` items and its text, as `split_video_chunks` cuts them.

    `frames` are the sampled PIL images, `indices` their indices in the clip. An item
    holds `temporal_merge_kernel_size` frames (the last the rest); its text is its
    first frame's time and one `<|media_pad|>`.
    """
    size = sampling["temporal_merge_kernel_size"]
    chunks: list[dict[str, Any]] = []
    text: list[str] = []
    for start in range(0, len(frames), size):
        chunks.append(
            {"type": "video_chunk", "video_chunk": list(frames[start : start + size])}
        )
        stamp = kimi_timestamp(indices[start] / fps, sampling["timestamp_mode"])
        text.append(KIMI_CHUNK_PROMPT.format(timestamp=stamp))
    return chunks, "".join(text)


def kimi_sampling(model_config: Any) -> dict[str, Any]:
    """The checkpoint's `KIMI_SAMPLING` settings, from its `media_proc_cfg`."""
    from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

    raw = get_hf_file_to_dict(
        "preprocessor_config.json", model_config.model, model_config.revision
    )
    media_cfg = (raw or {}).get("media_proc_cfg") or {}
    missing = [name for name in KIMI_SAMPLING if name not in media_cfg]
    if missing:
        raise MediaError(
            f"{model_config.model}'s preprocessor_config.json states no "
            f"media_proc_cfg {missing}"
        )
    return {name: media_cfg[name] for name in KIMI_SAMPLING}


@functools.cache
def _kimi_frame_loader() -> Any:
    """vLLM's uniform video backend, taking the frames Kimi's rule picks."""
    from vllm.multimodal.video import VideoBackend

    class KimiFrames(VideoBackend):
        @classmethod
        def compute_frames_index_to_sample(
            cls, source: Any, target: Any, *, sample_fps: float, **kwargs: Any
        ) -> list[int]:
            return kimi_frame_indices(
                source.total_frames_num, source.original_fps, sample_fps
            )

    return KimiFrames


def decode_kimi_clip(
    path: str, sampling: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    """One clip's chunks and text."""
    from PIL import Image

    frames, metadata = _kimi_frame_loader().load_bytes(
        Path(path).read_bytes(), sample_fps=sampling["sample_fps"]
    )
    return kimi_chunks(
        [Image.fromarray(frame) for frame in frames],
        metadata["frames_indices"],
        metadata["fps"],
        sampling,
    )


def _decode_kimi(
    task: Task,
    cfg: RunConfig,
    sampling: dict[str, Any],
    pool: Executor,
) -> Callable[[], SampleMedia]:
    """Every clip of `task` as Kimi K2.6's chunks, queued on `pool`.

    A chunk's digest is its clip's plus its index, as vLLM's chat path names chunks
    `{video uuid}-{index}`; without a uuid the item would be hashed whole.
    """
    futures = [pool.submit(decode_kimi_clip, path, sampling) for path in task.videos]

    def finish() -> SampleMedia:
        chunks: list[dict[str, Any]] = []
        digests: list[str] = []
        texts: list[str] = []
        for path, future in zip(task.videos, futures):
            clip_chunks, text = future.result()
            clip = clip_digest(path, cfg.mm_processor_kwargs, None, sampling)
            digests += [_sha256(f"{clip}\nchunk:{i}") for i in range(len(clip_chunks))]
            chunks += clip_chunks
            texts.append(text)
        return SampleMedia(
            data={KIMI_MODALITY: chunks},
            digests=tuple(digests),
            processor_kwargs=cfg.mm_processor_kwargs,
            modality=KIMI_MODALITY,
            placeholders=tuple(texts),
        )

    return finish


def decode_workers(clips: int) -> int:
    """Decode threads: one per clip, at most one per CPU (codecs release the GIL)."""
    cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    return max(1, min(clips, cpus))


def start_decode(
    task: Task, cfg: RunConfig, model_config: Any, pool: Executor
) -> Callable[[], SampleMedia]:
    """Queue `task`'s clips on `pool`; the returned call waits for them in order."""
    if _is_kimi(model_config):
        return _decode_kimi(task, cfg, kimi_sampling(model_config), pool)
    from vllm.transformers_utils.processor import get_video_processor_cls_name

    video_processor = get_video_processor_cls_name(model_config)
    store = os.environ.get(FRAME_STORE_ENV)
    decode_one = (
        functools.partial(stored_clip, store, cfg.mm_processor_kwargs)
        if store
        else decode_clip
    )
    futures = [
        pool.submit(decode_one, path, video_processor, cfg.allowed_local_media_path)
        for path in task.videos
    ]
    return lambda: SampleMedia(
        data={VIDEO_MODALITY: [future.result() for future in futures]},
        digests=tuple(
            clip_digest(path, cfg.mm_processor_kwargs, video_processor)
            for path in task.videos
        ),
        processor_kwargs=cfg.mm_processor_kwargs,
    )


def decode(task: Task, cfg: RunConfig, model_config: Any) -> SampleMedia:
    """Decode every clip of `task` as the engine's model takes them, a clip a thread."""
    with ThreadPoolExecutor(decode_workers(len(task.videos))) as pool:
        return start_decode(task, cfg, model_config, pool)()


def load_media(
    cfg: RunConfig, tasks: Iterable[Task], model_config: Any
) -> dict[str, SampleMedia]:
    """Every sample's clips, decoded, by `Task.source`; empty for a text run."""
    with_clips = {task.source: task for task in tasks if task.videos}
    if not with_clips:
        return {}
    clips = sum(len(task.videos) for task in with_clips.values())
    pool = ThreadPoolExecutor(decode_workers(clips))
    try:
        pending = {
            source: start_decode(task, cfg, model_config, pool)
            for source, task in with_clips.items()
        }
        media = {source: finish() for source, finish in pending.items()}
    finally:
        pool.shutdown(cancel_futures=True)
    return media


def clip_ref(task: Task) -> dict[str, Any] | None:
    """What a split run's driver sends for `task`'s clips, or None for text."""
    return {"videos": list(task.videos)} if task.videos else None


def clip_table(
    cfg: RunConfig, model_config: Callable[[], Any]
) -> dict[tuple[str, ...], SampleMedia]:
    """Every sample's clips, decoded in a split run's worker, by clip paths."""
    if not any(video_paths(cfg.sample_path(sample)) for sample in cfg.samples):
        return {}
    tasks = [load_task(cfg.sample_path(sample)) for sample in cfg.samples]
    by_source = load_media(cfg, tasks, model_config())
    return {tuple(task.videos): by_source[task.source] for task in tasks if task.videos}


def split_request_fields(
    table: dict[tuple[str, ...], SampleMedia] | None,
    ref: dict[str, Any],
    salt: str | None,
) -> dict[str, Any]:
    """The media fields of a split run's request naming `ref` under `salt`."""
    clips = (table or {}).get(tuple(ref["videos"]))
    if clips is None:
        raise MediaError(
            f"this worker decoded no clips {ref['videos']} at startup -- every "
            "sample's clips are decoded before the first request"
        )
    return clips.request_fields(salt or None)


def processor_expansion(cfg: RunConfig, task: Task, tokenizer: Any) -> tuple[Task, int]:
    """`task` bound to its decoded clips, and the ids the engine adds to every prompt.

    vLLM's processor runs over the placeholders alone: every stage carries them once
    and nothing else expands. If this raises, the dry run counts text only.
    """
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY

    stated = {
        name: getattr(cfg, name)
        for name in MULTIMODAL_FIELDS
        if getattr(cfg, name) is not None
    }
    stated["mm_processor_cache_gb"] = 0
    model_config = ModelConfig(
        model=cfg.model,
        trust_remote_code=bool(cfg.trust_remote_code),
        max_model_len=cfg.max_model_len,
        hf_overrides=cfg.hf_overrides(),
        **stated,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    media = decode(task, cfg, model_config)
    layout = expand_lines(cfg, processor, tokenizer, task, media)
    return bind(task, cfg, media), len(layout.served) - len(layout.submitted)


def expand_lines(
    cfg: RunConfig, processor: Any, tokenizer: Any, task: Task, media: SampleMedia
) -> ClipLayout:
    """`task`'s clip lines as submitted and as vLLM's `processor` expands them."""
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs

    ids = tokenizer.encode(bind(task, cfg, media).media_text, add_special_tokens=False)
    out = processor.apply(
        ProcessorInputs(
            ids,
            processor.info.parse_mm_data(media.data),
            hf_processor_mm_kwargs=media.processor_kwargs or {},
        ),
        TimingContext(enabled=False),
    )
    return ClipLayout(tuple(ids), tuple(out["prompt_token_ids"]))


def layouts_needed(cfg: RunConfig) -> bool:
    """Whether a split run places prompts by engine position: every P/D run does."""
    return getattr(cfg.pd, "enabled", False) is True


def clip_layouts(
    cfg: RunConfig,
    table: Mapping[tuple[str, ...], SampleMedia],
    model_config: Any,
    tokenizer: Any,
) -> dict[tuple[str, ...], ClipLayout]:
    """Every sample's `ClipLayout`, by clip paths, from its decoded clips.

    No processor cache: the sender cache would take a clip it never sent as held.
    """
    from vllm.multimodal import MULTIMODAL_REGISTRY

    processor = MULTIMODAL_REGISTRY.create_processor(model_config, tokenizer=tokenizer)
    layouts: dict[tuple[str, ...], ClipLayout] = {}
    for sample in cfg.samples:
        task = load_task(cfg.sample_path(sample))
        if task.videos and tuple(task.videos) not in layouts:
            layouts[tuple(task.videos)] = expand_lines(
                cfg, processor, tokenizer, task, table[tuple(task.videos)]
            )
    return layouts


# Threads processing clip-bearing inputs off the event loop; one with the cache on.
MEDIA_INPUT_WORKERS = 4


class InputLane:
    """A clip-bearing prompt's multimodal processing, off the event loop.

    `AsyncLLM.add_request` would run it on the frontend's loop, stalling every program
    while a video is processed. Here `InputPreprocessor.preprocess` runs on a thread and
    `add_request` takes its result as it takes `Renderer.render_cmpl`'s. With vLLM's
    processor cache on, one thread runs in order, as vLLM's async renderer does: the
    cache is not thread-safe and the engine mirrors its LRU order.
    """

    def __init__(self, llm: Any, workers: int) -> None:
        cached = llm.renderer.mm_processor_cache is not None
        self.workers = 1 if cached else workers
        self._preprocess = llm.input_processor.input_preprocessor.preprocess
        self._pool = ThreadPoolExecutor(self.workers, thread_name_prefix="media-input")

    async def prepare(self, prompt: Any) -> Any:
        """`prompt` as `add_request` takes it, processed here if it has clips."""
        if not prompt.get("multi_modal_data"):
            return prompt
        arrival = time.time()
        processed = await asyncio.get_running_loop().run_in_executor(
            self._pool, self._preprocess, prompt
        )
        processed["arrival_time"] = arrival
        return processed


_LANES: weakref.WeakKeyDictionary[Any, InputLane] = weakref.WeakKeyDictionary()


def input_lane(llm: Any) -> InputLane:
    """`llm`'s one lane: every clip-bearing request to it queues on it."""
    lane = _LANES.get(llm)
    if lane is None:
        lane = _LANES[llm] = InputLane(llm, MEDIA_INPUT_WORKERS)
    return lane

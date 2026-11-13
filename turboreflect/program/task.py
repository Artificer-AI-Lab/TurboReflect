# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""The task file layout: everything downstream sees `Task`, never the raw JSON.

`videos`, optional, lists clip files in order, relative to the task file.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REQUIRED_FIELDS = ("id", "system", "user_task", "context", "rubric")

_OPTIONAL_FIELDS = ("videos",)


class SampleError(Exception):
    """A sample file is missing, malformed, or does not match the format."""


@dataclass(frozen=True)
class Task:
    """One bench task, decoupled from the file it came from."""

    id: str
    system: str
    user_task: str
    context: dict[str, str]
    rubric: list[str]
    source: str = ""
    extras: dict[str, Any] = field(default_factory=dict)
    # Resolved clip files, in placeholder order.
    videos: tuple[str, ...] = ()
    # One model placeholder line per clip (`media.bind`); empty for a text sample.
    media_text: str = ""

    def as_prompt_task(self) -> dict[str, Any]:
        """The keys the bench prompt layer reads; `media` only for clips."""
        payload = {
            "system": self.system,
            "user_task": self.user_task,
            "context": self.context,
            "rubric": self.rubric,
        }
        if self.media_text:
            payload["media"] = self.media_text
        return payload


def _videos(raw: dict[str, Any], path: Path) -> tuple[str, ...]:
    videos = raw.get("videos", [])
    if not isinstance(videos, list) or not all(
        isinstance(v, str) and v for v in videos
    ):
        raise SampleError(f"{path}: field 'videos' must be a list of file paths")
    return tuple(str(path.parent / v) if not os.path.isabs(v) else v for v in videos)


def video_paths(path: str | Path) -> tuple[str, ...]:
    """The clip files a sample names, resolved; none if it is not a JSON object.

    Raises:
        SampleError: If `videos` is not a list of paths.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_bytes())
    except ValueError:
        return ()
    return _videos(raw, path) if isinstance(raw, dict) else ()


_DIGESTS: dict[tuple[str, int, int], str] = {}


def file_sha256(path: str | Path) -> str:
    """A file's SHA-256, cached per process until its size or mtime changes."""
    stat = os.stat(path)
    key = (str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _DIGESTS:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        _DIGESTS[key] = digest.hexdigest()
    return _DIGESTS[key]


def load_task(path: str | Path) -> Task:
    """Read one `data/<id>.json` into a `Task`.

    Context documents keep their written order: the bench shuffles them so position is
    no proxy for importance.

    Raises:
        SampleError: If the file is missing, malformed or lacks a required field.
    """
    path = Path(path)
    if not path.is_file():
        raise SampleError(f"sample file does not exist: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SampleError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise SampleError(f"{path}: expected a JSON object at the top level")

    missing = [k for k in _REQUIRED_FIELDS if k not in raw]
    if missing:
        raise SampleError(f"{path}: missing required field(s): {', '.join(missing)}")

    if not isinstance(raw["context"], dict):
        raise SampleError(f"{path}: field 'context' must be an object")
    if not isinstance(raw["rubric"], list):
        raise SampleError(f"{path}: field 'rubric' must be a list")

    return Task(
        id=raw["id"],
        system=raw["system"],
        user_task=raw["user_task"],
        context=dict(raw["context"]),
        rubric=list(raw["rubric"]),
        source=str(path),
        extras={
            k: v
            for k, v in raw.items()
            if k not in _REQUIRED_FIELDS and k not in _OPTIONAL_FIELDS
        },
        videos=_videos(raw, path),
    )

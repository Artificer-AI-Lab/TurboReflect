# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors
"""The driver's admission controller and the stage event it is told."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from turboreflect.config import RunConfig

STAGE_CLASS_STRIDE = 1_000_000


@dataclass(frozen=True)
class StageSubmit:
    """A request submitted to the engine; priority does not use stage depth."""

    has_predecessor: bool
    program_index: int


class AdmissionController:
    """Cap the programs in flight and give each request its vLLM priority."""

    def __init__(self, cfg: RunConfig, *, concurrency: int) -> None:
        self.cfg = cfg
        self._semaphore = asyncio.Semaphore(concurrency)

    async def start(self) -> None:
        await self._semaphore.acquire()

    async def submit(self, sub: StageSubmit) -> int:
        return self.cfg.stage_priority_value(
            successor=sub.has_predecessor,
            program_index=sub.program_index,
        )

    async def end(self) -> None:
        self._semaphore.release()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""The stages of one program's reflection loop.

The run configuration fixes the round count and no generated text is read; the
controller is told only that a stage went out, never how many follow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from turboreflect.config import (
    FINAL_ONLY_PATHS,
    PIPELINE_MODE_ROUNDS,
    RunConfig,
)
from turboreflect.metrics.results import STAGE_OUTPUT_NAMES, round_output_name
from turboreflect.program.prompt import base_stage, stage_name

if TYPE_CHECKING:
    from turboreflect.metrics.results import ProgramResult


@dataclass(frozen=True)
class PlannedStage:
    """One request of a program, as its driver submits it.

    Attributes:
        stage: Request and record name; carries its round under the round loop.
        nest: Extend the previous stage's transcript instead of calling the bench's
            builders; set from the loop's second round on.
        max_tokens: Generation length when it differs from the program's.
    """

    stage: str
    output_name: str
    nest: bool = False
    max_tokens: int | None = None

    @property
    def base(self) -> str:
        return base_stage(self.stage)


@dataclass(frozen=True)
class StagePlan:
    """A program's stages in order, and the rounds its loop runs, if it loops."""

    stages: tuple[PlannedStage, ...]
    rounds_run: int = 0


def stage_plan(cfg: RunConfig, *, decode_path: str) -> StagePlan:
    """Every stage one program runs, fixed before the first starts.

    The fixed pipeline is `cfg.stages`; the loop is generation, then per round a
    feedback and a refinement stage (only refinement under `cfg.two_stage`).
    `decode_path` selects whether refinement runs to `refinement_gen_tokens`.
    """
    refinement_tokens = (
        cfg.refinement_gen_tokens if decode_path in FINAL_ONLY_PATHS else None
    )
    if cfg.pipeline_mode != PIPELINE_MODE_ROUNDS:
        return StagePlan(
            tuple(
                PlannedStage(
                    stage,
                    STAGE_OUTPUT_NAMES[stage],
                    max_tokens=refinement_tokens if stage == "refinement" else None,
                )
                for stage in cfg.stages
            )
        )
    rounds = cfg.max_rounds
    stages = [PlannedStage("generation", round_output_name("generation"))]
    for round_index in range(1, rounds + 1):
        if not cfg.two_stage:
            feedback = stage_name("feedback", round_index)
            stages.append(
                PlannedStage(
                    feedback,
                    round_output_name(feedback),
                    nest=round_index > 1,
                )
            )
        refinement = stage_name("refinement", round_index)
        stages.append(
            PlannedStage(
                refinement,
                round_output_name(refinement),
                nest=round_index > 1,
                max_tokens=refinement_tokens,
            )
        )
    return StagePlan(tuple(stages), rounds_run=rounds)


def next_inputs(
    stage: str, text: str, draft: str | None, critique: str | None
) -> tuple[str | None, str | None]:
    """The `(draft, critique)` the stage after `stage` reads, given its `text`."""
    base = base_stage(stage)
    if base == "feedback":
        return draft, text
    return text, critique


def record_rounds(result: ProgramResult, plan: StagePlan) -> None:
    """File the rounds run; `revision`, which scoring reads, is the last refinement."""
    result.rounds_run = plan.rounds_run
    final_key = next(
        (s.output_name for s in reversed(plan.stages) if s.base == "refinement"),
        round_output_name("generation"),
    )
    result.outputs["revision"] = result.outputs[final_key]
    result.output_token_ids["revision"] = list(result.output_token_ids[final_key])
    result.artifact = result.outputs["revision"]

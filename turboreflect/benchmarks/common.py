# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""What the public benchmarks share: stage assembly, task files, fit and run reading.

The task `x` is the benchmark's own prompt, unaltered: one user message, no system
message, with the document (kept in `context`) substituted into its `{context}` slot
and nothing else interpolated. Stages nest as `custom.prompts` describes
for layout `continue`, the only layout built here. A builder writes `PROMPTS_PY`
beside its tasks, which is how a run finds this stage assembly
(`config.stage_assembly_root`).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from turboreflect.program.stages import (
    FEEDBACK_INSTRUCTIONS,
    MERGED_REFINEMENTS,
    REFINEMENT_INSTRUCTIONS,
    merged_parts,
)

# What a task folder's `prompts.py` re-exports.
__all__ = [
    "CONTEXT_SLOT",
    "REFINEMENT_INSTRUCTIONS",
    "FEEDBACK_INSTRUCTIONS",
    "build_messages",
    "build_prefix_user",
]

CONTEXT_SLOT = "{context}"

PROMPTS_PY = '''\
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""Stage assembly of these tasks: `turboreflect.benchmarks.common`."""

from turboreflect.benchmarks.common import *  # noqa: F403
from turboreflect.benchmarks.common import __all__  # noqa: F401
'''

# Re-cuts by the miss; tokens can merge across the join, so the closest cut is kept.
MAX_FIT_STEPS = 4

# Export folder per stage -> the program record's output key.
EXPORTED_STAGES = {"generation": "draft", "refinement": "revision"}

Offsets = Sequence[tuple[int, int]]


class PromptError(Exception):
    """A task this stage assembly cannot render."""


def render_document(context: Any) -> str:
    """`context` -> the document body, its values joined in insertion order."""
    if isinstance(context, dict):
        return "\n\n".join(str(v) for v in context.values())
    return str(context)


def build_prefix_user(task: dict[str, Any]) -> str:
    """The one user message all three stages share, token for token.

    Raises:
        PromptError: If the task's prompt has no `{context}` slot for its document.
    """
    head, slot, tail = str(task["user_task"]).partition(CONTEXT_SLOT)
    if not slot:
        raise PromptError(
            f"this task's user_task has no {CONTEXT_SLOT} slot, so its document has "
            "nowhere to go"
        )
    return head + render_document(task.get("context") or {}) + tail


def prefix_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    """The shared prefix itself: one user message, opening all three stages."""
    return [{"role": "user", "content": build_prefix_user(task)}]


def generation_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    """x -> draft answer"""
    return prefix_messages(task)


def feedback_messages(
    task: dict[str, Any], draft: str, feedback: str = "mustfix"
) -> list[dict[str, str]]:
    """x + draft + I_reflect -> critique"""
    return generation_messages(task) + [
        {"role": "assistant", "content": draft},
        {"role": "user", "content": FEEDBACK_INSTRUCTIONS[feedback]},
    ]


def refinement_messages(
    task: dict[str, Any],
    draft: str,
    critique: str,
    feedback: str = "mustfix",
    refinement: str = "full",
) -> list[dict[str, str]]:
    """x + draft + I_reflect + critique + I_revise -> final answer

    A two-stage refinement reviews and revises in one turn after the generation
    stage.
    """
    if refinement in MERGED_REFINEMENTS:
        return generation_messages(task) + [
            {"role": "assistant", "content": draft},
            {"role": "user", "content": REFINEMENT_INSTRUCTIONS[refinement]},
        ]
    return feedback_messages(task, draft, feedback) + [
        {"role": "assistant", "content": critique},
        {"role": "user", "content": REFINEMENT_INSTRUCTIONS[refinement]},
    ]


def build_messages(
    task: dict[str, Any],
    stage: str = "generation",
    draft: str | None = None,
    critique: str | None = None,
    feedback: str = "mustfix",
    refinement: str = "full",
) -> list[dict[str, str]]:
    """Single entry point. `stage` is `generation`, `feedback` or `refinement`."""
    if stage == "generation":
        return generation_messages(task)
    if stage == "feedback":
        if draft is None:
            raise ValueError("the feedback stage needs a draft")
        return feedback_messages(task, draft, feedback)
    if stage == "refinement":
        if draft is None or (critique is None and refinement not in MERGED_REFINEMENTS):
            raise ValueError("the refinement stage needs a draft and a critique")
        return refinement_messages(task, draft, critique, feedback, refinement)
    raise ValueError(f"unknown stage {stage!r}; use generation / feedback / refinement")


def task_folder(root: Path) -> tuple[Path, Path]:
    """Lay out `prompts.py`, `data/` and `rubrics/` (gold answers a run never reads)."""
    data_dir, rubric_dir = root / "data", root / "rubrics"
    data_dir.mkdir(parents=True, exist_ok=True)
    rubric_dir.mkdir(parents=True, exist_ok=True)
    (root / "prompts.py").write_text(PROMPTS_PY)
    return data_dir, rubric_dir


def write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def generation_counter(tokenizer: Any, root: Path) -> Callable[[dict[str, Any]], int]:
    """A task's rendered generation-prompt length, through the render a run uses."""
    from turboreflect.config import RunConfig  # noqa: PLC0415
    from turboreflect.program.prompt import render  # noqa: PLC0415
    from turboreflect.program.task import Task  # noqa: PLC0415

    cfg = RunConfig(
        model=".",
        samples=("x",),
        gen_tokens=1,
        max_model_len=1 << 30,
        bench_root=str(root),
        layout="continue",
    )

    def count(task: dict[str, Any]) -> int:
        fields = {k: task[k] for k in Task.__dataclass_fields__ if k in task}
        return render(tokenizer, Task(**fields), "generation", cfg).n_tokens

    return count


def middle_truncate(text: str, offsets: Offsets, budget: int) -> str:
    """The first `budget // 2` tokens of `text` and the last `budget - budget // 2`.

    Cut on the tokens' character offsets, so no character is split.
    """
    n = len(offsets)
    if budget >= n:
        return text
    head = budget // 2
    tail = budget - head
    head_end = offsets[head - 1][1] if head else 0
    tail_start = offsets[n - tail][0] if tail else len(text)
    return text[:head_end] + text[tail_start:]


def fit_document(
    make: Callable[[str, bool], dict[str, Any]],
    text: str,
    offsets: Offsets,
    target: int,
    count: Callable[[dict[str, Any]], int],
) -> tuple[dict[str, Any], int]:
    """A task whole when it fits `target` tokens, else its document cut to land on it.

    Args:
        make: Builds the task from a document and whether that document was cut.
        count: `generation_counter`'s function.

    Returns:
        The task and its rendered generation-prompt length.
    """
    whole = make(text, False)
    n = count(whole)
    if n <= target:
        return whole, n
    budget = target - count(make("", True))
    best = None
    for _ in range(MAX_FIT_STEPS):
        task = make(middle_truncate(text, offsets, budget), True)
        n = count(task)
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (task, n)
        if n == target:
            break
        budget += target - n
    return best


def read_run(
    path: str | Path, tasks: str | None = None
) -> tuple[Path, dict[str, dict[str, str]]]:
    """A finished run's task folder, and `{task id: {stage: answer}}`.

    A task run more than once keeps its first program in id order; a two-stage
    refinement's answer is its revised response, without the review.

    Args:
        path: A run directory, or a folder of them, whose newest is read.
        tasks: The task folder; by default the one the run read.
    """
    from turboreflect.metrics.report import run_dirs  # noqa: PLC0415
    from turboreflect.metrics.rundir import RunDir  # noqa: PLC0415

    found = run_dirs(path)
    if not found:
        raise SystemExit(f"no run directory under {path}")
    run = RunDir(found[-1])
    answers: dict[str, dict[str, str]] = {}
    for _, program in sorted(run.programs.items()):
        outputs = program["outputs"]
        answers.setdefault(
            program["sample_id"],
            {
                stage: merged_parts(outputs.get(key) or "")[1]
                for stage, key in EXPORTED_STAGES.items()
            },
        )
    return Path(tasks or run.config_value("bench_root")), answers


def read_rubric(root: Path, rubric_id: str) -> dict[str, Any]:
    path = root / "rubrics" / f"{rubric_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))

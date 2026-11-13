# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""Stage assembly for tasks in the shared task format.

Task files carry `system` / `user_task` / `context` / `rubric`; the stage
instructions are the pipeline's (`turboreflect.program.stages`).
`layout="continue"` nests each stage in the one before:

    x           = [system] + [user: Task / Context / Rubric + generation ask]
    Generation  = x                                                   -> draft
    Feedback    = Generation + [assistant: draft]    + [user: I_reflect]
    Refinement  = Feedback   + [assistant: critique] + [user: I_revise]

Context reuse, and so suffix decoding, needs `x` token-for-token identical in all
three stages and each stage a strict extension of the one before; reuse degrades
silently when that breaks. The rubric sits in `x` because the feedback instruction
judges against "the original task".

`layout="requote"` keeps the generation stage; each later stage is one user message
after the system, holding the task, each upstream output under a short header, and
the stage's instruction. The feedback prompt's matched prefix runs through the task
and its append is the draft and I_reflect; the refinement prompt's matched prefix
runs through the draft and its append is the critique and I_revise:

    Feedback    = [system] + [user: task + "Draft:" draft + I_reflect]
    Refinement  = [system] + [user: task + "Draft:" draft
                                     + "Critique:" critique + I_revise]

A later round passes its sections as `quoted`, numbered (`Draft 2:`). A two-stage
refinement (`MERGED_REFINEMENTS`) writes the review and the revision in one reply,
built as the feedback stage is with its own instruction:

    Refinement  = Generation + [assistant: draft] + [user: I_merged]
    Refinement  = [system] + [user: task + "Draft:" draft + I_merged]  (requote)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from turboreflect.program.stages import (
    FEEDBACK_INSTRUCTIONS,
    GENERATION_INSTRUCTION,
    MERGED_REFINEMENTS,
    REFINEMENT_INSTRUCTIONS,
)

__all__ = [
    "GENERATION_INSTRUCTION",
    "LAYOUTS",
    "REFINEMENT_INSTRUCTIONS",
    "QUOTE_SEP",
    "FEEDBACK_INSTRUCTIONS",
    "build_messages",
    "build_prefix_user",
    "quote_header",
    "requote_user",
]

LAYOUTS = ("continue", "requote")

# What follows the task and each quoted section under `requote`.
QUOTE_SEP = "\n\n"

_QUOTE_LABELS = {"draft": "Draft", "critique": "Critique"}


def render_document(text: str) -> str:
    """One context document -> a section with a second-level heading.

    The `## ` on its first line, its title, tells a new document from an all-caps
    subheading inside one.
    """
    head, _, body = str(text).partition("\n")
    head, body = head.strip(), body.strip("\n")
    return f"## {head}\n\n{body}" if body else f"## {head}"


def render_context(context: Any) -> str:
    """The body of `# Context`.

    A dict's keys are segment labels and are not rendered; its values keep the
    insertion order the sample sets.
    """
    if isinstance(context, dict):
        return "\n\n".join(render_document(v) for v in context.values())
    return render_document(str(context))


def build_prefix_user(task: dict[str, Any]) -> str:
    """The task as the dataset states it, without the generation ask.

    Clip placeholders, one line each, open `# Context`, so every stage carries
    them inside the shared prefix.
    """
    rubric = task.get("rubric") or []
    if isinstance(rubric, list):
        rubric = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rubric))
    context = render_context(task.get("context") or {})
    if task.get("media"):
        context = f"{task['media']}\n\n{context}" if context else task["media"]
    return (
        f"# Task\n{task['user_task']}\n\n"
        f"# Context\n{context}\n\n"
        f"# Rubric (the response must satisfy all of these)\n{rubric}"
    )


def build_task_user(task: dict[str, Any]) -> str:
    """`x`: the dataset's task followed by the generation ask."""
    return f"{build_prefix_user(task)}\n\n{GENERATION_INSTRUCTION}"


def generation_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    """x -> draft"""
    return [
        {"role": "system", "content": task["system"]},
        {"role": "user", "content": build_task_user(task)},
    ]


def feedback_messages(
    task: dict[str, Any], draft: str, feedback: str = "mustfix"
) -> list[dict[str, str]]:
    """x + draft + I_reflect -> critique, extending `generation_messages`."""
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
    """x + draft + I_reflect + critique + I_revise -> revision

    A two-stage refinement takes no critique: x + draft + I_merged -> review +
    revision. Keeping the feedback turn shows the model the critique was asked
    for, and lets the feedback instruction's tokens hit the cache too.
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


def quote_header(kind: str, round_index: int | None = None) -> str:
    """The line a quoted `draft` or `critique` opens with: `Draft 2:` in round 2."""
    label = _QUOTE_LABELS[kind]
    return f"{label}:\n" if round_index is None else f"{label} {round_index}:\n"


def requote_user(
    task: dict[str, Any], quoted: Sequence[tuple[str, str]], instruction: str
) -> str:
    """The task, each quoted `(header, text)` section, then `instruction`."""
    sections = "".join(f"{header}{text}{QUOTE_SEP}" for header, text in quoted)
    return f"{build_prefix_user(task)}{QUOTE_SEP}{sections}{instruction}"


def build_messages(
    task: dict[str, Any],
    stage: str = "generation",
    draft: str | None = None,
    critique: str | None = None,
    feedback: str = "mustfix",
    refinement: str = "full",
    layout: str = "continue",
    quoted: Sequence[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Single entry point. `stage` is `generation`, `feedback` or `refinement`.

    Under `layout="requote"`, `quoted` lists a later stage's `(header, text)`
    sections in prompt order; None quotes the draft (and critique) unnumbered.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; use one of {list(LAYOUTS)}")
    if stage == "generation":
        return generation_messages(task)
    if stage == "feedback":
        if draft is None:
            raise ValueError("the feedback stage needs a draft")
        if layout == "requote":
            sections = [(quote_header("draft"), draft)]
            return _requote_messages(
                task, quoted, sections, FEEDBACK_INSTRUCTIONS[feedback]
            )
        return feedback_messages(task, draft, feedback)
    if stage == "refinement":
        merged = refinement in MERGED_REFINEMENTS
        if draft is None or (critique is None and not merged):
            raise ValueError("the refinement stage needs a draft and a critique")
        if layout == "requote":
            sections = [(quote_header("draft"), draft)]
            if not merged:
                sections.append((quote_header("critique"), critique))
            return _requote_messages(
                task, quoted, sections, REFINEMENT_INSTRUCTIONS[refinement]
            )
        return refinement_messages(task, draft, critique, feedback, refinement)
    raise ValueError(f"unknown stage {stage!r}; use generation / feedback / refinement")


def _requote_messages(
    task: dict[str, Any],
    quoted: Sequence[tuple[str, str]] | None,
    default: Sequence[tuple[str, str]],
    instruction: str,
) -> list[dict[str, str]]:
    sections = default if quoted is None else quoted
    return [
        {"role": "system", "content": task["system"]},
        {"role": "user", "content": requote_user(task, sections, instruction)},
    ]

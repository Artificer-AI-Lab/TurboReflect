# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""Stage instructions, identical across task sets; a task set only supplies `x`.

Each stage is a token-level extension of the one before it:

    Generation  [system?][user: x]                              -> d
    Feedback    Generation + [assistant: d][user: I_reflect]    -> c
    Refinement  Feedback + [assistant: c][user: I_revise]       -> r

The two-stage form, `Generation + [assistant: d][user: I_merged] -> review + r`, writes
the review and the whole revision under `MERGED_REVISION_MARKER` in one call; each round
revises the newest response. `GENERATION_INSTRUCTION` is appended to `x` only where the
task is not already a complete request (RULER items end in their own answer prefix).
"""

from __future__ import annotations

import regex as re

GENERATION_INSTRUCTION = (
    "Complete the task above. Follow all requirements in the original request "
    "and return the response in the requested format."
)

FEEDBACK_INSTRUCTION = (
    "Review your previous response against the original task.\n\n"
    "Determine whether the response is correct, complete, relevant, and "
    "compliant with every requirement and output constraint in the original "
    "request. Recheck the source material, reasoning, calculations, and "
    "factual claims whenever they are relevant.\n\n"
    "Identify only concrete problems that require correction. For each "
    "problem, state:\n"
    "1. what is wrong or missing;\n"
    "2. why it conflicts with the original task or available evidence;\n"
    "3. what must change in the revision.\n\n"
    "If the response already satisfies the task, state that no correction is "
    "required. Do not produce the revised response in this stage.\n\n"
    "Format your review as:\n"
    "VERDICT: PASS or REVISION_NEEDED\n\n"
    "ISSUES:\n"
    "- <specific issue, supporting reason, and required change>\n\n"
    'If there are no problems, write "VERDICT: PASS" and "ISSUES:" followed '
    'by "- None."'
)

REFINEMENT_INSTRUCTION = (
    "Produce the final response to the original task using the critique "
    "above.\n\n"
    "Correct every identified problem while preserving content that is "
    "already correct. Recheck the final response against the original "
    "instructions, evidence, and output constraints.\n\n"
    "Return the complete revised response in exactly the format requested by "
    "the original task. Do not include the critique, revision notes, or any "
    "discussion of the editing process."
)

GOAL_FEEDBACK_INSTRUCTION = (
    "Review your latest response against the goal criteria in the original "
    "task. Judge what the response actually contains, not what it says it "
    "does, and check it against the source material.\n\n"
    "First judge, then act:\n"
    "1. JUDGE the latest response:\n"
    "   - accomplished: every goal criterion is met;\n"
    "   - in_progress: some criteria are not met yet, but revising this "
    "response can meet them;\n"
    "   - failed: it is wrong, broken, or contradicted by the source "
    "material.\n"
    "2. ACT: if every goal criterion is met, finish. Otherwise ask for the "
    "next one or two most important changes, the way a demanding but fair "
    "client would: short, specific, and never beyond the goal criteria. Do "
    "not produce the revised response in this stage.\n\n"
    "Format your review as:\n"
    "OUTCOME: accomplished | in_progress | failed\n"
    "REASON: <one short sentence grounded in the response and the source "
    "material>\n"
    "NEXT: DONE, or the requested change(s)"
)

GOAL_REFINEMENT_INSTRUCTION = (
    "Revise your latest response to make the change(s) requested in the "
    "review above.\n\n"
    "Keep everything that already meets the goal criteria. If the review says "
    "DONE, return the response unchanged.\n\n"
    "Return the complete revised response in exactly the format requested by "
    "the original task. Do not include the review, revision notes, or any "
    "discussion of the edits."
)

MERGED_REVISION_MARKER = "REVISED RESPONSE:"

MERGED_REFINEMENT_INSTRUCTION = (
    "Review your previous response against the original task, then produce "
    "the final response in the same reply. If your previous reply ended with "
    "a revised response, that revised response is the one to review.\n\n"
    "Determine whether the response is correct, complete, relevant, and "
    "compliant with every requirement and output constraint in the original "
    "request. Recheck the source material, reasoning, calculations, and "
    "factual claims whenever they are relevant.\n\n"
    "Identify only concrete problems that require correction. For each "
    "problem, state:\n"
    "1. what is wrong or missing;\n"
    "2. why it conflicts with the original task or available evidence;\n"
    "3. what must change in the revision.\n\n"
    "Then correct every identified problem while preserving content that is "
    "already correct. If there are no problems, repeat the response "
    "unchanged.\n\n"
    "Format your reply as:\n"
    "VERDICT: PASS or REVISION_NEEDED\n\n"
    "ISSUES:\n"
    "- <specific issue, supporting reason, and required change>\n\n"
    f"{MERGED_REVISION_MARKER}\n"
    "<the complete revised response, in exactly the format requested by the "
    "original task>\n\n"
    'If there are no problems, write "VERDICT: PASS" and "ISSUES:" followed '
    'by "- None." before the revised response.'
)

GOAL_MERGED_REFINEMENT_INSTRUCTION = (
    "Review your latest response against the goal criteria in the original "
    "task, then revise it in the same reply. If your previous reply ended "
    "with a revised response, that revised response is the latest response. "
    "Judge what the response actually contains, not what it says it does, "
    "and check it against the source material.\n\n"
    "First judge, then act:\n"
    "1. JUDGE the latest response:\n"
    "   - accomplished: every goal criterion is met;\n"
    "   - in_progress: some criteria are not met yet, but revising this "
    "response can meet them;\n"
    "   - failed: it is wrong, broken, or contradicted by the source "
    "material.\n"
    "2. ACT: if every goal criterion is met, keep the response unchanged. "
    "Otherwise name the next one or two most important changes, the way a "
    "demanding but fair client would: short, specific, and never beyond the "
    "goal criteria. Then make them, keeping everything that already meets "
    "the goal criteria.\n\n"
    "Format your reply as:\n"
    "OUTCOME: accomplished | in_progress | failed\n"
    "REASON: <one short sentence grounded in the response and the source "
    "material>\n"
    "NEXT: DONE, or the change(s) made below\n"
    f"{MERGED_REVISION_MARKER}\n"
    "<the complete revised response, in exactly the format requested by the "
    "original task; unchanged if NEXT is DONE>"
)

# Indexed by `cfg.feedback` and `cfg.refinement`. `mustfix` / `full` are the defaults;
# `goal` judges against the task's goal criteria and asks for one or two changes per
# round; `merged` and `goal_merged` are their two-stage forms (no feedback stage).
FEEDBACK_INSTRUCTIONS = {
    "mustfix": FEEDBACK_INSTRUCTION,
    "goal": GOAL_FEEDBACK_INSTRUCTION,
}

REFINEMENT_INSTRUCTIONS = {
    "full": REFINEMENT_INSTRUCTION,
    "goal": GOAL_REFINEMENT_INSTRUCTION,
    "merged": MERGED_REFINEMENT_INSTRUCTION,
    "goal_merged": GOAL_MERGED_REFINEMENT_INSTRUCTION,
}

MERGED_REFINEMENTS = frozenset({"merged", "goal_merged"})

_MERGED_REVISION = re.compile(
    r"^[ \t>#*_]*"
    + re.escape(MERGED_REVISION_MARKER.rstrip(":"))
    + r"[ \t*_]*:[ \t*_]*",
    re.M,
)


def merged_parts(reply: str) -> tuple[str, str]:
    """A two-stage refinement reply as `(review, revised response)`.

    Split at the last line opening with `MERGED_REVISION_MARKER`, markdown emphasis
    allowed; a reply with none, as any other stage's output, is all response.
    """
    marks = list(_MERGED_REVISION.finditer(reply))
    if not marks:
        return "", reply
    return reply[: marks[-1].start()].strip(), reply[marks[-1].end() :].strip()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""Stage messages to the exact token ids the engine is handed.

The chat template is applied here and ids, not strings, are submitted, so a dry run
prices a 128k sample with only a tokenizer and counts what the engine sees. Stage
assembly itself stays with the bench (`load_bench`).

A later stage continues the previous one by one of three joins (`Rendered.join`):

- `JOIN_TOKENS` (`render_successor`): the previous prompt's ids, the ids that stage
  generated and the ids the template writes up to the next generation prompt. A
  sampled sequence is rarely the canonical tokenization of its own text, so a
  re-rendered prompt would part from the cached sequence inside the generation.
- `JOIN_TEXT` (`render_incremental`): the messages re-rendered, encoding only from the
  last added token both prompts agree on. The pre-tokenizer splits on added tokens, so
  both encodes agree there, while a plain character boundary can merge (a draft
  opening with a newline joins the generation prompt's). The fallback where a token
  join is not possible, such as a template that rewrites history it rendered.
- `JOIN_REQUOTE` (`render_requote`, layout `requote`): one user turn quoting the task
  and each upstream output under a header.

`Rendered.prefix_shortfall` counts the previous prompt's tokens a stage does not
repeat, such as an empty `<think>` block a template puts in the generation prompt but
not in the assistant turn that replaces it.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import regex as re

from turboreflect.config import (
    PIPELINE_MODE_ROUNDS,
    PROMPT_LAYOUT_CONTINUE,
    PROMPT_LAYOUT_REQUOTE,
    RunConfig,
)
from turboreflect.program.stages import MERGED_REFINEMENTS
from turboreflect.program.task import Task

STAGES = ("generation", "feedback", "refinement")

# A round-loop stage is the base name plus `_<round>`; generation carries none, so its
# request id and result key match the fixed pipeline's.
_ROUND_STAGE = re.compile(r"^(generation|feedback|refinement)(?:_(\d+))?$")


def stage_name(base: str, round_index: int) -> str:
    """`("feedback", 2)` -> `"feedback_2"`; `"generation"` keeps its bare name."""
    if base == "generation":
        return base
    return f"{base}_{round_index}"


def base_stage(stage: str) -> str:
    """The base stage a possibly-numbered stage name stands for.

    Raises:
        PromptError: On a name that is not one of the three, numbered or not.
    """
    match = _ROUND_STAGE.match(stage)
    if match is None:
        raise PromptError(f"unknown stage {stage!r}; expected one of {list(STAGES)}")
    return match.group(1)


def stage_round(stage: str) -> int | None:
    """The round a stage name carries, or None when it carries none."""
    match = _ROUND_STAGE.match(stage)
    if match is None or match.group(2) is None:
        return None
    return int(match.group(2))


# How far back from the previous prompt's end a splice boundary may sit; ChatML's is
# about twenty characters back. Past it the render encodes in full.
_MAX_SPLICE_TAIL_CHARS = 4096

_BENCH_CACHE: dict[str, ModuleType] = {}


class PromptError(Exception):
    """The bench prompt layer is unreachable or rejected the stage's arguments."""


def load_bench(bench_root: str) -> ModuleType:
    """Import the bench's `prompts` module from `bench_root`."""
    cached = _BENCH_CACHE.get(bench_root)
    if cached is not None:
        return cached
    root = Path(bench_root)
    if not (root / "prompts.py").is_file():
        raise PromptError(
            f"no prompts.py under {root} -- point bench_root at a directory that "
            "holds one"
        )
    sys.path.insert(0, str(root))
    try:
        import prompts  # noqa: PLC0415
    except ImportError as exc:
        raise PromptError(f"failed to import prompts from {root}: {exc}") from exc
    _BENCH_CACHE[bench_root] = prompts
    return prompts


@dataclass(frozen=True)
class Rendered:
    """One stage's prompt, as both messages and the ids that get submitted."""

    stage: str
    token_ids: list[int]
    messages: list[dict[str, str]]
    # The template as a string, for the next stage to splice against; None from a
    # tokenizer that only tokenizes, and the next stage pays a full encode.
    text: str | None = None
    # Previous-stage tokens this prompt does not repeat: 0 is an exact extension, a
    # large number an early divergence. This and `join` are None on a first stage.
    prefix_shortfall: int | None = None
    join: str | None = None
    # `JOIN_REQUOTE` only: where the task ends, where the last quoted section ends
    # (separator included; the next stage matches through it), and each section's
    # header and text in prompt order.
    frame_end: int | None = None
    body_end: int | None = None
    quoted: tuple[tuple[str, str], ...] = ()

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    def shape(self) -> list[dict[str, Any]]:
        """Per-message role and size; bodies are rebuilt from the sample id."""
        return [{"role": m["role"], "chars": len(m["content"])} for m in self.messages]


def bench_stage_kwargs(cfg: RunConfig) -> dict[str, Any]:
    """The stage-shaping arguments this run asks the bench for.

    `layout` goes only off its default, so a bench without layouts still runs.
    """
    kwargs = {"feedback": cfg.feedback, "refinement": cfg.refinement}
    if cfg.layout != PROMPT_LAYOUT_CONTINUE:
        kwargs["layout"] = cfg.layout
    return kwargs


# The instruction turn a later round appends, per (bench, variants): derived once, not
# rebuilt beside a 128k context for every stage.
_ROUND_TAIL_CACHE: dict[tuple[str, ...], dict[str, str]] = {}

# Any string: the instruction turn read off the probe does not carry it.
_TAIL_PROBE = "probe"


def _round_instruction(
    bench: ModuleType, payload: dict[str, Any], stage: str, kwargs: dict[str, Any]
) -> dict[str, str]:
    """The instruction turn the bench appends for `stage`.

    Differenced from the bench's own builders, so later rounds append exactly what the
    first round did.
    """
    key = (bench.__name__, *sorted(f"{k}={v}" for k, v in kwargs.items()))
    cache = _ROUND_TAIL_CACHE.setdefault(key, {})
    if stage in cache:
        return dict(cache[stage])
    if stage == "feedback":
        head = bench.build_messages(payload, "generation", **kwargs)
        full = bench.build_messages(payload, "feedback", draft=_TAIL_PROBE, **kwargs)
    else:
        # A two-stage refinement extends the generation stage, as feedback does.
        before = (
            "generation"
            if kwargs.get("refinement") in MERGED_REFINEMENTS
            else "feedback"
        )
        head = bench.build_messages(payload, before, draft=_TAIL_PROBE, **kwargs)
        full = bench.build_messages(
            payload, "refinement", draft=_TAIL_PROBE, critique=_TAIL_PROBE, **kwargs
        )
    instruction = full[len(head) :][-1]
    cache[stage] = dict(instruction)
    return dict(instruction)


def _stage_messages(
    task: Task,
    stage: str,
    cfg: RunConfig,
    draft: str | None,
    critique: str | None,
    base_messages: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """One stage's messages, shared by every render path.

    `base_messages`, from the round loop's second round on, is the previous stage's
    messages, extended with its generated text and the bench's instruction turn. None
    builds through the bench, matching the fixed pipeline message for message.
    """
    base = base_stage(stage)
    bench = load_bench(cfg.bench_root)
    if base_messages is not None:
        reads_critique = base == "refinement" and not cfg.two_stage
        generated = critique if reads_critique else draft
        instruction = _round_instruction(
            bench, task.as_prompt_task(), base, bench_stage_kwargs(cfg)
        )
        return [
            *base_messages,
            {"role": "assistant", "content": generated},
            instruction,
        ]
    try:
        messages = bench.build_messages(
            task.as_prompt_task(),
            base,
            draft=draft,
            critique=critique,
            **bench_stage_kwargs(cfg),
        )
    except (KeyError, ValueError) as exc:
        raise PromptError(f"prompt layer rejected stage {stage!r}: {exc}") from exc

    # A sample may override the system message per stage (`stage_systems`).
    stage_systems = task.extras.get("stage_systems")
    if stage_systems:
        if base not in stage_systems:
            raise PromptError(f"stage_systems carries no entry for stage {stage!r}")
        messages = [
            {"role": "system", "content": stage_systems[base]},
            *messages[1:],
        ]
    return messages


def _template_ids(tokenizer: Any, messages: Any, cfg: RunConfig) -> list[int]:
    # transformers 5.x defaults return_dict to True, and iterating the BatchEncoding
    # would silently yield its two key names as the prompt.
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        **{cfg.thinking_kwarg: cfg.enable_thinking},
    )
    return list(token_ids)


def _template_text(tokenizer: Any, messages: Any, cfg: RunConfig) -> str | None:
    """The same template as a string, or None from a tokenizer that only tokenizes."""
    try:
        # The thinking switch `_template_ids` passes, so the text matches the ids.
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **{cfg.thinking_kwarg: cfg.enable_thinking},
        )
    except TypeError:
        return None
    return text if isinstance(text, str) else None


def _encode(tokenizer: Any, text: str) -> list[int] | None:
    """`text` as ids with nothing added around it, or None if it cannot be."""
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        return None
    try:
        ids = encode(text, add_special_tokens=False)
    except TypeError:
        return None
    if not isinstance(ids, list) or not all(isinstance(t, int) for t in ids):
        return None
    return ids


def _common_prefix_len(a: Any, b: Any) -> int:
    """Common prefix length, by binary search on slices rather than a per-item loop."""
    hi = min(len(a), len(b))
    if a[:hi] == b[:hi]:
        return hi
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _splice_boundary(tokenizer: Any, prev_text: str, upto: int) -> int | None:
    """Where the last added token at or before `upto` starts, or None.

    `upto` is how much of `prev_text` the next template repeats. None when the tokenizer
    declares no added tokens or the nearest is more than `_MAX_SPLICE_TAIL_CHARS` from
    the end.
    """
    markers: list[str] = []
    for attr in ("all_special_tokens", "additional_special_tokens"):
        value = getattr(tokenizer, attr, None) or ()
        if isinstance(value, str):
            continue
        markers.extend(m for m in value if isinstance(m, str) and m)
    if not markers:
        return None
    head = prev_text[:upto]
    at = max(head.rfind(marker) for marker in markers)
    if at < 0 or len(prev_text) - at > _MAX_SPLICE_TAIL_CHARS:
        return None
    return at


def _splice(
    tokenizer: Any, previous: Rendered, text: str
) -> tuple[list[int], int] | None:
    """`(ids, prefix_shortfall)` with only what this stage appends encoded, or None.

    Spliced at the last added token inside the common prefix, so a disagreement at the
    previous prompt's end is re-encoded. None, and a full encode, when a check fails;
    the key one is that the tail past the boundary re-encodes to the ids `previous`
    holds.
    """
    prev_text = previous.text
    if prev_text is None or not previous.token_ids:
        return None
    common = _common_prefix_len(prev_text, text)
    if common == len(text):
        return None
    at = _splice_boundary(tokenizer, prev_text, common)
    if at is None:
        return None
    tail = _encode(tokenizer, prev_text[at:])
    if not tail or len(tail) >= len(previous.token_ids):
        return None
    if previous.token_ids[-len(tail) :] != tail:
        return None
    rest = _encode(tokenizer, text[at:])
    if rest is None:
        return None
    head = previous.token_ids[: len(previous.token_ids) - len(tail)]
    return head + rest, len(tail) - _common_prefix_len(tail, rest)


def render(
    tokenizer: Any,
    task: Task,
    stage: str,
    cfg: RunConfig,
    draft: str | None = None,
    critique: str | None = None,
    base_messages: list[dict[str, str]] | None = None,
) -> Rendered:
    """One stage's prompt, fully encoded; `base_messages` as in `_stage_messages`."""
    messages = _stage_messages(
        task, stage, cfg, draft, critique, base_messages=base_messages
    )
    return Rendered(
        stage=stage,
        token_ids=_template_ids(tokenizer, messages, cfg),
        messages=messages,
    )


def render_incremental(
    tokenizer: Any,
    task: Task,
    stage: str,
    cfg: RunConfig,
    previous: Rendered | None = None,
    draft: str | None = None,
    critique: str | None = None,
    nest: bool = False,
) -> Rendered:
    """`render`, encoding only what this stage appends to `previous`.

    Under layout `requote` a later stage is `render_requote`'s.

    Args:
        previous: The previous stage's render; None on a program's first stage.
        draft: The generation stage's output; the latest revision from round two on.
        nest: Extend `previous.messages` instead of building through the bench. The
            round loop needs it from round two on: the bench builders take one draft
            and one critique, so rebuilding would drop every turn since the first
            revision.
    """
    if previous is not None and cfg.layout == PROMPT_LAYOUT_REQUOTE:
        return render_requote(
            tokenizer,
            task,
            stage,
            cfg,
            previous,
            draft=draft,
            critique=critique,
        )
    messages = _stage_messages(
        task,
        stage,
        cfg,
        draft,
        critique,
        base_messages=previous.messages if nest else None,
    )
    text = _template_text(tokenizer, messages, cfg)
    outcome = (
        _splice(tokenizer, previous, text)
        if previous is not None and text is not None
        else None
    )
    if outcome is not None:
        spliced, shortfall = outcome
        return Rendered(
            stage=stage,
            token_ids=spliced,
            messages=messages,
            text=text,
            prefix_shortfall=shortfall,
        )
    full = _template_ids(tokenizer, messages, cfg)
    shortfall = (
        None
        if previous is None
        else len(previous.token_ids) - _common_prefix_len(previous.token_ids, full)
    )
    return Rendered(
        stage=stage,
        token_ids=full,
        messages=messages,
        text=text,
        prefix_shortfall=shortfall,
    )


JOIN_TOKENS = "tokens"
JOIN_TEXT = "text"
JOIN_REQUOTE = "requote"

# Stands in for a generated turn so the template's next-turn text can be cut out.
_TURN_SENTINEL = "REFLECTION_GENERATED_TURN_7f3c1a"


def _next_turn_ids(
    tokenizer: Any, previous: Rendered, messages: list[dict[str, str]], cfg: RunConfig
) -> list[int] | None:
    """The ids the template writes after `previous`'s generated turn, or None.

    Read off a render with a sentinel as the assistant content. None unless `messages`
    is `previous.messages` plus an assistant turn and one more message, and the render
    before the sentinel extends `previous.text`: a template that rewrites history would
    contradict a token join. What it adds to `previous.text` (an empty think block, say)
    the generated ids stand in for, as for the content.
    """
    n = len(previous.messages)
    if (
        previous.text is None
        or len(messages) != n + 2
        or messages[:n] != previous.messages
        or messages[n].get("role") != "assistant"
    ):
        return None
    sentinel = {"role": "assistant", "content": _TURN_SENTINEL}
    probe = _template_text(tokenizer, [*previous.messages, sentinel, messages[-1]], cfg)
    if probe is None or probe.count(_TURN_SENTINEL) != 1:
        return None
    head, tail = probe.split(_TURN_SENTINEL)
    if not head.startswith(previous.text):
        return None
    return _encode(tokenizer, tail) or None


def _requote_bench(cfg: RunConfig) -> ModuleType:
    """The bench, once it says it builds `requote`."""
    bench = load_bench(cfg.bench_root)
    if PROMPT_LAYOUT_REQUOTE not in getattr(bench, "LAYOUTS", ()):
        raise PromptError(
            f"layout is 'requote' but the bench under {cfg.bench_root} does not "
            "list it in LAYOUTS, so it has no quoted sections to build"
        )
    return bench


def _quote_round(cfg: RunConfig, stage: str) -> int | None:
    """The quoted section's round; None, as in the fixed pipeline, below two rounds."""
    if cfg.pipeline_mode != PIPELINE_MODE_ROUNDS or cfg.max_rounds < 2:
        return None
    return stage_round(stage)


def _required_encode(tokenizer: Any, text: str) -> list[int]:
    ids = _encode(tokenizer, text)
    if ids is None:
        raise PromptError(
            "layout 'requote' encodes its headers and instructions on their "
            "own, and this tokenizer has no plain encode"
        )
    return ids


def _user_close(
    tokenizer: Any, before: list[dict[str, str]], cfg: RunConfig
) -> tuple[str, list[int]]:
    """The template text before a last user message, and the ids after it.

    Read off a sentinel render, so the close is the template's own for any family.
    """
    probe = _template_text(
        tokenizer, [*before, {"role": "user", "content": _TURN_SENTINEL}], cfg
    )
    if probe is None or probe.count(_TURN_SENTINEL) != 1:
        raise PromptError(
            "layout 'requote' reads the template's close off a text render, "
            "and this tokenizer renders none"
        )
    head, tail = probe.split(_TURN_SENTINEL)
    return head, _required_encode(tokenizer, tail)


def _task_ids(
    tokenizer: Any,
    generation: Rendered,
    task_part: str,
    head_text: str,
    close: list[int],
) -> list[int]:
    """The generation stage's ids through its task.

    Its ids less its ask and the close when those match its tail; otherwise a full
    encode, differing at most at the task's end (the next `prefix_shortfall`).
    """
    user = generation.messages[-1]["content"] if generation.messages else ""
    if user.startswith(task_part):
        tail = [*_required_encode(tokenizer, user[len(task_part) :]), *close]
        n = generation.n_tokens
        if len(tail) < n and generation.token_ids[n - len(tail) :] == tail:
            return generation.token_ids[: n - len(tail)]
    return _required_encode(tokenizer, head_text + task_part)


def _quoted_ids(
    tokenizer: Any, generated_ids: list[int] | None, text: str
) -> list[int]:
    """An output's generated ids less trailing special tokens, else its text encoded."""
    if not generated_ids:
        return _required_encode(tokenizer, text)
    special = set(getattr(tokenizer, "all_special_ids", None) or ())
    ids = list(generated_ids)
    while ids and ids[-1] in special:
        ids.pop()
    return ids


def render_requote(
    tokenizer: Any,
    task: Task,
    stage: str,
    cfg: RunConfig,
    previous: Rendered,
    generated_ids: list[int] | None = None,
    draft: str | None = None,
    critique: str | None = None,
) -> Rendered:
    """A later stage's prompt under layout `requote`, built from id segments.

    The generation prompt through its task, each quoted section (header, the quoted
    stage's generated ids, separator), then the instruction and the template's close. A
    feedback stage quotes the newest draft after every section of the stage before it
    (after the task alone in round one); a refinement stage quotes its critique after
    its feedback stage's sections. A stage thus matches the stage before through
    `body_end`, or the generation prompt through its task, and appends one output and
    an instruction; only short segments are encoded, at any context length.

    Args:
        generated_ids: The ids `previous` generated; None quotes the encoded text
            (stray reasoning was stripped).

    Raises:
        PromptError: On a bench that does not build `requote`, or on a tokenizer with
            no text render or plain encode.
    """
    base = base_stage(stage)
    bench = _requote_bench(cfg)
    payload = task.as_prompt_task()
    # A two-stage refinement stage quotes the newest draft, as a feedback stage does.
    quotes_draft = base == "feedback" or cfg.two_stage
    kind, text = ("draft", draft) if quotes_draft else ("critique", critique)
    after_quotes = previous.frame_end is not None and previous.body_end is not None
    from_task = quotes_draft and not after_quotes
    header = bench.quote_header(kind, _quote_round(cfg, stage))
    quoted = ((header, text),) if from_task else (*previous.quoted, (header, text))
    try:
        messages = bench.build_messages(
            payload,
            base,
            draft=draft,
            critique=critique,
            quoted=quoted,
            **bench_stage_kwargs(cfg),
        )
    except (KeyError, ValueError) as exc:
        raise PromptError(f"prompt layer rejected stage {stage!r}: {exc}") from exc
    task_part = bench.requote_user(payload, (), "")
    sections = "".join(f"{h}{t}{bench.QUOTE_SEP}" for h, t in quoted)
    user = messages[-1]["content"]
    instruction = user[len(task_part) + len(sections) :]
    head_text, close = _user_close(tokenizer, messages[:-1], cfg)
    if from_task:
        head = _task_ids(tokenizer, previous, task_part, head_text, close)
    else:
        head = previous.token_ids[: previous.body_end]
    frame_end = len(head) if from_task else previous.frame_end
    body = [
        *head,
        *_required_encode(tokenizer, header),
        *_quoted_ids(tokenizer, generated_ids, text),
        *_required_encode(tokenizer, bench.QUOTE_SEP),
    ]
    ids = [*body, *_required_encode(tokenizer, instruction), *close]
    shortfall = max(0, len(head) - _common_prefix_len(previous.token_ids, ids))
    return Rendered(
        stage=stage,
        token_ids=ids,
        messages=messages,
        prefix_shortfall=shortfall,
        join=JOIN_REQUOTE,
        frame_end=frame_end,
        body_end=len(body),
        quoted=quoted,
    )


def render_successor(
    tokenizer: Any,
    task: Task,
    stage: str,
    cfg: RunConfig,
    previous: Rendered | None = None,
    generated_ids: list[int] | None = None,
    draft: str | None = None,
    critique: str | None = None,
    nest: bool = False,
) -> Rendered:
    """This stage's prompt as the previous prompt's ids plus what it generated.

    The ids are `previous.token_ids + generated_ids + next-turn ids`, the next turn's
    first token dropped when the generation already ends with it (an end-of-turn stop).
    Other arguments are as for `render_incremental`.

    Returns:
        The prompt with `join` set to `JOIN_TOKENS`; `render_incremental`'s with
        `JOIN_TEXT` when the template or messages do not allow a token join; a first
        stage's unchanged; under layout `requote`, `render_requote`'s.
    """
    if previous is not None and cfg.layout == PROMPT_LAYOUT_REQUOTE:
        return render_requote(
            tokenizer,
            task,
            stage,
            cfg,
            previous,
            generated_ids=generated_ids,
            draft=draft,
            critique=critique,
        )
    if previous is None or not generated_ids:
        return render_incremental(
            tokenizer,
            task,
            stage,
            cfg,
            previous=previous,
            draft=draft,
            critique=critique,
            nest=nest,
        )
    messages = _stage_messages(
        task,
        stage,
        cfg,
        draft,
        critique,
        base_messages=previous.messages if nest else None,
    )
    tail = _next_turn_ids(tokenizer, previous, messages, cfg)
    if tail is None:
        rendered = render_incremental(
            tokenizer,
            task,
            stage,
            cfg,
            previous=previous,
            draft=draft,
            critique=critique,
            nest=nest,
        )
        return dataclasses.replace(rendered, join=JOIN_TEXT)
    overlap = int(generated_ids[-1] == tail[0])
    return Rendered(
        stage=stage,
        token_ids=[*previous.token_ids, *generated_ids, *tail[overlap:]],
        messages=messages,
        text=_template_text(tokenizer, messages, cfg),
        prefix_shortfall=0,
        join=JOIN_TOKENS,
    )


def join_record(rendered: Rendered, history: Sequence[int] | None) -> dict[str, Any]:
    """How a stage's prompt continues the stage before, for the stage's record.

    Args:
        history: The previous stage's prompt and every id it generated; None on a
            program's first stage.

    Returns:
        `{}` on a first stage; otherwise `join` (None, a text re-render after stripped
        reasoning, records as `JOIN_TEXT`), `prefix_shortfall`, `matched_tokens` (the
        matched prefix with `history`) and `append_tokens` (the rest), plus the
        divergence point `join_lcp_tokens` for any join but `JOIN_TOKENS`.
    """
    if history is None:
        return {}
    join = rendered.join or JOIN_TEXT
    matched = _common_prefix_len(history, rendered.token_ids)
    record: dict[str, Any] = {
        "join": join,
        "prefix_shortfall": rendered.prefix_shortfall,
        "matched_tokens": matched,
        "append_tokens": rendered.n_tokens - matched,
    }
    if join != JOIN_TOKENS:
        record["join_lcp_tokens"] = matched
    return record


def filler_ids(tokenizer: Any, n_tokens: int) -> list[int]:
    """Exactly `n_tokens` stand-in generated ids, for pricing a later stage."""
    unit = tokenizer.encode(
        "The scene opens on a wide shot of the mill floor.\n",
        add_special_tokens=False,
    )
    return (unit * (n_tokens // len(unit) + 1))[:n_tokens]

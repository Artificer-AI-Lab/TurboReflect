# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""LongBench-v2 as reflection programs: build the tasks, export the answers.

    python -m turboreflect.benchmarks.longbench_v2 build --model <model> \\
        --out tasks/longbench_v2
    python -m turboreflect.benchmarks.longbench_v2 export runs/longbench_v2 \\
        --out preds/longbench_v2

The task `x` is LongBench-v2's zero-shot prompt (`prompts/0shot.txt` of the LongBench
repository), unaltered. A document is middle-truncated with LongBench-v2's own cut
until the rendered generation prompt fits the target; unlike LongBench-v2, only the
document is cut, never the question or the answer format.

`build` cuts on the model's own tokenizer, so build once per model. It writes
`data/longbench_<id>_32k.json`, `rubrics/longbench_<id>.json` (the gold letter and
metadata) and `prompts.py`. `export` writes `<out>/generation.jsonl` and
`<out>/refinement.jsonl` with the fields LongBench's `result.py` reads.
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
from typing import Any

import regex as re

from turboreflect.benchmarks.common import (
    CONTEXT_SLOT,
    EXPORTED_STAGES,
    fit_document,
    generation_counter,
    read_rubric,
    read_run,
    task_folder,
    write_json,
    write_jsonl,
)

HF_DATASET = "THUDM/LongBench-v2"
HF_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
HF_FILE = "data.json"

TIER = "32k"
LETTERS = "ABCD"
RUBRIC_META = ("domain", "sub_domain", "difficulty", "length")

# The rendered generation-prompt length a document is cut to, in the model's tokens.
DEFAULT_TARGET = 32000

# `prompts/0shot.txt` of the LongBench repository, byte for byte; `$DOC$` becomes the
# stage assembly's `{context}` slot. Adapted from https://github.com/THUDM/LongBench
# (MIT), see THIRD_PARTY_NOTICES.md.
TEMPLATE = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n$DOC$\n</text>\n\n"
    "What is the correct answer to this question: $Q$\n"
    "Choices:\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\n\n"
    'Format your response as follows: "The correct answer is (insert answer here)".'
)


def sample_id(item_id: str) -> str:
    return f"longbench_{item_id}_{TIER}"


def rubric_id(item_id: str) -> str:
    return f"longbench_{item_id}"


def load_source(source: str | None) -> list[dict[str, Any]]:
    """The items of `source`, a local `data.json`, else of the pinned revision."""
    if source is None:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        source = hf_hub_download(
            HF_DATASET, HF_FILE, repo_type="dataset", revision=HF_REVISION
        )
    return json.loads(Path(source).read_text(encoding="utf-8"))


def user_task(row: dict[str, Any]) -> str:
    """The template filled the way LongBench-v2's `pred.py` fills it, but `$DOC$`."""
    text = TEMPLATE.replace("$DOC$", CONTEXT_SLOT).replace(
        "$Q$", str(row["question"]).strip()
    )
    for x in LETTERS:
        text = text.replace(f"$C_{x}$", str(row[f"choice_{x}"]).strip())
    return text


def make_task(row: dict[str, Any], document: str, truncated: bool) -> dict[str, Any]:
    """What the model sees. The gold letter is in the rubric, never here."""
    return {
        "id": sample_id(row["_id"]),
        "system": "",
        "user_task": user_task(row),
        "context": {"document": document},
        "rubric": [],
        "source": {
            "bench": "longbench-v2",
            "prompt": "0shot",
            "tier": TIER,
            "truncated": truncated,
        },
    }


def build(args: argparse.Namespace) -> None:
    """Write the task folder `args.out`."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    root = Path(args.out)
    data_dir, rubric_dir = task_folder(root)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    target = args.target
    count = generation_counter(tokenizer, root)
    rows = load_source(args.source)[: args.limit]
    lengths = []
    for row in rows:
        text = str(row["context"]).strip()
        encoding = tokenizer.backend_tokenizer.encode(text, add_special_tokens=False)
        make = functools.partial(make_task, row)
        task, n = fit_document(make, text, encoding.offsets, target, count)
        lengths.append(n)
        write_json(data_dir / f"{task['id']}.json", task)
        write_json(
            rubric_dir / f"{rubric_id(row['_id'])}.json",
            {"answer": row["answer"], **{k: row[k] for k in RUBRIC_META}},
        )
    print(
        f"{len(rows)} tasks, generation prompts up to {max(lengths)} tokens "
        f"(target {target}) -> {data_dir}"
    )


def extract_answer(response):
    """`extract_answer` of `pred.py`, unchanged.

    From https://github.com/THUDM/LongBench, MIT License, Copyright (c) 2023 THU-KEG &
    Zhipu AI; see THIRD_PARTY_NOTICES.md.
    """
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    else:
        match = re.search(r"The correct answer is ([A-D])", response)
        if match:
            return match.group(1)
        else:
            return None


def export(args: argparse.Namespace) -> None:
    """Write each stage's answers as `{out}/{stage}.jsonl`, for `result.py`."""
    root, answers = read_run(args.run, args.tasks)
    for stage in EXPORTED_STAGES:
        rows = []
        for task_id in sorted(answers):
            item_id = task_id.removeprefix("longbench_").removesuffix(f"_{TIER}")
            rubric = read_rubric(root, rubric_id(item_id))
            response = answers[task_id][stage]
            pred = extract_answer(response)
            rows.append(
                {
                    "_id": item_id,
                    **{k: rubric[k] for k in RUBRIC_META},
                    "response": response,
                    "pred": pred,
                    "answer": rubric["answer"],
                    "judge": pred == rubric["answer"],
                }
            )
        write_jsonl(Path(args.out) / f"{stage}.jsonl", rows)
    print(f"{len(answers)} programs -> {args.out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m turboreflect.benchmarks.longbench_v2"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    make = commands.add_parser("build", help="write a task folder")
    make.add_argument("--model", required=True, help="the model the tasks run on")
    make.add_argument("--out", required=True, help="the task folder")
    make.add_argument("--source", help=f"a local {HF_FILE}; else the pinned revision")
    make.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help=f"the generation-prompt length to cut to (default {DEFAULT_TARGET})",
    )
    make.add_argument("--limit", type=int, help="build only the first LIMIT items")
    make.set_defaults(handler=build)

    write = commands.add_parser("export", help="write a run's answers for result.py")
    write.add_argument("run", help="a run directory, or a folder of them")
    write.add_argument("--out", required=True, help="the prediction folder")
    write.add_argument("--tasks", help="the task folder, if it has moved since the run")
    write.set_defaults(handler=export)

    args = parser.parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

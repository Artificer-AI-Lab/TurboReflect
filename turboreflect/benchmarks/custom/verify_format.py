# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""Check task files against the format a run reads.

    python -m turboreflect.benchmarks.custom.verify_format my_tasks/

Finds and reads task files as a run does, and checks that task ids are unique and
that every clip is on disk.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from turboreflect.config import ConfigError, task_files
from turboreflect.program.task import SampleError, load_task


def check(tasks: str) -> tuple[int, list[str]]:
    """How many task files `tasks` names, and what is wrong with them.

    Raises:
        ConfigError: If `tasks` names no task file.
    """
    files = task_files(tasks)
    problems: list[str] = []
    ids: Counter[str] = Counter()
    for file in files:
        try:
            task = load_task(file)
        except SampleError as exc:
            problems.append(str(exc))
            continue
        ids[task.id] += 1
        problems += [
            f"{file}: clip does not exist: {clip}"
            for clip in task.videos
            if not Path(clip).is_file()
        ]
    problems += [
        f"task id {id_!r} is used {n} times" for id_, n in ids.items() if n > 1
    ]
    return len(files), problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m turboreflect.benchmarks.custom.verify_format"
    )
    parser.add_argument("tasks", help="a task file, or a folder of them")
    args = parser.parse_args(argv)
    try:
        count, problems = check(args.tasks)
    except ConfigError as exc:
        parser.error(str(exc))
    for problem in problems:
        print(problem)
    print(f"{count} task files, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

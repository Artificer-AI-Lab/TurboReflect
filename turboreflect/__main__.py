# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""The command line: `calibrate`, `run` and `report`, each taking `--help`.

A command is imported only once chosen, so vLLM is not imported before a P/D run's
workers pin their devices (`locality_pd.driver`).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m turboreflect",
        description="Accelerate batches of reflection programs on vLLM.",
    )
    parser.add_argument(
        "command",
        choices=("run", "calibrate", "report"),
        help="run a configuration, calibrate its deployment, or report on run "
        "directories; each takes --help",
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.command == "run":
        from turboreflect.engine.runner import main as command
    elif args.command == "calibrate":
        from turboreflect.calibration.calibrate import main as command
    else:
        from turboreflect.metrics.report import main as command
    return command(args.arguments)


if __name__ == "__main__":
    sys.exit(main())

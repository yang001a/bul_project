"""Cross-platform one-command orchestration for the complete paper workflow.

This wrapper invokes existing public commands in their required order. It
contains no scientific calculation and does not alter any frozen parameter.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from bul_project.config import DATASET_ROOT, PROJECT_ROOT


def _parser() -> argparse.ArgumentParser:
    """Build the complete-workflow command parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce parameter artifacts, the 3150-condition main experiment, "
            "derived BUL results, sensitivity results, and paper statistics."
        )
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip unit and real-XES integration tests before reproduction.",
    )
    parser.add_argument(
        "--include-external-validation",
        action="store_true",
        help=(
            "Also execute the optional representative PM4Py feasibility "
            "validation before statistical analysis."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Directory containing the three user-supplied XES files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display the command sequence without executing it.",
    )
    return parser


def _workflow_commands(
    *,
    skip_tests: bool,
    include_external_validation: bool,
) -> list[list[str]]:
    """Return the frozen workflow command sequence."""

    commands: list[list[str]] = []
    if not skip_tests:
        commands.extend(
            [
                [sys.executable, "-m", "pytest", "-q", "-m", "not integration"],
                [sys.executable, "-m", "pytest", "-q", "-m", "integration"],
            ]
        )
    main = [sys.executable, str(PROJECT_ROOT / "main.py")]
    commands.extend(
        [
            main
            + [
                "freeze-parameters",
                "--output-dir",
                "configs/generated",
            ],
            main + ["run-main", "--plan-only"],
            main + ["run-main", "--execute-numeric-reference"],
            main + ["derive-bul"],
            main + ["run-sensitivity", "--execute"],
        ]
    )
    if include_external_validation:
        commands.append(main + ["run-external-validation", "--execute"])
    commands.append(main + ["analyze"])
    commands.append(main + ["build-manifest"])
    return commands


def main(argv: list[str] | None = None) -> int:
    """Execute or display the complete reproducibility workflow."""

    args = _parser().parse_args(argv)
    dataset_root = args.dataset_root.expanduser().resolve()
    commands = _workflow_commands(
        skip_tests=args.skip_tests,
        include_external_validation=args.include_external_validation,
    )
    logger = logging.getLogger(__name__)
    for command in commands:
        logger.info("Workflow command: %s", " ".join(command))
    if args.dry_run:
        return 0

    environment = os.environ.copy()
    environment["BUL_DATASET_ROOT"] = str(dataset_root)
    for index, command in enumerate(commands):
        command_environment = environment.copy()
        if (
            not args.skip_tests
            and index == 1
            and command[1:4] == ["-m", "pytest", "-q"]
        ):
            command_environment["RUN_REAL_XES_TESTS"] = "1"
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=command_environment,
            check=False,
        )
        if completed.returncode != 0:
            logger.error(
                "Workflow stopped with exit code %d: %s",
                completed.returncode,
                " ".join(command),
            )
            return int(completed.returncode)
    logger.info("Complete BUL paper workflow finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

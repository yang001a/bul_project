"""CLI for deterministic, read-only BUL result derivation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from bul_project.config import (
    FROZEN_PARAMETER_DIR,
    MAIN_EXPERIMENT_RESULT_DIR,
    REPRODUCED_DERIVED_ANALYSIS_DIR,
)
from bul_project.experiments.bul_analysis import (
    DerivedAnalysisError,
    run_derived_analysis,
)
from bul_project.logging_utils import configure_logging


def _parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for this entry point.

    Inputs:
        None.
    Outputs:
        A value of type `argparse.ArgumentParser` produced by the operation.
    Paper role:
        Release command-line orchestration and reproducible execution.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Derive fixed-entropy BUL, dimension ablation, and cross-dataset "
            "summaries from the completed numerical reference without rerunning DP."
        )
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=MAIN_EXPERIMENT_RESULT_DIR,
    )
    parser.add_argument(
        "--parameter-dir",
        dest="parameter_dir",
        type=Path,
        default=FROZEN_PARAMETER_DIR,
        help="Frozen parameter-artifact directory.",
    )
    parser.add_argument(
        "--phase7c-dir",
        dest="parameter_dir",
        type=Path,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPRODUCED_DERIVED_ANALYSIS_DIR,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the release command-line workflow and return its exit status.

    Inputs:
        argv: Input value used by this operation.
    Outputs:
        A value of type `int` produced by the operation.
    Paper role:
        Release command-line orchestration and reproducible execution.
    """
    configure_logging("derive_bul_results")
    args = _parser().parse_args(argv)
    try:
        manifest = run_derived_analysis(
            args.reference_dir,
            args.parameter_dir,
            args.output_dir,
        )
        logging.info(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except DerivedAnalysisError as exc:
        logging.info(
            json.dumps(
                {
                    "original_phase": 6,
                    "status": "blocked",
                    "error_type": "DerivedAnalysisError",
                    "error_message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

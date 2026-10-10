"""CLI for authenticated statistical analysis and paper figures."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from bul_project.config import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DERIVED_ANALYSIS_RESULT_DIR,
    EXTERNAL_VALIDATION_RESULT_DIR,
    MAIN_EXPERIMENT_RESULT_DIR,
    REPRODUCED_STATISTICAL_ANALYSIS_DIR,
    SENSITIVITY_ANALYSIS_RESULT_DIR,
)
from bul_project.logging_utils import configure_logging
from bul_project.statistics.analysis_runner import (
    StatisticalAnalysisError,
    run_statistical_analysis,
)


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
            "Run Wilcoxon/Friedman/Holm, bootstrap confidence intervals, paired "
            "effect sizes, exact-permutation Spearman trends, and publication-ready "
            "PNG/PDF paper plots from frozen experiment tables."
        )
    )
    parser.add_argument(
        "--derived-dir",
        type=Path,
        default=DERIVED_ANALYSIS_RESULT_DIR,
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=MAIN_EXPERIMENT_RESULT_DIR,
    )
    parser.add_argument(
        "--sensitivity-dir",
        type=Path,
        default=SENSITIVITY_ANALYSIS_RESULT_DIR,
    )
    parser.add_argument(
        "--external-dir",
        type=Path,
        default=EXTERNAL_VALIDATION_RESULT_DIR,
        help=(
            "Optional directory containing representative PM4Py feasibility "
            "validation outputs. Its absence does not block BUL statistics."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPRODUCED_STATISTICAL_ANALYSIS_DIR,
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
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
    configure_logging("run_statistical_analysis")
    args = _parser().parse_args(argv)
    try:
        manifest = run_statistical_analysis(
            derived_dir=args.derived_dir,
            reference_dir=args.reference_dir,
            sensitivity_dir=args.sensitivity_dir,
            external_dir=args.external_dir,
            output_dir=args.output_dir,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        logging.info(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except StatisticalAnalysisError as exc:
        logging.info(
            json.dumps(
                {
                    "original_phase": 7,
                    "status": "blocked",
                    "error_type": "StatisticalAnalysisError",
                    "error_message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

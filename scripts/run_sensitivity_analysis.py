"""CLI for the BPIC2017 Top-k and edge-threshold sensitivity experiment."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

from bul_project.config import (
    DATASETS,
    DERIVED_ANALYSIS_RESULT_DIR,
    FROZEN_CONFIG_PATH,
    MAIN_EXPERIMENT_RESULT_DIR,
    REPRODUCED_SENSITIVITY_ANALYSIS_DIR,
)
from bul_project.experiments.sensitivity import (
    SensitivityAnalysisError,
    SensitivityExperimentRunner,
    load_sensitivity_sources,
)
from bul_project.logging_utils import configure_logging
from bul_project.preprocessing import preprocess_dataset


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
            "Plan or resume the compact BPIC2017 Top-k x edge-tau BUL "
            "sensitivity experiment."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=FROZEN_CONFIG_PATH,
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=MAIN_EXPERIMENT_RESULT_DIR,
    )
    parser.add_argument(
        "--derived-dir",
        type=Path,
        default=DERIVED_ANALYSIS_RESULT_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPRODUCED_SENSITIVITY_ANALYSIS_DIR,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Read BPIC2017 once and execute/resume unfinished new conditions.",
    )
    parser.add_argument(
        "--max-new-conditions",
        type=int,
        default=None,
        help="Optional positive checkpoint chunk; requires --execute.",
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
    configure_logging("run_sensitivity_analysis")
    args = _parser().parse_args(argv)
    if args.max_new_conditions is not None and not args.execute:
        _parser().error("--max-new-conditions requires --execute")
    try:
        sources = load_sensitivity_sources(
            args.config, args.reference_dir, args.derived_dir
        )
        runner = SensitivityExperimentRunner(sources, args.output_dir)
        if not args.execute:
            logging.info(
                json.dumps(
                    {
                        "original_phase": 6,
                        "status": "planned",
                        "output_dir": str(args.output_dir.resolve()),
                        "execution_plan_sha256": runner.plan_sha256,
                        "expected_conditions": 810,
                        "authenticated_reference_reuse": 90,
                        "new_sensitivity_execution": 720,
                        "remaining_new_conditions": runner.remaining_new_conditions(),
                        "xes_files_read": 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        remaining = runner.remaining_new_conditions()
        traces = None
        if remaining:
            phase1 = preprocess_dataset(DATASETS["BPIC2017"])
            if (
                phase1.source_sha256 != sources.source_sha256
                or phase1.source_path is None
                or Path(phase1.source_path).resolve() != sources.source_path
            ):
                raise SensitivityAnalysisError(
                    "The preprocessed BPIC2017 source changed."
                )
            traces = tuple(phase1.traces)
            del phase1
            gc.collect()
        summary = runner.run(
            traces,
            max_new_conditions=args.max_new_conditions,
        )
        logging.info(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0
    except SensitivityAnalysisError as exc:
        logging.info(
            json.dumps(
                {
                    "original_phase": 6,
                    "status": "blocked",
                    "error_type": "SensitivityAnalysisError",
                    "error_message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

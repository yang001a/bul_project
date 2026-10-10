"""Command-line entry point for the frozen main numerical experiment."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

from bul_project.config import (
    DATASETS,
    FROZEN_CONFIG_PATH,
    LOG_ROOT,
    REPRODUCED_MAIN_EXPERIMENT_DIR,
)
from bul_project.experiments.main_experiment import (
    FormalConditionExecutionError,
    FormalExperimentRunner,
    FormalRunIntegrityError,
    build_formal_execution_plan,
    load_frozen_execution_spec,
    write_formal_execution_plan,
)
from bul_project.experiments.parameter_freeze import DatasetParameterInput
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
            "Validate the frozen parameter bundle, create the main experiment "
            "plan, and optionally execute its numerical conditions with "
            "resumable checkpoints."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=FROZEN_CONFIG_PATH,
        help="Frozen experiment_config_final.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPRODUCED_MAIN_EXPERIMENT_DIR,
        help="Destination for the immutable plan and resumable checkpoints.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Validate and write the plan without reading XES files. This is "
            "also the default when no execution flag is supplied."
        ),
    )
    mode.add_argument(
        "--execute-numeric-reference",
        action="store_true",
        help=(
            "Explicitly read the frozen datasets and execute/resume the "
            "reference numerical matrix. Formal PM4Py evaluation remains "
            "disabled until its condition scope is separately frozen."
        ),
    )
    parser.add_argument(
        "--max-new-conditions",
        type=int,
        default=None,
        help=(
            "Optional positive chunk size for this invocation. It does not "
            "alter the frozen plan and is valid only with execution enabled."
        ),
    )
    return parser


def _configure_logging() -> None:
    """Configure structured logging for the release workflow.

    Inputs:
        None.
    Outputs:
        None.
    Paper role:
        Release command-line orchestration and reproducible execution.
    """
    configure_logging("run_main_experiment", log_dir=LOG_ROOT)


def _load_frozen_inputs(spec) -> tuple[DatasetParameterInput, ...]:
    """Read each frozen Phase 1 dataset exactly once for formal execution.

    Input
    -----
    spec:
        Fully validated frozen execution specification.

    Output
    ------
    tuple[DatasetParameterInput, ...]
        Traces and, where frozen, the complete external evaluation log.

    Paper experiment step
    ---------------------
    Main-experiment input materialization. Source and evaluation fingerprints are
    subsequently rechecked by ``FormalExperimentRunner`` before DP execution.
    """

    frozen_names = set(spec.dataset_names)
    configured_names = set(DATASETS)
    if not frozen_names.issubset(configured_names):
        missing = sorted(frozen_names - configured_names)
        raise FormalRunIntegrityError(
            f"Frozen datasets are absent from DATASETS: {missing!r}"
        )

    inputs: list[DatasetParameterInput] = []
    for item in spec.datasets:
        logging.info("Main experiment loading frozen dataset %s", item.dataset)
        phase1 = preprocess_dataset(DATASETS[item.dataset])
        inputs.append(
            DatasetParameterInput.from_phase1_result(
                phase1,
                include_evaluation_log=(item.evaluation_log is not None),
                evaluation_log_id=(
                    None
                    if item.evaluation_log is None
                    else item.evaluation_log.evaluation_log_id
                ),
            )
        )
        del phase1
        gc.collect()
    return tuple(inputs)


def main(argv: list[str] | None = None) -> int:
    """Run the release command-line workflow and return its exit status.

    Inputs:
        argv: Input value used by this operation.
    Outputs:
        A value of type `int` produced by the operation.
    Paper role:
        Release command-line orchestration and reproducible execution.
    """
    args = _parser().parse_args(argv)
    _configure_logging()
    if args.max_new_conditions is not None and not args.execute_numeric_reference:
        _parser().error(
            "--max-new-conditions requires --execute-numeric-reference"
        )

    try:
        spec = load_frozen_execution_spec(args.config)
        if not args.execute_numeric_reference:
            plan = build_formal_execution_plan(spec)
            plan_path = write_formal_execution_plan(plan, args.output_dir)
            logging.info(
                json.dumps(
                    {
                        "phase": "8A",
                        "status": "planned",
                        "mode": "plan_only",
                        "scenario_id": spec.scenario_id,
                        "output_dir": str(args.output_dir.resolve()),
                        "plan": str(plan_path.resolve()),
                        "plan_sha256": plan.plan_sha256,
                        "counts": plan.to_dict()["counts"],
                        "process_quality_scope_status": (
                            "deferred_not_frozen"
                        ),
                        "sensitivity_pending": True,
                        "xes_files_read": False,
                        "conditions_executed": 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        frozen_inputs = _load_frozen_inputs(spec)
        runner = FormalExperimentRunner(
            spec,
            frozen_inputs,
            args.output_dir,
        )
        summary = runner.run(
            max_new_conditions=args.max_new_conditions,
        )
        logging.info(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0
    except FormalRunIntegrityError as exc:
        logging.error("Main-experiment integrity validation failed: %s", exc)
        logging.info(
            json.dumps(
                {
                    "phase": "8A",
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    except FormalConditionExecutionError as exc:
        logging.exception("Main experiment stopped at a failed condition")
        logging.info(
            json.dumps(
                {
                    "phase": "8A",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "resume_supported": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    except KeyboardInterrupt:
        logging.warning(
            "Main experiment interrupted; committed checkpoints are reusable."
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

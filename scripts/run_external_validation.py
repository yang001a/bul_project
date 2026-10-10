"""CLI for the representative three-condition external-feasibility appendix."""

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
    MAIN_EXPERIMENT_RESULT_DIR,
    REPRODUCED_EXTERNAL_VALIDATION_DIR,
)
from bul_project.experiments.parameter_freeze import build_evaluation_log_metadata
from bul_project.validation.evaluation_sample import (
    EvaluationSampleIntegrityError,
    build_fixed_bpic2017_evaluation_sample,
)
from bul_project.validation.runner import (
    ProcessQualityFormalConditionError,
    ProcessQualityFormalIntegrityError,
    RepresentativeProcessQualityRunner,
    build_process_quality_execution_plan,
    validate_disjoint_directories,
    write_process_quality_execution_plan,
)
from bul_project.validation.scope import (
    ProcessQualityScopeIntegrityError,
    build_process_quality_scope,
    write_process_quality_scope,
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
    reference = MAIN_EXPERIMENT_RESULT_DIR
    parser = argparse.ArgumentParser(
        description=(
            "Freeze and optionally execute the three representative BPIC2017 "
            "PM4Py external-feasibility conditions."
        )
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=reference,
        help="Completed, read-only 3150-condition numerical reference directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=FROZEN_CONFIG_PATH,
        help="Frozen experiment configuration used by the numerical reference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPRODUCED_EXTERNAL_VALIDATION_DIR,
        help="Independent destination; the reference directory is never modified.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan-only",
        action="store_true",
        help="Freeze/validate the scope without reading the BPIC2017 XES file.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Execute or resume the three representative conditions.",
    )
    parser.add_argument(
        "--max-new-conditions",
        type=int,
        default=None,
        help="Optional positive probe/chunk size; it never changes the frozen plan.",
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
    configure_logging("run_external_validation", log_dir=LOG_ROOT)


def _load_evaluation_events(scope_payload):
    """Load and authenticate the frozen BPIC2017 evaluation case sample.

    This function is not called in plan-only mode.  Its event table is passed
    only to ``ProcessModelEvaluator``; DP temporal and synthetic generation
    consume authenticated DP checkpoint metadata through a separate interface.
    The complete canonical log is authenticated first, then the same frozen
    SHA-256-ranked 3000-case sample is selected for every epsilon condition.
    """

    phase1 = preprocess_dataset(DATASETS["BPIC2017"])
    expected_source = scope_payload["scope"]["dataset_source_sha256"]
    if phase1.source_sha256 != expected_source:
        raise ProcessQualityFormalIntegrityError(
            "The BPIC2017 XES source differs from the frozen numerical reference."
        )
    expected_evaluation_source = scope_payload["evaluation_log_source"]
    actual_evaluation_source = build_evaluation_log_metadata(
        phase1.events,
        str(expected_evaluation_source["evaluation_log_id"]),
    ).to_dict()
    if actual_evaluation_source != expected_evaluation_source:
        raise ProcessQualityFormalIntegrityError(
            "The complete BPIC2017 evaluation log differs from its frozen source."
        )
    try:
        events = build_fixed_bpic2017_evaluation_sample(phase1.events)
    except EvaluationSampleIntegrityError as exc:
        raise ProcessQualityFormalIntegrityError(str(exc)) from exc
    del phase1
    gc.collect()
    return events


def main(argv: list[str] | None = None) -> int:
    """Run the release command-line workflow and return its exit status.

    Inputs:
        argv: Input value used by this operation.
    Outputs:
        A value of type `int` produced by the operation.
    Paper role:
        Release command-line orchestration and reproducible execution.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    _configure_logging()
    if args.max_new_conditions is not None and not args.execute:
        parser.error("--max-new-conditions requires --execute")

    reference = args.reference_dir.resolve()
    output = args.output_dir.resolve()
    plan_path = reference / "formal_execution_plan.json"
    manifest_path = reference / "formal_run_manifest.json"
    scope_path = output / "process_quality_scope_freeze.json"

    try:
        reference, output = validate_disjoint_directories(reference, output)
        scope = build_process_quality_scope(
            plan_path,
            manifest_path,
            args.config,
            checkpoint_root=reference / "checkpoints",
        )
        write_process_quality_scope(
            scope,
            scope_path,
            plan_path,
            manifest_path,
            args.config,
            checkpoint_root=reference / "checkpoints",
        )
        execution_plan = build_process_quality_execution_plan(scope)
        written_plan = write_process_quality_execution_plan(
            execution_plan, output
        )

        if not args.execute:
            logging.info(
                json.dumps(
                    {
                        "original_phase": 6,
                        "status": "planned",
                        "mode": "plan_only",
                        "scope": str(scope_path),
                        "scope_sha256": scope.scope_sha256,
                        "execution_plan": str(written_plan),
                        "execution_plan_sha256": execution_plan.plan_sha256,
                        "conditions": scope.to_dict()["conditions"],
                        "phase6_scope_boundary": scope.to_dict()[
                            "phase6_scope_boundary"
                        ],
                        "xes_files_read": False,
                        "conditions_executed": 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        runner = RepresentativeProcessQualityRunner(
            scope,
            reference,
            lambda: _load_evaluation_events(scope.to_dict()),
            output,
            config_path=args.config,
        )
        summary = runner.run(max_new_conditions=args.max_new_conditions)
        logging.info(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0
    except (
        ProcessQualityScopeIntegrityError,
        ProcessQualityFormalIntegrityError,
    ) as exc:
        logging.error("Phase 6 external-validity integrity failure: %s", exc)
        logging.info(
            json.dumps(
                {
                    "original_phase": 6,
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    except ProcessQualityFormalConditionError as exc:
        logging.exception("Phase 6 representative PM4Py condition failed")
        logging.info(
            json.dumps(
                {
                    "original_phase": 6,
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
        logging.warning("Interrupted; committed representative checkpoints remain valid.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

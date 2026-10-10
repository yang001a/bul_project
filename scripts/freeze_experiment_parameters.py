"""Command-line entry point for diagnostics and parameter freezing."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from dataclasses import replace
from pathlib import Path

from bul_project.config import (
    DATASETS,
    EDGE_MASS_DIAGNOSTIC_THRESHOLD,
    INDUCTIVE_MINER_NOISE_THRESHOLD,
    LOG_ROOT,
    PARAMETER_SELECTION_SCOPE,
    PROCESS_CONFORMANCE_VARIANT,
    PROCESS_QUALITY_DATASETS,
    REPRODUCED_PARAMETER_DIR,
    TIME_SAMPLE_DIAGNOSTIC_THRESHOLD,
    VARIANT_COVERAGE_DIAGNOSTIC_THRESHOLD,
)
from bul_project.experiments.parameter_freeze import (
    DatasetParameterInput,
    ParameterFreezePolicy,
    ProcessQualityFreezeSettings,
    analyze_dataset_parameters,
    finalize_experiment_parameters,
    run_parameter_smoke_test,
    select_experiment_parameters,
    write_parameter_artifacts,
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
            "Run real-data diagnostics, smoke validation, and immutable "
            "experiment-parameter freezing."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPRODUCED_PARAMETER_DIR,
        help="Destination for parameter diagnostics and the final config.",
    )
    parser.add_argument(
        "--selection-scope",
        choices=("global", "per_dataset"),
        default=PARAMETER_SELECTION_SCOPE,
        help="Use one shared k/tau or apply the same rule per dataset.",
    )
    parser.add_argument(
        "--variant-coverage-threshold",
        type=float,
        default=VARIANT_COVERAGE_DIAGNOSTIC_THRESHOLD,
        help=(
            "Diagnostic head-Variant coverage threshold; it is reported but "
            "never used to select or block the central reference setting."
        ),
    )
    parser.add_argument(
        "--edge-mass-threshold",
        type=float,
        default=EDGE_MASS_DIAGNOSTIC_THRESHOLD,
    )
    parser.add_argument(
        "--time-sample-threshold",
        type=float,
        default=TIME_SAMPLE_DIAGNOSTIC_THRESHOLD,
    )
    parser.add_argument(
        "--process-quality-datasets",
        nargs="*",
        choices=tuple(DATASETS),
        default=PROCESS_QUALITY_DATASETS,
        help=(
            "Datasets whose complete preprocessed log is frozen for the epsilon=8 "
            "PM4Py smoke condition."
        ),
    )
    parser.add_argument(
        "--conformance-variant",
        choices=("token_based", "alignments"),
        default=PROCESS_CONFORMANCE_VARIANT,
    )
    parser.add_argument(
        "--inductive-noise-threshold",
        type=float,
        default=INDUCTIVE_MINER_NOISE_THRESHOLD,
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help=(
            "Write diagnostics and stop at ready_for_smoke; never write the "
            "final frozen configuration."
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
    configure_logging("freeze_experiment_parameters", log_dir=LOG_ROOT)


def _load_and_analyze(
    dataset_name: str,
    policy: ParameterFreezePolicy,
    quality_datasets: set[str],
):
    """Load and analyze.

    Inputs:
        dataset_name: Canonical dataset identifier.
        policy: Input value used by this operation.
        quality_datasets: Input value used by this operation.
    Outputs:
        The value produced by the documented operation.
    Paper role:
        Release command-line orchestration and reproducible execution.
    """
    logging.info("Parameter diagnostics loading %s", dataset_name)
    phase1 = preprocess_dataset(DATASETS[dataset_name])
    include_evaluation = dataset_name in quality_datasets
    dataset_input = DatasetParameterInput.from_phase1_result(
        phase1,
        include_evaluation_log=include_evaluation,
    )
    analysis = analyze_dataset_parameters(dataset_input, policy)
    logging.info(
        "Parameter diagnostics analyzed %s: cases=%d variants=%d Lmax=%d Tmax=%.6f",
        dataset_name,
        analysis.profile.case_count,
        analysis.profile.clipped_variant_count,
        analysis.profile.l_max,
        analysis.profile.t_max_seconds,
    )
    return analysis


def _load_smoke_inputs(quality_datasets: set[str]):
    """Load smoke inputs.

    Inputs:
        quality_datasets: Input value used by this operation.
    Outputs:
        The value produced by the documented operation.
    Paper role:
        Release command-line orchestration and reproducible execution.
    """
    inputs = []
    for dataset_name in DATASETS:
        logging.info("Parameter smoke validation reload for %s", dataset_name)
        phase1 = preprocess_dataset(DATASETS[dataset_name])
        inputs.append(
            DatasetParameterInput.from_phase1_result(
                phase1,
                include_evaluation_log=(dataset_name in quality_datasets),
            )
        )
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
    policy = ParameterFreezePolicy(
        variant_coverage_threshold=args.variant_coverage_threshold,
        edge_mass_threshold=args.edge_mass_threshold,
        time_sample_threshold=args.time_sample_threshold,
        selection_scope=args.selection_scope,
    )
    quality_datasets = set(args.process_quality_datasets)
    quality = (
        None
        if not quality_datasets
        else ProcessQualityFreezeSettings(
            datasets=tuple(
                name for name in DATASETS if name in quality_datasets
            ),
            conformance_variant=args.conformance_variant,
            noise_threshold=args.inductive_noise_threshold,
        )
    )

    analyses = []
    for dataset_name in DATASETS:
        analyses.append(
            _load_and_analyze(dataset_name, policy, quality_datasets)
        )
        gc.collect()
    preflight = select_experiment_parameters(
        analyses,
        policy,
        process_quality=quality,
    )

    if preflight.status == "blocked" or args.analysis_only:
        artifacts = write_parameter_artifacts(preflight, args.output_dir)
        logging.info(
            json.dumps(
                {
                    "phase": "7C",
                    "status": preflight.status,
                    "blocking_reasons": list(preflight.blocking_reasons),
                    "output_dir": str(args.output_dir.resolve()),
                    "artifacts": artifacts,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2 if preflight.status == "blocked" else 0

    try:
        smoke_inputs = _load_smoke_inputs(quality_datasets)
        _, evidence = run_parameter_smoke_test(
            preflight,
            smoke_inputs,
            args.output_dir / "smoke",
        )
        final = finalize_experiment_parameters(preflight, evidence)
    except Exception as exc:
        failed = replace(
            preflight,
            status="blocked",
            frozen_parameters=(),
            blocking_reasons=(f"Parameter smoke validation failed: {exc}",),
        )
        artifacts = write_parameter_artifacts(failed, args.output_dir)
        logging.exception("Parameter smoke validation failed")
        logging.info(
            json.dumps(
                {
                    "phase": "7C",
                    "status": "blocked",
                    "blocking_reasons": list(failed.blocking_reasons),
                    "output_dir": str(args.output_dir.resolve()),
                    "artifacts": artifacts,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    artifacts = write_parameter_artifacts(final, args.output_dir)
    logging.info(
        json.dumps(
            {
                "phase": "7C",
                "status": final.status,
                "output_dir": str(args.output_dir.resolve()),
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

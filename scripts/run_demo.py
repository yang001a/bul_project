"""Run a small end-to-end BUL demonstration without licensed datasets.

The demo uses the bundled XES log and the existing preprocessing, parameter
diagnostic, single-condition DP, metric, and BUL implementations. Demo-specific
support threshold 1 is used because the bundled log has only three cases; this
does not alter the frozen main-experiment threshold ``tau=10``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from bul_project.config import EPS_LIST, PROJECT_ROOT, RESULTS_ROOT, DatasetSpec
from bul_project.experiments.parameter_freeze import (
    DatasetParameterInput,
    ParameterFreezePolicy,
    analyze_dataset_parameters,
)
from bul_project.experiments.single_run import (
    SingleExperimentConfig,
    run_single_experiment,
    write_single_experiment_artifacts,
)
from bul_project.preprocessing import preprocess_dataset


DEMO_DATASET_PATH = PROJECT_ROOT / "data" / "demo" / "demo_log.xes"
DEMO_OUTPUT_DIR = RESULTS_ROOT / "demo"


def _parser() -> argparse.ArgumentParser:
    """Build the quick-demo command parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the bundled XES through preprocessing, behavioral statistics, "
            "case-level DP release, established losses, and BUL aggregation."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEMO_OUTPUT_DIR,
        help="Destination for the demo CSV and metadata JSON.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        choices=EPS_LIST,
        default=8.0,
        help="Configured privacy budget used by the one-condition demo.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=range(30),
        default=0,
        help="Experiment seed in the frozen 0-29 range.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the deterministic quick demo and write two small artifacts."""

    args = _parser().parse_args(argv)
    spec = DatasetSpec(
        name="DEMO",
        path=DEMO_DATASET_PATH,
        role="quick reproducibility demonstration",
    )
    preprocessed = preprocess_dataset(spec)
    diagnostic_input = DatasetParameterInput.from_phase1_result(
        preprocessed,
        include_evaluation_log=False,
    )
    diagnostic = analyze_dataset_parameters(
        diagnostic_input,
        ParameterFreezePolicy(
            edge_tau_candidates=(1, 5, 10, 20),
            reference_edge_tau=10,
        ),
    )
    normalization_scales = {
        row.edge: row.normalization_factor_seconds
        for row in diagnostic.normalization_factors
    }
    config = SingleExperimentConfig(
        dataset="DEMO",
        method="DP-full",
        epsilon=args.epsilon,
        seed=args.seed,
        l_max=diagnostic.profile.l_max,
        t_max=diagnostic.profile.t_max_seconds,
        top_k_variants=100,
        min_edge_support=1,
        budget_strategy="uniform",
        dp_edge_mode=None,
        evaluate_process_quality=False,
        conformance_variant=None,
        process_model_noise_threshold=None,
    )
    result = run_single_experiment(
        preprocessed.traces,
        config,
        normalization_scales=normalization_scales,
    )
    artifacts = write_single_experiment_artifacts(result, args.output_dir)
    summary = {
        "status": "complete",
        "dataset": "DEMO",
        "source": str(DEMO_DATASET_PATH),
        "events": preprocessed.dataset_report.events,
        "cases": preprocessed.dataset_report.cases,
        "epsilon": args.epsilon,
        "seed": args.seed,
        "main_experiment_parameters_unchanged": True,
        "demo_specific_min_edge_support": 1,
        "BUL": result.total_losses.bul,
        "output_dir": str(args.output_dir.resolve()),
        "artifacts": artifacts,
    }
    logging.getLogger(__name__).info(
        "Quick demo completed: %s",
        json.dumps(summary, ensure_ascii=False, sort_keys=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

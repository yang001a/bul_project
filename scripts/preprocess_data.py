"""Command-line entry point for canonical XES preprocessing."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

from bul_project.config import (
    DATASETS,
    DEFAULT_PREPROCESSING,
    LOG_ROOT,
    PROCESSED_DATA_ROOT,
    get_dataset_spec,
)
from bul_project.logging_utils import configure_logging
from bul_project.preprocessing import preprocess_dataset, write_phase1_artifacts


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
        description="BUL XES ingestion and canonical event-log preprocessing."
    )
    parser.add_argument(
        "dataset",
        choices=tuple(DATASETS),
        help="Configured dataset to preprocess.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: data/processed/<dataset>).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write reports and manifest without the large events/traces artifacts.",
    )
    parser.add_argument(
        "--keep-extensions",
        action="store_true",
        help="Retain non-core XES columns; core modules still use only canonical fields.",
    )
    parser.add_argument(
        "--drop-exact-duplicates",
        action="store_true",
        help="Drop exact duplicate core events. Disabled by default to avoid silent loss.",
    )
    return parser


def _configure_logging(dataset: str) -> None:
    """Configure structured logging for the release workflow.

    Inputs:
        dataset: Dataset identifier or dataset-specific input.
    Outputs:
        None.
    Paper role:
        Release command-line orchestration and reproducible execution.
    """
    configure_logging(f"preprocess_{dataset.lower()}", log_dir=LOG_ROOT)


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
    spec = get_dataset_spec(args.dataset)
    output_dir = args.output_dir or PROCESSED_DATA_ROOT / spec.name
    policy = replace(
        DEFAULT_PREPROCESSING,
        keep_extensions=args.keep_extensions,
        drop_exact_duplicates=args.drop_exact_duplicates,
    )
    _configure_logging(spec.name)

    started = time.perf_counter()
    logging.info("Starting canonical preprocessing for %s", spec.name)
    result = preprocess_dataset(spec, policy)
    artifacts = write_phase1_artifacts(
        result,
        output_dir,
        include_events=not args.report_only,
        include_traces=not args.report_only,
    )
    elapsed = time.perf_counter() - started
    summary = {
        "dataset": spec.name,
        "output_dir": str(output_dir.resolve()),
        "events": result.dataset_report.events,
        "cases": result.dataset_report.cases,
        "activities": result.dataset_report.activities,
        "runtime_seconds": elapsed,
        "artifacts": artifacts,
    }
    logging.info("Preprocessing completed for %s in %.3f seconds", spec.name, elapsed)
    logging.info(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

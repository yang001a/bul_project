"""High-level orchestration and artifact writing for Phase 1 only."""

from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import pandas as pd

from bul_project.config import CORE_COLUMNS, DEFAULT_PREPROCESSING, DatasetSpec, PreprocessingConfig
from .cleaning import normalize_and_clean
from .dataset_report import build_dataset_report
from .models import Phase1Result
from .traces import build_traces
from .xes_reader import read_xes_dataframe, sha256_file


Reader = Callable[..., pd.DataFrame]


class EmptyEventLogError(ValueError):
    """Raised when no valid core events remain after Phase 1 cleaning."""


def _package_version(package: str) -> str | None:
    """Return the installed version of a Python package when available.

    Inputs:
        package: Input value used by this operation.
    Outputs:
        A value of type `str | None` produced by the operation.
    Paper role:
        Event-log ingestion, cleaning, and trace construction.
    """
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def preprocess_dataset(
    dataset: DatasetSpec,
    policy: PreprocessingConfig = DEFAULT_PREPROCESSING,
    *,
    reader: Reader = read_xes_dataframe,
) -> Phase1Result:
    """Execute XES read, canonical cleaning, trace build, and reporting."""

    if policy.transition_time_unit != "seconds":
        raise ValueError("Phase 1 requires transition_time_unit='seconds'.")
    raw_events = reader(dataset.path, keep_extensions=policy.keep_extensions)
    events, cleaning_report = normalize_and_clean(raw_events, dataset, policy)
    if events.empty:
        raise EmptyEventLogError(
            f"Dataset {dataset.name} has no valid events after core-field cleaning."
        )
    traces = build_traces(
        events,
        sort_events=False,
        transition_time_unit=policy.transition_time_unit,
    )
    dataset_report = build_dataset_report(
        dataset.name, events, traces, cleaning_report
    )
    return Phase1Result(
        dataset=dataset.name,
        dataset_spec=dataset,
        preprocessing_config=policy,
        reader=f"{reader.__module__}.{reader.__name__}",
        source_path=str(dataset.path),
        source_sha256=sha256_file(dataset.path),
        events=events,
        traces=traces,
        cleaning_report=cleaning_report,
        dataset_report=dataset_report,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 text atomically.

    Inputs:
        path: Path to the input or output artifact.
        text: Input value used by this operation.
    Outputs:
        None.
    Paper role:
        Event-log ingestion, cleaning, and trace construction.
    """
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_json(path: Path, payload: object) -> None:
    """Write json.

    Inputs:
        path: Path to the input or output artifact.
        payload: Structured content to validate, serialize, or write.
    Outputs:
        None.
    Paper role:
        Event-log ingestion, cleaning, and trace construction.
    """
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_phase1_artifacts(
    result: Phase1Result,
    output_dir: str | Path,
    *,
    include_events: bool = True,
    include_traces: bool = True,
) -> dict[str, str]:
    """Persist canonical Phase 1 outputs and a reproducibility manifest."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    # A manifest is the commit marker for the complete artifact set. Removing
    # the previous marker first prevents a failed rerun from looking complete.
    manifest_path.unlink(missing_ok=True)
    artifacts: dict[str, str] = {}

    cleaning_path = destination / "cleaning_report.json"
    dataset_path = destination / "dataset_report.json"
    _write_json(cleaning_path, result.cleaning_report.to_dict())
    _write_json(dataset_path, result.dataset_report.to_dict())
    artifacts["cleaning_report"] = cleaning_path.name
    artifacts["dataset_report"] = dataset_path.name

    if include_events:
        events_path = destination / "events.csv"
        temporary_events = events_path.with_name(f".{events_path.name}.tmp")
        result.events.to_csv(temporary_events, index=False, encoding="utf-8")
        temporary_events.replace(events_path)
        artifacts["events"] = events_path.name
    else:
        (destination / "events.csv").unlink(missing_ok=True)

    if include_traces:
        traces_path = destination / "traces.jsonl"
        temporary_traces = traces_path.with_name(f".{traces_path.name}.tmp")
        with temporary_traces.open("w", encoding="utf-8", newline="\n") as handle:
            for trace in result.traces:
                handle.write(
                    json.dumps(trace.to_dict(), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
        temporary_traces.replace(traces_path)
        artifacts["traces"] = traces_path.name
    else:
        (destination / "traces.jsonl").unlink(missing_ok=True)

    artifact_sha256 = {
        key: sha256_file(destination / filename)
        for key, filename in artifacts.items()
    }

    manifest = {
        "phase": 1,
        "dataset": result.dataset,
        "dataset_spec": result.dataset_spec.to_dict(),
        "source_path": result.source_path,
        "source_sha256": result.source_sha256,
        "reader": result.reader,
        "canonical_columns": list(CORE_COLUMNS),
        "preprocessing_config": result.preprocessing_config.to_dict(),
        "runtime": {
            "python": platform.python_version(),
            "pandas": _package_version("pandas"),
            "pm4py": _package_version("pm4py"),
            "numpy": _package_version("numpy"),
            "lxml": _package_version("lxml"),
        },
        "counts": {
            "events": result.dataset_report.events,
            "cases": result.dataset_report.cases,
            "activities": result.dataset_report.activities,
            "variants": result.dataset_report.variants,
        },
        "artifacts": dict(artifacts),
        "artifact_sha256": artifact_sha256,
    }
    _write_json(manifest_path, manifest)
    artifacts["manifest"] = manifest_path.name
    return artifacts

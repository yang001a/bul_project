from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bul_project.config import DEFAULT_PREPROCESSING, DatasetSpec
from bul_project.preprocessing.pipeline import (
    EmptyEventLogError,
    preprocess_dataset,
    write_phase1_artifacts,
)


def test_phase1_pipeline_writes_reproducible_artifacts(
    demo_spec: DatasetSpec, tmp_path: Path
) -> None:
    result = preprocess_dataset(demo_spec)
    artifacts = write_phase1_artifacts(result, tmp_path)

    assert set(artifacts) == {
        "cleaning_report",
        "dataset_report",
        "events",
        "traces",
        "manifest",
    }
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["phase"] == 1
    assert manifest["dataset"] == "DEMO"
    assert manifest["counts"] == {
        "events": 7,
        "cases": 3,
        "activities": 7,
        "variants": 3,
    }
    assert len(manifest["source_sha256"]) == 64
    assert manifest["preprocessing_config"] == DEFAULT_PREPROCESSING.to_dict()
    assert manifest["dataset_spec"]["activity_column"] == "concept:name"
    assert manifest["reader"].endswith("read_xes_dataframe")
    assert set(manifest["artifact_sha256"]) == {
        "cleaning_report",
        "dataset_report",
        "events",
        "traces",
    }
    first_hashes = manifest["artifact_sha256"]

    write_phase1_artifacts(result, tmp_path)
    repeated_manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert repeated_manifest["artifact_sha256"] == first_hashes

    event_header = (tmp_path / "events.csv").read_text(encoding="utf-8").splitlines()[0]
    assert event_header == "case_id,activity,timestamp"
    trace_lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(trace_lines) == 3


def test_report_only_artifacts_omit_large_tables(
    demo_spec: DatasetSpec, tmp_path: Path
) -> None:
    result = preprocess_dataset(demo_spec)
    write_phase1_artifacts(result, tmp_path)
    artifacts = write_phase1_artifacts(
        result,
        tmp_path,
        include_events=False,
        include_traces=False,
    )

    assert "events" not in artifacts
    assert "traces" not in artifacts
    assert not (tmp_path / "events.csv").exists()
    assert not (tmp_path / "traces.jsonl").exists()
    assert (tmp_path / "manifest.json").is_file()


def test_pipeline_rejects_an_empty_cleaned_log(demo_spec: DatasetSpec) -> None:
    def empty_reader(path: object, *, keep_extensions: bool = False) -> pd.DataFrame:
        del path, keep_extensions
        return pd.DataFrame(
            columns=["case:concept:name", "concept:name", "time:timestamp"]
        )

    with pytest.raises(EmptyEventLogError, match="no valid events"):
        preprocess_dataset(demo_spec, reader=empty_reader)

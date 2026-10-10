from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pm4py
import pytest

from bul_project.validation.artifacts import (
    is_committed_artifact,
    petri_net_semantic_sha256,
    sha256_file,
    synthetic_event_semantic_sha256,
    write_petri_pnml_atomic,
    write_process_quality_metadata_json_atomic,
    write_process_quality_result_csv_atomic,
    write_synthetic_xes_atomic,
)
from bul_project.validation.process_model import ProcessModelEvaluator
from bul_project.validation.synthetic_log import SyntheticLogResult


@pytest.fixture(scope="module")
def synthetic_log() -> SyntheticLogResult:
    start = pd.Timestamp("2026-01-01 00:00:00")
    events = pd.DataFrame(
        [
            {"case_id": "synthetic_000001", "activity": "A", "timestamp": start},
            {
                "case_id": "synthetic_000001",
                "activity": "B",
                "timestamp": start + pd.to_timedelta(1.123456789, unit="s"),
            },
            {"case_id": "synthetic_000002", "activity": "A", "timestamp": start},
            {
                "case_id": "synthetic_000002",
                "activity": "C",
                "timestamp": start + pd.to_timedelta(2.25, unit="s"),
            },
        ],
        columns=("case_id", "activity", "timestamp"),
    )
    return SyntheticLogResult(
        events=events,
        seed=4,
        case_count=2,
        metadata={
            "module": "SyntheticLogGenerator",
            "status": "success",
            "dp_post_processing_only": True,
            "raw_data_access": False,
            "seed": 4,
            "case_count": 2,
            "event_count": 4,
        },
    )


@pytest.fixture(scope="module")
def process_quality(synthetic_log: SyntheticLogResult):
    return ProcessModelEvaluator(
        conformance_variant="token_based", show_progress=False
    ).evaluate(
        synthetic_log,
        synthetic_log.events,
        evaluation_log_id="artifact-demo-evaluation",
    )


def test_synthetic_xes_gzip_roundtrip_has_raw_and_semantic_hashes(
    tmp_path: Path,
    synthetic_log: SyntheticLogResult,
) -> None:
    destination = tmp_path / "synthetic_log.xes.gz"

    artifact = write_synthetic_xes_atomic(synthetic_log, destination)

    assert destination.read_bytes()[:2] == b"\x1f\x8b"
    assert artifact.format == "XES.GZ"
    assert artifact.sha256 == sha256_file(destination)
    assert artifact.semantic_sha256 == synthetic_event_semantic_sha256(
        synthetic_log.events, timestamp_precision="microsecond"
    )
    assert artifact.metadata["source_full_precision_semantic_sha256"] == (
        synthetic_event_semantic_sha256(
            synthetic_log.events, timestamp_precision="nanosecond"
        )
    )
    assert artifact.metadata["roundtrip_verified"] is True
    assert artifact.metadata["temporal_samples_serialized"] is False
    assert is_committed_artifact(
        destination, expected_sha256=artifact.sha256
    )
    assert not list(tmp_path.glob("*.tmp-*"))

    imported = pm4py.read_xes(
        str(destination),
        return_legacy_log_object=False,
        show_progress_bar=False,
    )
    assert len(imported) == 4
    assert imported["case:concept:name"].nunique() == 2
    with pytest.raises(FileExistsError):
        write_synthetic_xes_atomic(synthetic_log, destination)


def test_pnml_roundtrip_preserves_model_semantics_and_quality_metrics(
    tmp_path: Path,
    process_quality,
) -> None:
    destination = tmp_path / "petri_net.pnml"

    artifact = write_petri_pnml_atomic(process_quality.model, destination)

    assert artifact.format == "PNML"
    assert artifact.sha256 == sha256_file(destination)
    assert artifact.semantic_sha256 == petri_net_semantic_sha256(
        process_quality.model.net,
        process_quality.model.initial_marking,
        process_quality.model.final_marking,
    )
    roundtrip = pm4py.read_pnml(str(destination))
    assert artifact.semantic_sha256 == petri_net_semantic_sha256(*roundtrip)
    assert artifact.metadata["roundtrip_verified"] is True
    assert 0.0 <= process_quality.fitness <= 1.0
    assert 0.0 <= process_quality.precision <= 1.0
    assert 0.0 <= process_quality.generalization <= 1.0
    assert is_committed_artifact(
        destination, expected_sha256=artifact.sha256
    )
    assert not list(tmp_path.glob("*.tmp-*"))


def test_condition_csv_and_json_are_atomic_and_exclude_event_and_sample_arrays(
    tmp_path: Path,
    synthetic_log: SyntheticLogResult,
    process_quality,
) -> None:
    xes = write_synthetic_xes_atomic(
        synthetic_log, tmp_path / "synthetic.xes.gz"
    )
    pnml = write_petri_pnml_atomic(
        process_quality.model, tmp_path / "model.pnml"
    )
    condition = {
        "dataset": "BPIC2017",
        "epsilon": 8.0,
        "budget_strategy": "uniform",
        "seed": 4,
    }

    csv_digest = write_process_quality_result_csv_atomic(
        condition,
        process_quality,
        xes,
        pnml,
        tmp_path / "process_quality_result.csv",
    )
    json_digest = write_process_quality_metadata_json_atomic(
        condition,
        synthetic_log,
        process_quality,
        xes,
        pnml,
        tmp_path / "process_quality_metadata.json",
        temporal_metadata={
            "module": "TemporalDistributionGenerator",
            "generated_sample_count": 4,
            "raw_data_access": False,
        },
    )

    result = pd.read_csv(tmp_path / "process_quality_result.csv")
    metadata_text = (tmp_path / "process_quality_metadata.json").read_text(
        encoding="utf-8"
    )
    metadata = json.loads(metadata_text)
    assert len(result) == 1
    assert result.loc[0, "fitness"] == pytest.approx(process_quality.fitness)
    assert result.loc[0, "synthetic_xes_sha256"] == xes.sha256
    assert result.loc[0, "petri_pnml_sha256"] == pnml.sha256
    assert csv_digest.sha256 == sha256_file(
        tmp_path / "process_quality_result.csv"
    )
    assert json_digest.sha256 == sha256_file(
        tmp_path / "process_quality_metadata.json"
    )
    assert metadata["event_rows_serialized_in_metadata"] is False
    assert metadata["temporal_samples_serialized"] is False
    assert "samples_seconds" not in metadata_text
    assert "synthetic_000001" not in metadata_text


def test_temporal_sample_arrays_are_rejected_from_metadata(
    tmp_path: Path,
    synthetic_log: SyntheticLogResult,
    process_quality,
) -> None:
    xes = write_synthetic_xes_atomic(
        synthetic_log, tmp_path / "synthetic.xes.gz"
    )
    pnml = write_petri_pnml_atomic(
        process_quality.model, tmp_path / "model.pnml"
    )

    with pytest.raises(ValueError, match="must not be serialized"):
        write_process_quality_metadata_json_atomic(
            {"condition_id": "demo"},
            synthetic_log,
            process_quality,
            xes,
            pnml,
            tmp_path / "metadata.json",
            temporal_metadata={"samples_seconds": [1.0, 2.0]},
        )
    assert not (tmp_path / "metadata.json").exists()


def test_failed_export_cannot_leave_a_completed_or_temporary_artifact(
    tmp_path: Path,
    synthetic_log: SyntheticLogResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "failed.xes.gz"

    def fail_after_partial_write(log, file_path, **kwargs):
        Path(file_path).write_bytes(b"partial")
        raise RuntimeError("forced exporter failure")

    monkeypatch.setattr(pm4py, "write_xes", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="forced exporter failure"):
        write_synthetic_xes_atomic(synthetic_log, destination)

    assert not destination.exists()
    assert not is_committed_artifact(destination)
    assert not list(tmp_path.iterdir())


def test_temporary_sibling_is_never_recognized_as_committed(tmp_path: Path) -> None:
    temporary = tmp_path / ".synthetic.xes.gz.tmp-deadbeef.xes.gz"
    temporary.write_bytes(b"temporary")

    assert not is_committed_artifact(temporary)

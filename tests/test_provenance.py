from __future__ import annotations

import json
from pathlib import Path

import pytest

from bul_project.config import PROJECT_ROOT
from bul_project.experiments.main_experiment import build_code_manifest as build_plan_hashes
from bul_project.provenance import (
    ProvenanceError,
    build_code_hashes,
    build_code_manifest,
    build_release_manifest,
    verify_provenance_artifacts,
    write_provenance_artifacts,
)


def _minimal_repository(
    root: Path,
    *,
    include_excluded_python: bool = False,
) -> None:
    (root / "src" / "bul_project").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "tests").mkdir()
    (root / "data" / "raw").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "src" / "bul_project" / "__init__.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    (root / "scripts" / "run.py").write_text("VALUE = 3\n", encoding="utf-8")
    if include_excluded_python:
        (root / "tests" / "test_ignored.py").write_text(
            "VALUE = 4\n", encoding="utf-8"
        )
        (root / "data" / "raw" / "ignored.py").write_text(
            "VALUE = 5\n", encoding="utf-8"
        )
        (root / "results" / "ignored.py").write_text(
            "VALUE = 6\n", encoding="utf-8"
        )


def test_complete_code_boundary_includes_package_and_workflow() -> None:
    hashes = build_code_hashes(PROJECT_ROOT)

    assert "main.py" in hashes
    assert "src/bul_project/provenance.py" in hashes
    assert "src/bul_project/experiments/main_experiment.py" in hashes
    assert "scripts/build_provenance_manifest.py" in hashes
    assert not any(path.startswith("tests/") for path in hashes)
    assert not any(path.startswith("data/") for path in hashes)
    assert not any(path.startswith("results/") for path in hashes)
    assert build_plan_hashes(PROJECT_ROOT) == hashes


def test_code_manifest_is_deterministic_and_excludes_untrusted_trees(
    tmp_path: Path,
) -> None:
    _minimal_repository(tmp_path, include_excluded_python=True)
    first = build_code_manifest(tmp_path)
    second = build_code_manifest(tmp_path)

    assert first == second
    assert first["counts"] == {
        "total": 3,
        "entry_points": 1,
        "package_modules": 1,
        "workflow_modules": 1,
    }
    assert {row["path"] for row in first["files"]} == {
        "main.py",
        "scripts/run.py",
        "src/bul_project/__init__.py",
    }

    (tmp_path / "src" / "bul_project" / "__init__.py").write_text(
        "VALUE = 20\n", encoding="utf-8"
    )
    changed = build_code_manifest(tmp_path)
    assert changed["aggregate_sha256"] != first["aggregate_sha256"]


def test_clean_release_manifest_does_not_claim_completed_results(
    tmp_path: Path,
) -> None:
    _minimal_repository(tmp_path)
    manifest = build_release_manifest(tmp_path)

    assert manifest["release_state"] == "clean_source_release"
    assert manifest["input_data"]["file_count"] == 0
    assert manifest["generated_configuration"]["file_count"] == 0
    assert manifest["generated_results"]["file_count"] == 0
    assert manifest["scientific_guards"]["main_condition_count"] == 3150
    assert manifest["scientific_guards"]["mathematical_logic_modified_by_provenance"] is False


def test_written_manifests_verify_and_fail_after_code_change(
    tmp_path: Path,
) -> None:
    _minimal_repository(tmp_path)
    summary = write_provenance_artifacts(tmp_path)

    assert summary["release_state"] == "clean_source_release"
    assert summary["generated_result_files"] == 0
    assert verify_provenance_artifacts(tmp_path)["status"] == "verified"
    stored = json.loads(
        (tmp_path / "paper_artifacts" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored["manifest_sha256"]

    (tmp_path / "main.py").write_text("VALUE = 10\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="code manifest"):
        verify_provenance_artifacts(tmp_path)

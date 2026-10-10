from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_main_experiment as phase8a
from bul_project.experiments.main_experiment import (
    FormalConditionExecutionError,
    FormalExperimentRunner,
    FormalRunIntegrityError,
    build_formal_execution_plan,
    current_runtime_metadata,
    load_frozen_execution_spec,
)
from bul_project.experiments.parameter_freeze import Phase7CDatasetInput
from bul_project.experiments.single_run import LossDimensions
from bul_project.preprocessing.models import Trace


FIXED_DEMO_SCALES = {
    ("A", "B"): 3600.0,
    ("Review", "End"): 3600.0,
    ("Start", "Review"): 3600.0,
}


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _refresh_config_manifest(config_path: Path) -> None:
    manifest_path = config_path.parent / "phase7c_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"][config_path.name] = _sha256(config_path)
    _write_json(manifest_path, manifest)


@pytest.fixture()
def demo_frozen_bundle(
    tmp_path: Path,
    demo_path: Path,
    demo_traces: list[Trace],
) -> dict[str, object]:
    phase7c_dir = tmp_path / "phase7c"
    normalization_path = phase7c_dir / "edge_normalization.json"
    normalization = {
        "name": "BUL_edge_normalization",
        "phase": "7C",
        "status": "frozen",
        "policy": {
            "dp_release": False,
            "epsilon_independent": True,
            "high_quantile": 0.95,
            "low_quantile": 0.05,
            "normalization_only": True,
            "quantile_method": "linear",
            "raw_reference_used": True,
            "synthetic_generation_input": False,
            "unit": "seconds",
            "zero_scale_policy": "replace exact zero with 1 only",
        },
        "datasets": {
            "DEMO": [
                {
                    "source": edge[0],
                    "target": edge[1],
                    "normalization_factor_seconds": value,
                    "selected_for_experiment": True,
                }
                for edge, value in sorted(FIXED_DEMO_SCALES.items())
            ]
        },
        "raw_time_samples_serialized": False,
    }
    _write_json(normalization_path, normalization)

    smoke_dir = phase7c_dir / "smoke"
    smoke_files = {
        "batch_manifest": smoke_dir / "batch_manifest.json",
        "experiment_result": smoke_dir / "experiment_result.csv",
        "experiment_run_metadata": (
            smoke_dir / "experiment_run_metadata.jsonl"
        ),
        "experiment_stage_result": (
            smoke_dir / "experiment_stage_result.csv"
        ),
    }
    for logical_name, smoke_path in smoke_files.items():
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        smoke_path.write_text(f"demo {logical_name}\n", encoding="utf-8")

    source_path = demo_path.resolve()
    source_sha256 = _sha256(source_path)
    config_path = phase7c_dir / "experiment_config_final.json"
    config = {
        "name": "BUL_experiment_config_final",
        "phase": "7C",
        "status": "frozen",
        "artifact_sha256": {
            normalization_path.name: _sha256(normalization_path),
        },
        "datasets": [
            {
                "dataset": "DEMO",
                "evaluation_log": None,
                "l_max": 3,
                "min_edge_support": 1,
                "normalization_edge_count": len(FIXED_DEMO_SCALES),
                "normalization_reference": normalization_path.name,
                "source_path": str(source_path),
                "source_sha256": source_sha256,
                "t_max_seconds": 10.0,
                "top_k_variants": 3,
            }
        ],
        "dp_edge_only_missing_dimensions": [
            "L_temporal",
            "L_information",
            "BUL",
            "fitness",
            "precision",
            "generalization",
        ],
        "experiment_matrix": {
            "appendix_dp_edge_mode": None,
            "budget_strategies": ["uniform"],
            "dp_edge_modes": ["component"],
            "epsilon_values": [0.5, 8.0],
            "main_dp_edge_mode": "component",
            "seeds": [0],
            "time_budget_internal_split": {
                "time_squared_sum": 0.5,
                "time_sum": 0.5,
            },
        },
        "parameter_robustness": {
            "reference_edge_tau": 1,
            "reference_top_k": 3,
            "full_cross_product_required": True,
        },
        "policy": {
            "reference_edge_tau": 1,
            "reference_top_k": 3,
            "synthetic_consistency_scope": (
                "process_quality_datasets_only"
            ),
            "variant_coverage_role": "diagnostic_only",
        },
        "privacy_boundary": {"conditional_dp_claim": True},
        "process_quality": None,
        "runtime": current_runtime_metadata(),
        "smoke_evidence": {
            "passed": True,
            "artifact_sha256": {
                name: _sha256(path) for name, path in smoke_files.items()
            },
        },
    }
    _write_json(config_path, config)
    manifest = {
        "name": "BUL_Phase7C_manifest",
        "phase": "7C",
        "status": "frozen",
        "artifact_sha256": {
            config_path.name: _sha256(config_path),
            normalization_path.name: _sha256(normalization_path),
        },
        "source_sha256": {"DEMO": source_sha256},
        "runtime": current_runtime_metadata(),
    }
    _write_json(phase7c_dir / "phase7c_manifest.json", manifest)
    phase7c_input = Phase7CDatasetInput(
        dataset="DEMO",
        traces=demo_traces,
        source_sha256=source_sha256,
        source_path=str(source_path),
    )
    return {
        "config_path": config_path,
        "normalization_path": normalization_path,
        "phase7c_input": phase7c_input,
        "project_root": Path(__file__).resolve().parents[1],
    }


def _runner(
    bundle: dict[str, object],
    output_dir: Path,
    *,
    executor=None,
) -> FormalExperimentRunner:
    spec = load_frozen_execution_spec(bundle["config_path"])
    return FormalExperimentRunner(
        spec,
        (bundle["phase7c_input"],),
        output_dir,
        executor=executor,
        project_root=bundle["project_root"],
    )


def test_valid_bundle_builds_stable_unique_demo_plan(
    demo_frozen_bundle: dict[str, object],
) -> None:
    spec = load_frozen_execution_spec(demo_frozen_bundle["config_path"])
    first = build_formal_execution_plan(
        spec,
        project_root=demo_frozen_bundle["project_root"],
    )
    second = build_formal_execution_plan(
        spec,
        project_root=demo_frozen_bundle["project_root"],
    )

    assert first.plan_sha256 == second.plan_sha256
    assert len(first.conditions) == 4
    assert len({item.formal_condition_id for item in first.conditions}) == 4
    assert all(
        item.formal_condition_id.startswith("reference_k3_tau1|")
        for item in first.conditions
    )
    assert first.to_dict()["counts"] == {
        "datasets": 1,
        "dp_conditions": 4,
        "baseline_rows": 2,
        "experiment_rows": 6,
        "stage_rows": 12,
        "metadata_rows": 4,
    }
    assert first.to_dict()["process_quality_conditions"] == 0


def test_non_frozen_config_is_rejected(
    demo_frozen_bundle: dict[str, object],
) -> None:
    config_path = demo_frozen_bundle["config_path"]
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["status"] = "ready_for_smoke"
    _write_json(config_path, payload)
    _refresh_config_manifest(config_path)

    with pytest.raises(FormalRunIntegrityError, match="frozen"):
        load_frozen_execution_spec(config_path)


def test_changed_final_config_or_normalization_is_rejected(
    demo_frozen_bundle: dict[str, object],
) -> None:
    config_path = demo_frozen_bundle["config_path"]
    config_path.write_bytes(config_path.read_bytes() + b" ")
    with pytest.raises(FormalRunIntegrityError, match="manifest"):
        load_frozen_execution_spec(config_path)

    config_path.write_bytes(config_path.read_bytes()[:-1])
    normalization_path = demo_frozen_bundle["normalization_path"]
    normalization_path.write_bytes(normalization_path.read_bytes() + b" ")
    with pytest.raises(FormalRunIntegrityError, match="artifact changed"):
        load_frozen_execution_spec(config_path)


def test_runtime_and_temporal_split_must_match_execution_code(
    demo_frozen_bundle: dict[str, object],
) -> None:
    config_path = demo_frozen_bundle["config_path"]
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["runtime"]["python"] = "0.0.0"
    _write_json(config_path, payload)
    _refresh_config_manifest(config_path)
    with pytest.raises(FormalRunIntegrityError, match="runtime"):
        load_frozen_execution_spec(config_path)

    payload["runtime"] = current_runtime_metadata()
    payload["experiment_matrix"]["time_budget_internal_split"] = {
        "time_sum": 0.4,
        "time_squared_sum": 0.6,
    }
    _write_json(config_path, payload)
    _refresh_config_manifest(config_path)
    with pytest.raises(FormalRunIntegrityError, match="temporal budget split"):
        load_frozen_execution_spec(config_path)


def test_source_hash_change_is_rejected_before_execution(
    demo_frozen_bundle: dict[str, object],
) -> None:
    source = demo_frozen_bundle["phase7c_input"]
    changed = Phase7CDatasetInput(
        dataset=source.dataset,
        traces=source.traces,
        source_sha256="f" * 64,
        source_path=source.source_path,
    )
    spec = load_frozen_execution_spec(demo_frozen_bundle["config_path"])
    with pytest.raises(FormalRunIntegrityError, match="source hash changed"):
        FormalExperimentRunner(
            spec,
            (changed,),
            Path(demo_frozen_bundle["config_path"]).parent / "bad-run",
            project_root=demo_frozen_bundle["project_root"],
        )


def test_large_tmax_sensitivity_audit_accepts_paper_formula_roundoff(
    demo_frozen_bundle: dict[str, object],
    tmp_path: Path,
) -> None:
    config_path = demo_frozen_bundle["config_path"]
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["datasets"][0]["l_max"] = 70
    payload["datasets"][0]["t_max_seconds"] = 326449.43039999995
    _write_json(config_path, payload)
    _refresh_config_manifest(config_path)

    power_value = float(70 - 1) * 326449.43039999995**2
    multiplication_value = (
        float(70 - 1) * 326449.43039999995 * 326449.43039999995
    )
    assert power_value != multiplication_value
    assert abs(power_value - multiplication_value) == 0.0009765625
    assert math.isclose(
        power_value,
        multiplication_value,
        rel_tol=1e-15,
        abs_tol=1e-12,
    )

    runner = _runner(demo_frozen_bundle, tmp_path / "large-tmax")
    summary = runner.run(max_new_conditions=1)
    assert summary.status == "partial"
    assert summary.completed_conditions == 1
    formal = runner.plan.conditions[0]
    metadata = json.loads(
        (
            runner._checkpoint_dir(formal) / "experiment_metadata.json"
        ).read_text(encoding="utf-8")
    )
    sensitivities = metadata["release"]["sensitivities"]
    assert sensitivities["time_squared_sum"] == (
        float(70 - 1) * 326449.43039999995**2
    )


def test_large_tmax_sensitivity_audit_still_rejects_material_change(
    demo_frozen_bundle: dict[str, object],
    tmp_path: Path,
) -> None:
    config_path = demo_frozen_bundle["config_path"]
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["datasets"][0]["l_max"] = 70
    payload["datasets"][0]["t_max_seconds"] = 326449.43039999995
    _write_json(config_path, payload)
    _refresh_config_manifest(config_path)

    output = tmp_path / "large-tmax-tamper"
    runner = _runner(demo_frozen_bundle, output)
    runner.run(max_new_conditions=1)
    formal = runner.plan.conditions[0]
    checkpoint_dir = runner._checkpoint_dir(formal)
    metadata_path = checkpoint_dir / "experiment_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["release"]["sensitivities"]["time_squared_sum"] *= 1.0 + 1e-12
    _write_json(metadata_path, metadata)
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"][metadata_path.name] = _sha256(metadata_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(
        FormalRunIntegrityError,
        match="sensitivities differ",
    ):
        _runner(demo_frozen_bundle, output).run()


def test_partial_run_resumes_and_complete_run_is_idempotent(
    demo_frozen_bundle: dict[str, object],
    tmp_path: Path,
) -> None:
    output = tmp_path / "formal"
    first_calls: list[str] = []

    def first_executor(dataset_input, condition):
        first_calls.append(condition.condition_id)
        return FormalExperimentRunner._default_executor(dataset_input, condition)

    first = _runner(
        demo_frozen_bundle,
        output,
        executor=first_executor,
    ).run(max_new_conditions=2)
    assert first.status == "partial"
    assert first.completed_conditions == 2
    assert first.executed_this_invocation == 2
    assert len(first_calls) == 2
    assert not (output / "formal_run_manifest.json").exists()

    second_calls: list[str] = []

    def second_executor(dataset_input, condition):
        second_calls.append(condition.condition_id)
        return FormalExperimentRunner._default_executor(dataset_input, condition)

    second = _runner(
        demo_frozen_bundle,
        output,
        executor=second_executor,
    ).run()
    assert second.status == "complete"
    assert second.completed_conditions == 4
    assert second.executed_this_invocation == 2
    assert second.skipped_this_invocation == 2
    assert len(second_calls) == 2

    scientific_files = (
        "experiment_result.csv",
        "experiment_stage_result.csv",
        "experiment_run_metadata.jsonl",
        "formal_run_manifest.json",
    )
    before = {name: (output / name).read_bytes() for name in scientific_files}
    third_calls: list[str] = []

    def forbidden_executor(dataset_input, condition):
        third_calls.append(condition.condition_id)
        raise AssertionError("A committed condition was executed again.")

    third = _runner(
        demo_frozen_bundle,
        output,
        executor=forbidden_executor,
    ).run()
    after = {name: (output / name).read_bytes() for name in scientific_files}
    assert third.status == "complete"
    assert third.executed_this_invocation == 0
    assert third.skipped_this_invocation == 4
    assert third_calls == []
    assert before == after

    results = pd.read_csv(output / "experiment_result.csv", keep_default_na=False)
    stages = pd.read_csv(
        output / "experiment_stage_result.csv",
        keep_default_na=False,
    )
    assert len(results) == 6
    assert len(stages) == 12
    assert (results["condition_id"].nunique()) == 6
    edge_rows = results[results["method"] == "DP-edge-only"]
    for field in (
        "L_temporal",
        "L_information",
        "BUL",
        "fitness",
        "precision",
        "generalization",
    ):
        assert set(edge_rows[field]) == {"NA"}

    manifest_path = output / "formal_run_manifest.json"
    tampered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered_manifest["counts"]["stage_rows"] -= 1
    _write_json(manifest_path, tampered_manifest)
    with pytest.raises(FormalRunIntegrityError, match="counts changed"):
        _runner(demo_frozen_bundle, output).run()


def test_uncommitted_payload_is_rerun_but_committed_tamper_fails_closed(
    demo_frozen_bundle: dict[str, object],
    tmp_path: Path,
) -> None:
    output = tmp_path / "formal"
    runner = _runner(demo_frozen_bundle, output)
    formal = runner.plan.conditions[0]
    orphan = runner._checkpoint_dir(formal)
    orphan.mkdir(parents=True)
    (orphan / "experiment_metadata.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    def recording_executor(dataset_input, condition):
        calls.append(condition.condition_id)
        return FormalExperimentRunner._default_executor(dataset_input, condition)

    partial = _runner(
        demo_frozen_bundle,
        output,
        executor=recording_executor,
    ).run(max_new_conditions=1)
    assert partial.completed_conditions == 1
    assert calls == [formal.condition.condition_id]

    metadata_path = orphan / "experiment_metadata.json"
    metadata_path.write_bytes(metadata_path.read_bytes() + b" ")
    with pytest.raises(FormalRunIntegrityError, match="artifact changed"):
        _runner(
            demo_frozen_bundle,
            output,
            executor=recording_executor,
        ).run()
    assert len(calls) == 1


def test_failed_condition_has_no_commit_marker_and_is_retryable(
    demo_frozen_bundle: dict[str, object],
    tmp_path: Path,
) -> None:
    output = tmp_path / "formal"
    attempts: list[str] = []

    def failing_executor(dataset_input, condition):
        attempts.append(condition.condition_id)
        if len(attempts) == 2:
            raise RuntimeError("simulated interruption")
        return FormalExperimentRunner._default_executor(dataset_input, condition)

    runner = _runner(
        demo_frozen_bundle,
        output,
        executor=failing_executor,
    )
    with pytest.raises(FormalConditionExecutionError):
        runner.run()
    failed = runner.plan.conditions[1]
    failed_dir = runner._checkpoint_dir(failed)
    assert (failed_dir / "failure.json").is_file()
    assert not (failed_dir / "checkpoint_manifest.json").exists()
    assert not (output / "formal_run_manifest.json").exists()

    retry_calls: list[str] = []

    def retry_executor(dataset_input, condition):
        retry_calls.append(condition.condition_id)
        return FormalExperimentRunner._default_executor(dataset_input, condition)

    summary = _runner(
        demo_frozen_bundle,
        output,
        executor=retry_executor,
    ).run()
    assert summary.status == "complete"
    assert summary.skipped_this_invocation == 1
    assert len(retry_calls) == 3


def test_injected_edge_only_non_na_result_is_rejected_before_commit(
    demo_frozen_bundle: dict[str, object],
    tmp_path: Path,
) -> None:
    output = tmp_path / "formal"

    def invalid_executor(dataset_input, condition):
        result = FormalExperimentRunner._default_executor(
            dataset_input,
            condition,
        )
        total = result.stage_losses["total"]
        changed_stages = dict(result.stage_losses)
        changed_stages["total"] = LossDimensions(
            l_structure=total.l_structure,
            l_temporal=0.0,
            l_information=0.0,
            bul=0.0,
        )
        return replace(result, stage_losses=changed_stages)

    runner = _runner(
        demo_frozen_bundle,
        output,
        executor=invalid_executor,
    )
    edge_formal = next(
        item
        for item in runner.plan.conditions
        if item.condition.method == "DP-edge-only"
    )
    with pytest.raises(FormalRunIntegrityError, match="must remain NA"):
        runner._execute_condition(edge_formal)
    assert not (
        runner._checkpoint_dir(edge_formal) / "checkpoint_manifest.json"
    ).exists()


def test_existing_different_plan_is_rejected(
    demo_frozen_bundle: dict[str, object],
    tmp_path: Path,
) -> None:
    output = tmp_path / "formal"
    output.mkdir()
    (output / "formal_execution_plan.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(FormalRunIntegrityError, match="plan differs"):
        _runner(demo_frozen_bundle, output)


def test_plan_only_cli_never_reads_xes(
    demo_frozen_bundle: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden_preprocess(*args, **kwargs):
        raise AssertionError("Plan-only mode must not read an XES file.")

    monkeypatch.setattr(phase8a, "preprocess_dataset", forbidden_preprocess)
    output = tmp_path / "cli-plan"
    code = phase8a.main(
        [
            "--config",
            str(demo_frozen_bundle["config_path"]),
            "--output-dir",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(
        (output / "formal_execution_plan.json").read_text(encoding="utf-8")
    )
    assert code == 0
    assert captured.out == ""
    assert payload["counts"]["dp_conditions"] == 4
    assert (output / "formal_execution_plan.json").is_file()

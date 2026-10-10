from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bul_project.validation.scope import (
    ProcessQualityScopeIntegrityError,
    build_process_quality_scope,
    load_process_quality_scope,
    score_preregistered_seed,
    select_preregistered_seed,
    validate_process_quality_scope,
    write_process_quality_scope,
)
from bul_project.validation.evaluation_sample import (
    expected_evaluation_sample_metadata,
    expected_evaluation_sampling_policy,
)


CURRENT_REFERENCE_PLAN_HASH = (
    "86ff4a18250da89750362cddebbec3d79e8e097ea274bd250067706b33350726"
)


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _condition(epsilon: float, seed: int = 27) -> dict[str, object]:
    condition_id = (
        f"BPIC2017|DP-full|epsilon={epsilon:g}|seed={seed}|mechanism=uniform"
    )
    formal_id = f"reference_k100_tau10|{condition_id}"
    return {
        "scenario_id": "reference_k100_tau10",
        "formal_condition_id": formal_id,
        "checkpoint_key": _sha256_bytes(formal_id.encode("utf-8")),
        "condition": {
            "condition_id": condition_id,
            "dataset": "BPIC2017",
            "epsilon": epsilon,
            "seed": seed,
            "method": "DP-full",
            "budget_strategy": "uniform",
            "dp_edge_mode": None,
            "evaluate_process_quality": False,
            "conformance_variant": None,
            "process_model_noise_threshold": None,
        },
    }


def _build_bundle(tmp_path: Path) -> dict[str, Path]:
    phase7c_dir = tmp_path / "phase7c"
    formal_dir = tmp_path / "formal"
    config_path = phase7c_dir / "experiment_config_final.json"
    plan_path = formal_dir / "formal_execution_plan.json"
    manifest_path = formal_dir / "formal_run_manifest.json"
    checkpoint_root = formal_dir / "checkpoints"

    source_hashes = {
        "BPIC2012": "1" * 64,
        "BPIC2017": "2" * 64,
        "SEPSIS": "3" * 64,
    }
    evaluation_hash = "4" * 64
    config = {
        "name": "BUL_experiment_config_final",
        "phase": "7C",
        "status": "frozen",
        "datasets": [
            {
                "dataset": "BPIC2012",
                "source_sha256": source_hashes["BPIC2012"],
                "evaluation_log": None,
                "l_max": 58,
                "t_max_seconds": 258995.1866,
                "top_k_variants": 100,
                "min_edge_support": 10,
            },
            {
                "dataset": "BPIC2017",
                "source_sha256": source_hashes["BPIC2017"],
                "l_max": 70,
                "t_max_seconds": 326449.43039999995,
                "top_k_variants": 100,
                "min_edge_support": 10,
                "evaluation_log": {
                    "evaluation_log_id": (
                        "BPIC2017_phase1_complete_canonical_log"
                    ),
                    "evaluation_log_sha256": evaluation_hash,
                    "event_count": 1202267,
                    "case_count": 31509,
                    "core_columns": ["case_id", "activity", "timestamp"],
                    "timestamp_policy": (
                        "complete canonical timestamps required"
                    ),
                    "condition_policy": "same fixed evaluation rows",
                },
            },
            {
                "dataset": "SEPSIS",
                "source_sha256": source_hashes["SEPSIS"],
                "evaluation_log": None,
                "l_max": 30,
                "t_max_seconds": 259200.0,
                "top_k_variants": 100,
                "min_edge_support": 10,
            },
        ],
        "experiment_matrix": {
            "epsilon_values": [0.5, 1.0, 2.0, 4.0, 8.0],
            "seeds": list(range(30)),
            "budget_strategies": [
                "uniform",
                "activity_focus",
                "edge_focus",
                "variant_focus",
                "time_focus",
            ],
        },
        "process_quality": {
            "datasets": ["BPIC2017"],
            "conformance_variant": "token_based",
            "inductive_miner_noise_threshold": 0.0,
        },
        "synthetic_log_policy": {
            "variant_driven": True,
            "other_excluded": True,
            "same_generation_rule_across_conditions": True,
            "timestamp_start": "2026-01-01T00:00:00",
        },
    }
    _write_json(config_path, config)
    config_hash = _sha256_file(config_path)

    conditions = [_condition(epsilon) for epsilon in (0.5, 2.0, 8.0)]
    plan_core = {
        "name": "BUL_formal_execution_plan",
        "phase": "8A",
        "status": "planned",
        "scope": "reference_numeric",
        "scenario_id": "reference_k100_tau10",
        "phase7c_final_config_sha256": config_hash,
        "reference_parameters": {
            "top_k": 100,
            "edge_tau": 10,
            "main_dp_edge_mode": "component",
            "appendix_dp_edge_mode": "mechanism",
        },
        "source_sha256": source_hashes,
        "condition_plan": conditions,
        "process_quality_scope_status": "deferred_not_frozen",
        "process_quality_conditions": 0,
        "sensitivity_pending": True,
        "code_sha256": {},
    }
    plan_hash = ""
    for nonce in range(10000):
        plan_core["code_sha256"] = {"test_nonce": str(nonce)}
        candidate = _sha256_bytes(_json_bytes(plan_core))
        if select_preregistered_seed(candidate).selected_seed == 27:
            plan_hash = candidate
            break
    assert plan_hash
    _write_json(plan_path, {**plan_core, "plan_sha256": plan_hash})

    checkpoint_hashes: dict[str, str] = {}
    for row in conditions:
        checkpoint_dir = checkpoint_root / str(row["checkpoint_key"])
        checkpoint_dir.mkdir(parents=True)
        metadata_path = checkpoint_dir / "experiment_metadata.json"
        result_path = checkpoint_dir / "experiment_result.csv"
        metadata_path.write_text(
            json.dumps(
                {
                    "name": "BUL_single_experiment",
                    "release": {
                        "method": "DP-full",
                        "seed": 27,
                        "epsilon_total": row["condition"]["epsilon"],
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        result_path.write_text("dataset,epsilon\nBPIC2017,1\n", encoding="utf-8")
        checkpoint = {
            "name": "BUL_formal_condition_checkpoint",
            "phase": "8A",
            "status": "complete",
            "scenario_id": "reference_k100_tau10",
            "formal_condition_id": row["formal_condition_id"],
            "condition": row["condition"],
            "execution_plan_sha256": plan_hash,
            "phase7c_final_config_sha256": config_hash,
            "dataset_source_sha256": source_hashes["BPIC2017"],
            "evaluation_log_sha256": evaluation_hash,
            "raw_data_rows_serialized": False,
            "artifacts": {
                "experiment_metadata": metadata_path.name,
                "experiment_result": result_path.name,
            },
            "artifact_sha256": {
                metadata_path.name: _sha256_file(metadata_path),
                result_path.name: _sha256_file(result_path),
            },
        }
        checkpoint_path = checkpoint_dir / "checkpoint_manifest.json"
        _write_json(checkpoint_path, checkpoint)
        checkpoint_hashes[str(row["formal_condition_id"])] = _sha256_file(
            checkpoint_path
        )

    manifest = {
        "name": "BUL_formal_run",
        "phase": "8A",
        "status": "complete",
        "scope": "reference_numeric",
        "scenario_id": "reference_k100_tau10",
        "execution_plan_sha256": plan_hash,
        "phase7c_final_config_sha256": config_hash,
        "process_quality_scope_status": "deferred_not_frozen",
        "counts": {
            "dp_conditions": 3,
            "completed_conditions": 3,
            "failed_conditions": 0,
        },
        "artifact_sha256": {plan_path.name: _sha256_file(plan_path)},
        "source_sha256": source_hashes,
        "reference_parameters": plan_core["reference_parameters"],
        "checkpoint_manifest_sha256": checkpoint_hashes,
    }
    _write_json(manifest_path, manifest)
    return {
        "plan": plan_path,
        "manifest": manifest_path,
        "config": config_path,
        "checkpoints": checkpoint_root,
    }


def test_current_plan_hash_selects_preregistered_seed_27() -> None:
    selection = select_preregistered_seed(CURRENT_REFERENCE_PLAN_HASH)

    assert selection.selected_seed == 27
    assert selection.selected_score == (
        "0170970f78b64019bc3f3cc391b0700b47fd47e67cd04819321420cc81bc75f6"
    )
    assert selection.selected_score == min(score for _, score in selection.scores)
    assert score_preregistered_seed(CURRENT_REFERENCE_PLAN_HASH, 27) == (
        selection.selected_score
    )


def test_build_write_load_and_idempotent_immutable_scope(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    scope = build_process_quality_scope(
        bundle["plan"],
        bundle["manifest"],
        bundle["config"],
        checkpoint_root=bundle["checkpoints"],
    )
    payload = scope.to_dict()

    assert payload["scope"]["seed"] == 27
    assert payload["scope"]["condition_count"] == 3
    assert payload["scope"]["epsilon_values"] == [0.5, 2.0, 8.0]
    assert len(payload["conditions"]) == 3
    assert payload["explicit_exclusions"]["dp_edge_only_quality_fields"] == "NA"
    assert payload["claim_boundary"]["not_a_30_seed_estimate"] is True
    boundary = payload["phase6_scope_boundary"]
    assert boundary["included_core_tasks"] == [
        "main_BUL_numeric_matrix",
        "BUL_equal_and_fixed_reference_BUL_entropy",
        "dimension_ablation",
        "parameter_sensitivity",
        "cross_dataset_summary",
    ]
    assert boundary["included_core_tasks_are_not_completion_claims"] is True
    assert boundary["external_validity_exception"] == {
        "dataset": "BPIC2017",
        "method": "DP-full",
        "budget_strategy": "uniform",
        "epsilon_values": [0.5, 2.0, 8.0],
        "seed": 27,
        "condition_count": 3,
        "role": "representative_external_validity_check_only",
    }
    assert boundary["removed_non_bul_tasks"] == [
        "full_PM4Py_condition_matrix",
        "all_budget_strategies_30_seeds_multi_dataset_PM4Py",
        "DP_edge_only_PM4Py",
        "3150_condition_synthetic_logs_and_Petri_nets",
        "PM4Py_inferential_statistics",
        "PM4Py_trend_or_correlation_analysis",
    ]
    assert payload["evaluation_log"] == expected_evaluation_sample_metadata()
    assert payload["evaluation_sampling"] == expected_evaluation_sampling_policy()
    assert payload["evaluation_log_source"]["case_count"] == 31509
    assert "three_point_spearman_role" not in payload["claim_boundary"]

    path = tmp_path / "scope" / "process_quality_scope_freeze.json"
    first = write_process_quality_scope(
        scope,
        path,
        bundle["plan"],
        bundle["manifest"],
        bundle["config"],
        checkpoint_root=bundle["checkpoints"],
    )
    original = first.read_bytes()
    second = write_process_quality_scope(
        scope,
        path,
        bundle["plan"],
        bundle["manifest"],
        bundle["config"],
        checkpoint_root=bundle["checkpoints"],
    )
    loaded = load_process_quality_scope(
        path,
        bundle["plan"],
        bundle["manifest"],
        bundle["config"],
        checkpoint_root=bundle["checkpoints"],
    )

    assert second == first
    assert first.read_bytes() == original
    assert loaded.to_dict() == payload


def test_rejects_rehashed_scope_with_any_scope_change(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    scope = build_process_quality_scope(
        bundle["plan"],
        bundle["manifest"],
        bundle["config"],
        checkpoint_root=bundle["checkpoints"],
    ).to_dict()
    scope["explicit_exclusions"]["epsilon_values"] = [1.0]
    core = dict(scope)
    core.pop("scope_sha256")
    scope["scope_sha256"] = _sha256_bytes(_json_bytes(core))

    with pytest.raises(
        ProcessQualityScopeIntegrityError,
        match="differs from the frozen sources",
    ):
        validate_process_quality_scope(
            scope,
            bundle["plan"],
            bundle["manifest"],
            bundle["config"],
            checkpoint_root=bundle["checkpoints"],
        )


def test_fails_closed_when_selected_checkpoint_artifact_changes(
    tmp_path: Path,
) -> None:
    bundle = _build_bundle(tmp_path)
    scope = build_process_quality_scope(
        bundle["plan"],
        bundle["manifest"],
        bundle["config"],
        checkpoint_root=bundle["checkpoints"],
    )
    first_key = scope.to_dict()["conditions"][0]["checkpoint_key"]
    artifact = (
        bundle["checkpoints"] / first_key / "experiment_metadata.json"
    )
    artifact.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        ProcessQualityScopeIntegrityError,
        match="Selected checkpoint artifact changed",
    ):
        validate_process_quality_scope(
            scope,
            bundle["plan"],
            bundle["manifest"],
            bundle["config"],
            checkpoint_root=bundle["checkpoints"],
        )


def test_existing_different_scope_file_is_not_overwritten(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    scope = build_process_quality_scope(
        bundle["plan"],
        bundle["manifest"],
        bundle["config"],
        checkpoint_root=bundle["checkpoints"],
    )
    path = tmp_path / "scope.json"
    path.write_bytes(b"{}\n")

    with pytest.raises(
        ProcessQualityScopeIntegrityError,
        match="differs and is immutable",
    ):
        write_process_quality_scope(
            scope,
            path,
            bundle["plan"],
            bundle["manifest"],
            bundle["config"],
            checkpoint_root=bundle["checkpoints"],
        )
    assert path.read_bytes() == b"{}\n"

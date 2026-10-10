from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from bul_project.experiments.parameter_freeze import build_evaluation_log_metadata
from bul_project.validation.runner import (
    _exclusive_run_lock,
    ProcessQualityFormalConditionError,
    ProcessQualityFormalIntegrityError,
    RepresentativeProcessQualityRunner,
    build_process_quality_execution_plan,
    validate_disjoint_directories,
)
from bul_project.validation.scope import (
    EXTERNAL_VALIDITY_CLAIM_STRENGTH,
    EXTERNAL_VALIDITY_ROLE,
    FrozenProcessQualityScope,
    select_preregistered_seed,
)
from bul_project.privacy.temporal_generator import TemporalDistributionGenerator


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


def _temporal_generation(seed: int) -> dict[str, object]:
    generated = TemporalDistributionGenerator(t_max=10.0, seed=seed).generate(
        edge_counts={("A", "B"): 4, ("B", "C"): 3},
        time_sums={("A", "B"): 8.0, ("B", "C"): 9.0},
        time_squared_sums={("A", "B"): 20.0, ("B", "C"): 33.0},
    )
    return {
        "seed": seed,
        "t_max": 10.0,
        "delta": generated.delta,
        "sample_counts": [
            {
                "source": source,
                "target": target,
                "sample_count": generated.sample_counts[(source, target)],
            }
            for source, target in sorted(generated.sample_counts)
        ],
        "metadata": generated.metadata,
    }


def _source_metadata(epsilon: float, seed: int = 27) -> dict[str, object]:
    return {
        "name": "small_authenticated_DP_checkpoint",
        "config": {
            "dataset": "BPIC2017",
            "method": "DP-full",
            "budget_strategy": "uniform",
            "dp_edge_mode": None,
            "epsilon": epsilon,
            "seed": seed,
            "t_max": 10.0,
        },
        "release": {
            "name": "G_DP",
            "method": "DP-full",
            "budget_strategy": "uniform",
            "dp_edge_mode": None,
            "epsilon_total": epsilon,
            "epsilon_spent": epsilon,
            "seed": seed,
            "dfg": [
                {"source": "A", "target": "B", "count": 4, "probability": 4 / 7},
                {"source": "B", "target": "C", "count": 3, "probability": 3 / 7},
            ],
            "temporal": [
                {
                    "source": "A",
                    "target": "B",
                    "mean_seconds": 2.0,
                    "sum_seconds": 8.0,
                    "squared_sum_seconds2": 20.0,
                },
                {
                    "source": "B",
                    "target": "C",
                    "mean_seconds": 3.0,
                    "sum_seconds": 9.0,
                    "squared_sum_seconds2": 33.0,
                },
            ],
            "variants": [
                {"activities": ["A", "B", "C"], "count": 2, "probability": 2 / 3},
                {"activities": ["A", "B"], "count": 1, "probability": 1 / 3},
            ],
            "variant_other": {"label": "OTHER", "count": 9, "probability": 0.75},
        },
        "temporal_generation": _temporal_generation(seed),
        "result_record": {
            "dataset": "BPIC2017",
            "epsilon": epsilon,
            "method": "DP-full",
            "seed": seed,
            "L_structure": 0.1,
            "L_temporal": 0.2,
            "L_information": 0.3,
            "BUL": 0.2,
            "fitness": None,
            "precision": None,
            "generalization": None,
            "top_k_variants": 100,
            "min_edge_support": 10,
            "budget_strategy": "uniform",
            "dp_edge_mode": None,
            "process_quality_evaluated": False,
            "conformance_variant": None,
        },
    }


def _evaluation_events() -> pd.DataFrame:
    start = pd.Timestamp("2026-01-01 00:00:00")
    return pd.DataFrame(
        [
            {"case_id": "evaluation_1", "activity": "A", "timestamp": start},
            {
                "case_id": "evaluation_1",
                "activity": "B",
                "timestamp": start + pd.Timedelta(seconds=1),
            },
            {
                "case_id": "evaluation_1",
                "activity": "C",
                "timestamp": start + pd.Timedelta(seconds=2),
            },
            {"case_id": "evaluation_2", "activity": "A", "timestamp": start},
            {
                "case_id": "evaluation_2",
                "activity": "B",
                "timestamp": start + pd.Timedelta(seconds=1),
            },
        ]
    )


def _bundle(tmp_path: Path):
    reference = tmp_path / "reference"
    checkpoint_root = reference / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    source_plan_hash = (
        "86ff4a18250da89750362cddebbec3d79"
        "e8e097ea274bd250067706b33350726"
    )
    conditions = []
    for epsilon in (0.5, 2.0, 8.0):
        epsilon_label = f"{epsilon:g}"
        formal_id = (
            "reference_k100_tau10|BPIC2017|DP-full|"
            f"epsilon={epsilon_label}|seed=27|mechanism=uniform"
        )
        key = hashlib.sha256(formal_id.encode("utf-8")).hexdigest()
        directory = checkpoint_root / key
        directory.mkdir()
        metadata_path = directory / "experiment_metadata.json"
        metadata_path.write_bytes(_json_bytes(_source_metadata(epsilon)))
        result_path = directory / "experiment_result.csv"
        result_path.write_text("dataset,epsilon\nBPIC2017," + str(epsilon) + "\n", encoding="utf-8")
        metadata_sha = _sha256(metadata_path)
        result_sha = _sha256(result_path)
        manifest = {
            "name": "BUL_formal_condition_checkpoint",
            "status": "complete",
            "formal_condition_id": formal_id,
            "execution_plan_sha256": source_plan_hash,
            "artifacts": {
                "experiment_result": "experiment_result.csv",
                "experiment_metadata": "experiment_metadata.json",
            },
            "artifact_sha256": {
                "experiment_result.csv": result_sha,
                "experiment_metadata.json": metadata_sha,
            },
        }
        manifest_path = directory / "checkpoint_manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        conditions.append(
            {
                "epsilon": epsilon,
                "formal_condition_id": formal_id,
                "checkpoint_key": key,
                "checkpoint_manifest_sha256": _sha256(manifest_path),
                "experiment_metadata_sha256": metadata_sha,
                "experiment_result_sha256": result_sha,
                "source_role": "verified_DP_release_checkpoint",
            }
        )

    events = _evaluation_events()
    evaluation = build_evaluation_log_metadata(events, "demo-evaluation").to_dict()
    core = {
        "name": "BUL_formal_process_quality_scope",
        "original_phase": 6,
        "scope_version": "PM4Py_scope_v1",
        "status": "frozen",
        "source_bundle": {
            "formal_execution_plan": {
                "internal_plan_sha256": source_plan_hash,
                "file_sha256": "b" * 64,
            }
        },
        "seed_selection": select_preregistered_seed(source_plan_hash).to_dict(),
        "scope": {
            "dataset": "BPIC2017",
            "method": "DP-full",
            "budget_strategy": "uniform",
            "epsilon_values": [0.5, 2.0, 8.0],
            "seed": 27,
            "condition_count": 3,
            "top_k_variants": 100,
            "min_edge_support": 10,
            "l_max": 70,
            "t_max_seconds": 10.0,
            "dataset_source_sha256": "c" * 64,
        },
        "evaluation_log_source": evaluation,
        "evaluation_log": evaluation,
        "evaluation_sampling": {
            "policy_version": "test_case_sample_v1",
            "sampling_unit": "case_id",
            "method": "SHA256_rank_without_replacement",
            "seed": 27,
            "sample_size_cases": int(events["case_id"].nunique()),
            "same_sample_for_all_conditions": True,
            "role": "external_scoring_only_not_DP_generation",
        },
        "process_quality_method": {
            "discovery_algorithm": "PM4Py Inductive Miner",
            "conformance_variant": "token_based",
            "inductive_miner_noise_threshold": 0.0,
            "metrics": ["fitness", "precision", "generalization"],
            "same_evaluation_log_for_all_conditions": True,
        },
        "synthetic_log_policy": {
            "input_boundary": "verified_DP_release_only",
            "raw_trace_access": False,
            "variant_driven": True,
            "other_excluded": True,
            "missing_temporal_edge_policy": "fail_and_record_missing_edge",
            "timestamp_start": "2026-01-01T00:00:00",
            "timestamp_direction": "nondecreasing_add_interval",
            "seed": 27,
        },
        "conditions": conditions,
        "claim_boundary": {
            "role": EXTERNAL_VALIDITY_ROLE,
            "claim_strength": EXTERNAL_VALIDITY_CLAIM_STRENGTH,
            "not_a_30_seed_estimate": True,
            "not_budget_strategy_robustness_evidence": True,
            "not_cross_dataset_evidence": True,
            "synthetic_log_is_DP_post_processing": True,
            "quality_metrics_are_evaluation_artifacts_not_DP_releases": True,
        },
        "explicit_exclusions": {
            "datasets": ["BPIC2012", "SEPSIS"],
            "methods": ["DP-edge-only"],
            "epsilon_values": [1.0, 4.0],
            "budget_strategies": [
                "activity_focus",
                "edge_focus",
                "variant_focus",
                "time_focus",
            ],
            "seeds": [seed for seed in range(30) if seed != 27],
            "metrics": ["simplicity"],
            "dp_edge_only_quality_fields": "NA",
            "other_variant_in_synthetic_generation": False,
        },
        "phase6_scope_boundary": {
            "included_core_tasks": [
                "main_BUL_numeric_matrix",
                "BUL_equal_and_fixed_reference_BUL_entropy",
                "dimension_ablation",
                "parameter_sensitivity",
                "cross_dataset_summary",
            ],
            "included_core_tasks_are_not_completion_claims": True,
            "external_validity_exception": {
                "dataset": "BPIC2017",
                "method": "DP-full",
                "budget_strategy": "uniform",
                "epsilon_values": [0.5, 2.0, 8.0],
                "seed": 27,
                "condition_count": 3,
                "role": EXTERNAL_VALIDITY_ROLE,
            },
            "removed_non_bul_tasks": [
                "full_PM4Py_condition_matrix",
                "all_budget_strategies_30_seeds_multi_dataset_PM4Py",
                "DP_edge_only_PM4Py",
                "3150_condition_synthetic_logs_and_Petri_nets",
                "PM4Py_inferential_statistics",
                "PM4Py_trend_or_correlation_analysis",
            ],
            "scope_semantics": (
                "Defines Phase 6 membership and exclusions; it does not mark "
                "an included but unfinished BUL task as complete."
            ),
        },
    }
    payload = {**core, "scope_sha256": hashlib.sha256(_json_bytes(core)).hexdigest()}
    return FrozenProcessQualityScope(payload), reference, events


def _runner(
    scope: FrozenProcessQualityScope,
    reference: Path,
    events,
    output: Path,
    *,
    evaluator_factory=None,
) -> RepresentativeProcessQualityRunner:
    output.mkdir(parents=True, exist_ok=True)
    scope_path = output / "process_quality_scope_freeze.json"
    if not scope_path.exists():
        scope_path.write_bytes(_json_bytes(scope.to_dict()))
    config_path = reference / "test_config.json"
    if not config_path.exists():
        config_path.write_text("{}\n", encoding="utf-8")
    return RepresentativeProcessQualityRunner(
        scope,
        reference,
        events,
        output,
        config_path=config_path,
        evaluator_factory=evaluator_factory,
        _scope_source_validator=lambda *args, **kwargs: scope,
    )


def test_three_condition_runner_resumes_and_is_idempotent(tmp_path: Path) -> None:
    scope, reference, events = _bundle(tmp_path)
    output = tmp_path / "output"
    runner = _runner(scope, reference, events, output)
    first = runner.run(max_new_conditions=1)
    assert first.status == "partial"
    assert first.completed_conditions == 1
    assert not (output / "process_quality_manifest.json").exists()

    second = _runner(scope, reference, events, output).run()
    assert second.status == "complete"
    assert second.completed_conditions == 3
    assert second.executed_this_invocation == 2
    assert second.skipped_this_invocation == 1
    progress = json.loads(
        (output / "process_quality_progress.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "complete"
    assert progress["remaining_conditions"] == 0
    rows = pd.read_csv(output / "process_quality_result.csv")
    assert list(rows["epsilon"]) == [0.5, 2.0, 8.0]
    assert rows[["fitness", "precision", "generalization"]].apply(
        lambda column: column.between(0, 1).all()
    ).all()

    immutable_files = {
        name: (output / name).read_bytes()
        for name in (
            "process_quality_result.csv",
            "process_quality_run_metadata.jsonl",
            "process_quality_manifest.json",
        )
    }

    third = _runner(scope, reference, events, output).run()
    assert third.status == "complete"
    assert third.executed_this_invocation == 0
    assert third.skipped_this_invocation == 3
    assert immutable_files == {
        name: (output / name).read_bytes() for name in immutable_files
    }


def test_committed_artifact_tamper_fails_closed(tmp_path: Path) -> None:
    scope, reference, events = _bundle(tmp_path)
    output = tmp_path / "output"
    _runner(scope, reference, events, output).run()
    condition = scope.to_dict()["conditions"][0]
    artifact = (
        output
        / "checkpoints"
        / condition["checkpoint_key"]
        / "process_quality_result.csv"
    )
    artifact.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ProcessQualityFormalIntegrityError, match="artifact changed"):
        _runner(scope, reference, events, output).run()


def test_committed_semantic_or_final_claim_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    scope, reference, events = _bundle(tmp_path)
    output = tmp_path / "output"
    _runner(scope, reference, events, output).run()
    condition = scope.to_dict()["conditions"][0]
    checkpoint_manifest = (
        output
        / "checkpoints"
        / condition["checkpoint_key"]
        / "checkpoint_manifest.json"
    )
    checkpoint = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    checkpoint["artifact_semantic_sha256"]["synthetic_log.xes.gz"] = "0" * 64
    checkpoint_manifest.write_bytes(_json_bytes(checkpoint))
    with pytest.raises(ProcessQualityFormalIntegrityError, match="semantic"):
        _runner(scope, reference, events, output).run()


def test_final_claim_tamper_fails_closed(tmp_path: Path) -> None:
    scope, reference, events = _bundle(tmp_path)
    output = tmp_path / "output"
    _runner(scope, reference, events, output).run()
    final_path = output / "process_quality_manifest.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["claim_boundary"] = {"role": "tampered"}
    final_path.write_bytes(_json_bytes(final))

    with pytest.raises(ProcessQualityFormalIntegrityError, match="manifest changed"):
        _runner(scope, reference, events, output).run()


def test_condition_claim_and_disk_scope_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    scope, reference, events = _bundle(tmp_path)
    output = tmp_path / "output"
    runner = _runner(scope, reference, events, output)
    runner.run(max_new_conditions=1)
    condition = scope.to_dict()["conditions"][0]
    manifest_path = (
        output
        / "checkpoints"
        / condition["checkpoint_key"]
        / "checkpoint_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["claim_role"] = "tampered"
    manifest_path.write_bytes(_json_bytes(manifest))
    with pytest.raises(ProcessQualityFormalIntegrityError, match="identity"):
        _runner(scope, reference, events, output).run()

    manifest["claim_role"] = EXTERNAL_VALIDITY_ROLE
    manifest_path.write_bytes(_json_bytes(manifest))
    (output / "process_quality_scope_freeze.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(ProcessQualityFormalIntegrityError, match="on-disk"):
        _runner(scope, reference, events, output)


def test_completed_run_does_not_materialize_lazy_evaluation_log(
    tmp_path: Path,
) -> None:
    scope, reference, events = _bundle(tmp_path)
    output = tmp_path / "output"
    calls = 0

    def loader():
        nonlocal calls
        calls += 1
        return events

    _runner(scope, reference, loader, output).run()
    assert calls == 1

    def forbidden_loader():
        raise AssertionError("completed run must not reload BPIC2017")

    summary = _runner(scope, reference, forbidden_loader, output).run()
    assert summary.status == "complete"
    assert summary.skipped_this_invocation == 3


def test_execution_plan_is_stable_and_contains_removed_tasks(tmp_path: Path) -> None:
    scope, _, _ = _bundle(tmp_path)

    first = build_process_quality_execution_plan(scope)
    second = build_process_quality_execution_plan(scope)

    assert first.to_dict() == second.to_dict()
    assert first.to_dict()["counts"] == {"conditions": 3, "datasets": 1, "seeds": 1}
    boundary = first.to_dict()["phase6_scope_boundary"]
    assert "full_PM4Py_condition_matrix" in boundary["removed_non_bul_tasks"]
    assert first.to_dict()["source_DP_release_reused_without_renoising"] is True


def test_wrong_evaluation_log_is_rejected_before_execution(tmp_path: Path) -> None:
    scope, reference, events = _bundle(tmp_path)
    events = events.copy()
    events.loc[0, "activity"] = "changed"

    with pytest.raises(ProcessQualityFormalIntegrityError, match="evaluation log"):
        _runner(scope, reference, events, tmp_path / "output")


def test_output_and_read_only_reference_must_be_disjoint(tmp_path: Path) -> None:
    scope, reference, events = _bundle(tmp_path)

    with pytest.raises(ProcessQualityFormalIntegrityError, match="disjoint"):
        RepresentativeProcessQualityRunner(
            scope,
            reference,
            events,
            reference,
            config_path=reference / "test_config.json",
            _scope_source_validator=lambda *args, **kwargs: scope,
        )

    with pytest.raises(ProcessQualityFormalIntegrityError, match="disjoint"):
        validate_disjoint_directories(reference, reference / "new-output")


def test_failed_condition_has_no_commit_marker(tmp_path: Path) -> None:
    scope, reference, events = _bundle(tmp_path)
    output = tmp_path / "output"
    runner = _runner(
        scope, reference, events, output,
        evaluator_factory=lambda: object(),  # type: ignore[arg-type]
    )

    with pytest.raises(ProcessQualityFormalConditionError):
        runner.run(max_new_conditions=1)

    condition = scope.to_dict()["conditions"][0]
    directory = output / "checkpoints" / condition["checkpoint_key"]
    assert (directory / "failure.json").is_file()
    assert not (directory / "checkpoint_manifest.json").exists()


def test_source_integrity_failure_is_not_wrapped_as_retryable_condition(
    tmp_path: Path,
) -> None:
    scope, reference, events = _bundle(tmp_path)
    condition = scope.to_dict()["conditions"][0]
    source = (
        reference
        / "checkpoints"
        / condition["checkpoint_key"]
        / "experiment_metadata.json"
    )
    source.write_text("tampered\n", encoding="utf-8")
    runner = _runner(scope, reference, events, tmp_path / "output")

    with pytest.raises(ProcessQualityFormalIntegrityError):
        runner.run(max_new_conditions=1)

    progress = json.loads(
        (tmp_path / "output" / "process_quality_progress.json").read_text(
            encoding="utf-8"
        )
    )
    assert progress["status"] == "blocked"


def test_output_lock_rejects_overlapping_execution(tmp_path: Path) -> None:
    scope, reference, events = _bundle(tmp_path)
    output = tmp_path / "output"
    runner = _runner(scope, reference, events, output)

    with _exclusive_run_lock(output / ".process_quality_run.lock"):
        with pytest.raises(
            ProcessQualityFormalIntegrityError, match="already holds"
        ):
            runner.run(max_new_conditions=1)

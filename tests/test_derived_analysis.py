from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from bul_project.config import PROJECT_ROOT
from bul_project.experiments.bul_analysis import (
    build_cross_dataset_summary,
    build_dimension_ablation,
    calculate_frozen_entropy_weights,
    enrich_bul_results,
    run_derived_analysis,
)


DATASETS = ("BPIC2012", "BPIC2017", "SEPSIS")
EPSILONS = (0.5, 1.0, 2.0, 4.0, 8.0)
STRATEGIES = (
    "uniform",
    "activity_focus",
    "edge_focus",
    "variant_focus",
    "time_focus",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_payload(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def _main_result_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    condition = 0

    def append(
        *,
        dataset: str,
        method: str,
        epsilon: object,
        seed: object,
        strategy: str,
        edge_mode: str,
        structure: object,
        temporal: object,
        information: object,
        bul: object,
    ) -> None:
        nonlocal condition
        condition += 1
        rows.append(
            {
                "scenario_id": f"scenario-{condition:04d}",
                "condition_id": f"condition-{condition:04d}",
                "scenario_id": "reference_k100_tau10",
                "dataset": dataset,
                "epsilon": epsilon,
                "method": method,
                "seed": seed,
                "budget_strategy": strategy,
                "dp_edge_mode": edge_mode,
                "status": "success",
                "top_k_variants": 100,
                "min_edge_support": 10,
                "L_structure": structure,
                "L_temporal": temporal,
                "L_information": information,
                "BUL": bul,
            }
        )

    for dataset_index, dataset in enumerate(DATASETS):
        for method, losses in (
            ("Raw", (0.0, 0.0, 0.0)),
            ("ClippedRaw", (0.03, 0.06, 0.09)),
        ):
            append(
                dataset=dataset,
                method=method,
                epsilon="NA",
                seed="NA",
                strategy="NA",
                edge_mode="NA",
                structure=losses[0],
                temporal=losses[1],
                information=losses[2],
                bul=sum(losses) / 3.0,
            )

        for epsilon in EPSILONS:
            for strategy_index, strategy in enumerate(STRATEGIES):
                for seed in range(30):
                    structure = 0.10 + dataset_index * 0.01
                    temporal = 0.20 + strategy_index * 0.01
                    information = 0.30 + seed * 0.0001
                    append(
                        dataset=dataset,
                        method="DP-full",
                        epsilon=epsilon,
                        seed=seed,
                        strategy=strategy,
                        edge_mode="NA",
                        structure=structure,
                        temporal=temporal,
                        information=information,
                        bul=(structure + temporal + information) / 3.0,
                    )

            for edge_mode in ("component", "mechanism"):
                for seed in range(30):
                    append(
                        dataset=dataset,
                        method="DP-edge-only",
                        epsilon=epsilon,
                        seed=seed,
                        strategy="NA",
                        edge_mode=edge_mode,
                        structure=0.10 + dataset_index * 0.01,
                        temporal="NA",
                        information="NA",
                        bul="NA",
                    )

    assert len(rows) == 3156
    return pd.DataFrame(rows)


def _smoke_stage_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(DATASETS):
        for epsilon_index, epsilon in enumerate((0.5, 8.0)):
            for stage_index, stage in enumerate(("clipping", "dp_noise", "total")):
                rows.append(
                    {
                        "dataset": dataset,
                        "epsilon": epsilon,
                        "method": "DP-full",
                        "seed": 0,
                        "budget_strategy": "uniform",
                        "stage": stage,
                        "L_structure": 0.01 + 0.02 * stage_index,
                        "L_temporal": 0.02 + 0.03 * epsilon_index,
                        "L_information": 0.03 + 0.01 * dataset_index + 0.01 * stage_index,
                    }
                )
    return pd.DataFrame(rows)


def test_fixed_entropy_reference_is_dataset_specific_and_frozen() -> None:
    weights, metadata = calculate_frozen_entropy_weights(
        _smoke_stage_fixture(), source_sha256="a" * 64
    )

    assert set(weights) == set(DATASETS)
    assert metadata["selection"] == {
        "method": "DP-full",
        "budget_strategy": "uniform",
        "seed": 0,
        "epsilon_values": [0.5, 8.0],
        "stages": ["clipping", "dp_noise", "total"],
        "rows_per_dataset": 6,
        "recalculated_per_epsilon_or_seed": False,
    }
    for dataset in DATASETS:
        assert sum(weights[dataset].values()) == pytest.approx(1.0)
        assert metadata["datasets"][dataset]["reference_row_count"] == 6


def test_enrichment_ablation_and_cross_dataset_contracts() -> None:
    main = _main_result_fixture()
    weights = {
        dataset: {"structure": 0.5, "temporal": 0.3, "information": 0.2}
        for dataset in DATASETS
    }
    enriched = enrich_bul_results(main, weights)

    full = enriched.loc[
        enriched["method"].eq("DP-full")
        & enriched["BUL_entropy_eligible"]
    ].iloc[0]
    assert full["BUL_equal"] == pytest.approx(
        (full["L_structure"] + full["L_temporal"] + full["L_information"]) / 3
    )
    assert full["BUL_entropy"] == pytest.approx(
        0.5 * full["L_structure"]
        + 0.3 * full["L_temporal"]
        + 0.2 * full["L_information"]
    )
    assert full["BUL_wo_structure"] == pytest.approx(
        (full["L_temporal"] + full["L_information"]) / 2
    )
    assert full["BUL_wo_temporal"] == pytest.approx(
        (full["L_structure"] + full["L_information"]) / 2
    )
    assert full["BUL_wo_information"] == pytest.approx(
        (full["L_structure"] + full["L_temporal"]) / 2
    )

    edge_only = enriched.loc[enriched["method"].eq("DP-edge-only")]
    derived = [
        "BUL_equal",
        "BUL_entropy",
        "BUL_wo_structure",
        "BUL_wo_temporal",
        "BUL_wo_information",
    ]
    assert edge_only[derived].isna().all().all()
    assert not edge_only["BUL_entropy_eligible"].any()
    calibration = enriched.loc[
        enriched["method"].eq("ClippedRaw")
        | (
            enriched["method"].eq("DP-full")
            & enriched["budget_strategy"].eq("uniform")
            & pd.to_numeric(enriched["seed"], errors="coerce").eq(0)
            & pd.to_numeric(enriched["epsilon"], errors="coerce").isin((0.5, 8.0))
        )
    ]
    assert len(calibration) == 9
    assert not calibration["BUL_entropy_eligible"].any()
    assert calibration["BUL_entropy"].isna().all()
    assert len(build_dimension_ablation(enriched)) == 2250

    summary = build_cross_dataset_summary(enriched)
    assert len(summary) == 105
    assert set(summary["seed_count"]) == {30}
    edge_summary = summary.loc[summary["method"].eq("DP-edge-only")]
    assert set(edge_summary["BUL_equal_n"]) == {0}
    assert edge_summary["BUL_equal_mean"].isna().all()


def test_authenticated_end_to_end_analysis_is_deterministic(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    phase7c = tmp_path / "phase7c"
    output = tmp_path / "output"
    (phase7c / "smoke").mkdir(parents=True)
    reference.mkdir()

    main_path = reference / "experiment_result.csv"
    stage_path = reference / "experiment_stage_result.csv"
    plan_path = reference / "formal_execution_plan.json"
    smoke_path = phase7c / "smoke" / "experiment_stage_result.csv"
    _main_result_fixture().to_csv(main_path, index=False, na_rep="NA")
    stage_path.write_text("stage\n", encoding="utf-8")
    plan_core = {"name": "fixture-formal-plan"}
    plan_hash = hashlib.sha256(_json_payload(plan_core).encode("utf-8")).hexdigest()
    plan_path.write_text(
        _json_payload({**plan_core, "plan_sha256": plan_hash}), encoding="utf-8"
    )
    _smoke_stage_fixture().to_csv(smoke_path, index=False)
    config = {
        "status": "frozen",
        "smoke_evidence": {
            "artifact_sha256": {"experiment_stage_result": _sha256(smoke_path)},
            "passed": True,
            "seed": 0,
            "budget_strategy": "uniform",
            "epsilons": [0.5, 8.0],
            "dp_conditions": 12,
        },
    }
    config_path = phase7c / "experiment_config_final.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest = {
        "status": "complete",
        "counts": {
            "completed_conditions": 3150,
            "failed_conditions": 0,
            "experiment_rows": 3156,
            "stage_rows": 9450,
        },
        "artifact_sha256": {
            "experiment_result.csv": _sha256(main_path),
            "experiment_stage_result.csv": _sha256(stage_path),
            "formal_execution_plan.json": _sha256(plan_path),
        },
        "phase7c_final_config_sha256": _sha256(config_path),
        "execution_plan_sha256": plan_hash,
    }
    manifest_path = reference / "formal_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source_lock = {
        "formal_manifest_sha256": _sha256(manifest_path),
        "formal_result_sha256": _sha256(main_path),
        "formal_stage_result_sha256": _sha256(stage_path),
        "formal_plan_file_sha256": _sha256(plan_path),
        "formal_internal_plan_sha256": plan_hash,
        "phase7c_config_sha256": _sha256(config_path),
        "phase7c_smoke_stage_sha256": _sha256(smoke_path),
    }

    first = run_derived_analysis(
        reference,
        phase7c,
        output,
        project_root=PROJECT_ROOT,
        source_lock=source_lock,
    )
    hashes_before = dict(first["artifact_sha256"])
    second = run_derived_analysis(
        reference,
        phase7c,
        output,
        project_root=PROJECT_ROOT,
        source_lock=source_lock,
    )

    assert first["status"] == "complete"
    assert first["counts"] == {
        "source_rows": 3156,
        "dp_conditions": 3150,
        "enriched_rows": 3156,
        "dimension_ablation_rows": 2250,
        "cross_dataset_summary_rows": 105,
        "entropy_weight_datasets": 3,
        "entropy_eligible_rows": 2247,
        "entropy_calibration_excluded_rows": 9,
    }
    assert second["artifact_sha256"] == hashes_before
    assert (output / "phase6_bul_analysis_manifest.json").is_file()

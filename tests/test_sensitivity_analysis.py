from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bul_project.config import (
    DERIVED_ANALYSIS_RESULT_DIR,
    FROZEN_CONFIG_PATH,
    MAIN_EXPERIMENT_RESULT_DIR,
)
from bul_project.experiments.sensitivity import (
    SensitivityExperimentRunner,
    SensitivitySources,
    build_sensitivity_conditions,
    load_sensitivity_sources,
    summarize_sensitivity_records,
)
from bul_project.preprocessing import Trace


def test_sensitivity_plan_has_exact_grid_and_reuse_partition() -> None:
    conditions = build_sensitivity_conditions()

    assert len(conditions) == 810
    assert len({row.condition_id for row in conditions}) == 810
    assert sum(row.is_reference_reuse for row in conditions) == 90
    assert sum(not row.is_reference_reuse for row in conditions) == 720
    assert {
        (row.top_k, row.edge_tau) for row in conditions
    } == {
        (k, tau) for k in (50, 100, 200) for tau in (5, 10, 20)
    }


def test_real_frozen_sources_expose_all_scales_and_reference_rows() -> None:
    if not (MAIN_EXPERIMENT_RESULT_DIR / "formal_run_manifest.json").is_file():
        pytest.skip("requires main-experiment outputs from the clean rerun")
    sources = load_sensitivity_sources(
        FROZEN_CONFIG_PATH,
        MAIN_EXPERIMENT_RESULT_DIR,
        DERIVED_ANALYSIS_RESULT_DIR,
    )

    assert sources.l_max == 70
    assert sources.t_max == 326449.43039999995
    assert len(sources.normalization_scales) == 178
    assert len(sources.reference_rows) == 90
    assert sum(sources.entropy_weights.values()) == 1.0
    assert sources.source_metadata["reference_selected_factor_count"] == 138


def test_sensitivity_summary_has_27_complete_cells_and_entropy_exclusions() -> None:
    records = []
    for condition in build_sensitivity_conditions():
        overlap = (
            condition.is_reference_reuse
            and condition.seed == 0
            and condition.epsilon in (0.5, 8.0)
        )
        records.append(
            {
                "condition_id": condition.condition_id,
                "top_k_variants": condition.top_k,
                "min_edge_support": condition.edge_tau,
                "epsilon": condition.epsilon,
                "seed": condition.seed,
                "L_structure": 0.1,
                "L_temporal": 0.2,
                "L_information": 0.3,
                "BUL_equal": 0.2,
                "BUL_entropy": None if overlap else 0.2,
                "variant_other_count": float(condition.top_k),
                "variant_other_probability": 0.25,
            }
        )

    summary = summarize_sensitivity_records(records)

    assert len(summary) == 27
    assert set(summary["seed_count"]) == {30}
    assert (summary["BUL_equal_n"] == 30).all()
    entropy_counts = summary["BUL_entropy_n"].value_counts().to_dict()
    assert entropy_counts == {30: 25, 29: 2}


def test_checkpoint_runner_executes_one_condition_and_resumes(tmp_path: Path) -> None:
    traces = tuple(
        Trace(
            case_id=f"c{index}",
            activities=("A", "B"),
            timestamps=(
                pd.Timestamp("2026-01-01T00:00:00Z"),
                pd.Timestamp("2026-01-01T00:00:01Z"),
            ),
            transition_seconds=(1.0,),
        )
        for index in range(10)
    )
    sources = SensitivitySources(
        config_path=tmp_path / "config.json",
        reference_dir=tmp_path / "reference",
        derived_dir=tmp_path / "derived",
        source_path=tmp_path / "demo.xes",
        source_sha256="a" * 64,
        l_max=3,
        t_max=10.0,
        normalization_scales={("A", "B"): 1.0},
        entropy_weights={
            "structure": 1 / 3,
            "temporal": 1 / 3,
            "information": 1 / 3,
        },
        reference_rows={},
        source_metadata={"fixture": True},
        scientific_code_sha256={},
    )
    runner = SensitivityExperimentRunner(sources, tmp_path / "output")

    first = runner.run(traces, max_new_conditions=1)
    resumed = SensitivityExperimentRunner(sources, tmp_path / "output")

    assert first.status == "partial"
    assert first.executed_this_invocation == 1
    assert first.completed_conditions == 91
    assert resumed.completed_new_conditions() == 1
    assert resumed.remaining_new_conditions() == 719
    assert not (tmp_path / "output" / "phase6_sensitivity_manifest.json").exists()

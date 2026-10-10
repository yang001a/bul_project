from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bul_project.config import (
    DERIVED_ANALYSIS_RESULT_DIR,
    EXTERNAL_VALIDATION_RESULT_DIR,
    MAIN_EXPERIMENT_RESULT_DIR,
    SENSITIVITY_ANALYSIS_RESULT_DIR,
)
from bul_project.statistics.analysis_runner import (
    StatisticalAnalysisError,
    assess_external_validity,
    run_statistical_analysis,
)


def _source_dirs() -> dict[str, Path]:
    return {
        "derived_dir": DERIVED_ANALYSIS_RESULT_DIR,
        "reference_dir": MAIN_EXPERIMENT_RESULT_DIR,
        "sensitivity_dir": SENSITIVITY_ANALYSIS_RESULT_DIR,
        "external_dir": EXTERNAL_VALIDATION_RESULT_DIR,
    }


def test_external_validity_correlation_is_not_fabricated() -> None:
    frame = pd.DataFrame(
        {
            "BUL_equal": [0.3, 0.2, 0.1],
            "fitness": [0.9, 0.9, 0.9],
            "precision": [0.4, 0.4, 0.4],
            "generalization": [0.8, 0.8, 0.8],
        }
    )
    result = assess_external_validity(frame)

    assert result["spearman_correlation_eligible"] is False
    assert result["status"] == "not_estimable_zero_variance"
    assert result["process_quality_in_BUL"] is False


def test_external_validity_is_optional_for_core_bul_statistics() -> None:
    """Record an omitted PM4Py check without blocking the BUL analysis."""

    result = assess_external_validity(None)

    assert result["status"] == "not_run_optional"
    assert result["process_quality_in_BUL"] is False
    assert result["core_inferential_statistics_included"] is False


def test_statistical_runner_writes_complete_reproducible_artifacts(
    tmp_path: Path,
) -> None:
    if not (MAIN_EXPERIMENT_RESULT_DIR / "formal_run_manifest.json").is_file():
        pytest.skip("requires completed rerun outputs for all downstream stages")
    output = tmp_path / "phase7"
    kwargs = {
        **_source_dirs(),
        "output_dir": output,
        "bootstrap_resamples": 200,
        "bootstrap_seed": 0,
    }
    first = run_statistical_analysis(**kwargs)
    second = run_statistical_analysis(**kwargs)

    assert first == second
    assert first["status"] == "complete"
    assert first["counts"]["descriptive_summary.csv"] == 405
    assert first["counts"]["friedman_results.csv"] == 126
    assert first["counts"]["wilcoxon_holm_results.csv"] == 1215
    assert first["counts"]["spearman_epsilon_trends.csv"] == 66
    assert first["counts"]["figure_files"] == 12
    assert first["method_guards"]["strict_epsilon_monotonicity_forced"] is False
    assert first["method_guards"]["stage_additivity_claimed"] is False
    assert first["method_guards"]["spearman_p_value_method"] == (
        "exact_two_sided_permutation"
    )
    assert first["method_guards"]["PM4Py_correlation_status"] == "not_run_optional"
    for name in first["artifact_sha256"]:
        assert (output / name).is_file()
    manifest = json.loads((output / "phase7_analysis_manifest.json").read_text())
    assert manifest == first


def test_statistical_runner_rejects_out_of_range_bootstrap_seed(
    tmp_path: Path,
) -> None:
    with pytest.raises(StatisticalAnalysisError, match="0-29"):
        run_statistical_analysis(
            **_source_dirs(),
            output_dir=tmp_path / "bad-seed",
            bootstrap_resamples=10,
            bootstrap_seed=30,
        )

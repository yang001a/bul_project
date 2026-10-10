from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from bul_project.config import DERIVED_ANALYSIS_RESULT_DIR
from bul_project.statistics.statistical_tests import (
    StatisticalTestError,
    bootstrap_mean_ci,
    build_descriptive_summary,
    build_friedman_results,
    build_spearman_trends,
    build_wilcoxon_results,
    exact_spearman_permutation,
    friedman_test,
    holm_adjust,
    paired_wilcoxon,
)


@pytest.fixture(scope="module")
def enriched() -> pd.DataFrame:
    if not (DERIVED_ANALYSIS_RESULT_DIR / "bul_enriched_result.csv").is_file():
        pytest.skip("requires derived-analysis output from the clean rerun")
    return pd.read_csv(
        DERIVED_ANALYSIS_RESULT_DIR / "bul_enriched_result.csv"
    )


def test_bootstrap_mean_ci_is_seed_reproducible() -> None:
    first = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], n_resamples=2_000, seed=7)
    second = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], n_resamples=2_000, seed=7)

    assert first == second
    assert first["mean"] == pytest.approx(2.5)
    assert first["ci_lower"] <= first["mean"] <= first["ci_upper"]
    assert first["bootstrap_seed"] == 7


def test_holm_matches_known_small_example() -> None:
    adjusted, rejected = holm_adjust([0.01, 0.04, 0.03], alpha=0.05)

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert rejected.tolist() == [True, False, False]


def test_paired_wilcoxon_effect_direction_and_zero_case() -> None:
    result = paired_wilcoxon([2, 3, 4, 5, 6], [1, 1, 1, 1, 1])
    zero = paired_wilcoxon([1, 2, 3], [1, 2, 3])

    assert result["rank_biserial_first_minus_second"] == pytest.approx(1.0)
    assert result["mean_difference_first_minus_second"] > 0.0
    assert result["n_pairs"] == 5
    assert zero["p_raw"] == 1.0
    assert zero["rank_biserial_first_minus_second"] == 0.0
    assert zero["status"] == "all_zero_differences"


def test_friedman_matches_scipy_known_example() -> None:
    samples = ([1, 2, 3, 4, 5], [2, 1, 4, 3, 6], [5, 4, 3, 2, 1])
    expected = stats.friedmanchisquare(*samples)
    actual = friedman_test(samples)

    assert actual["statistic"] == pytest.approx(expected.statistic)
    assert actual["p_raw"] == pytest.approx(expected.pvalue)
    assert actual["n_blocks"] == 5
    assert actual["n_conditions"] == 3


def test_exact_spearman_uses_all_five_level_permutations() -> None:
    result = exact_spearman_permutation(
        [0.5, 1.0, 2.0, 4.0, 8.0],
        [5.0, 4.0, 3.0, 2.0, 1.0],
    )

    assert result["rho"] == pytest.approx(-1.0)
    assert result["p_exact"] == pytest.approx(2.0 / 120.0)
    assert result["n_permutations"] == 120
    assert result["p_method"] == "exact_two_sided_permutation"


def test_main_grid_builds_expected_statistical_tables(
    enriched: pd.DataFrame,
) -> None:
    descriptive = build_descriptive_summary(enriched, n_resamples=200, bootstrap_seed=0)
    friedman = build_friedman_results(enriched)
    wilcoxon = build_wilcoxon_results(enriched)
    trends_first = build_spearman_trends(enriched, n_resamples=200, bootstrap_seed=0)
    trends_second = build_spearman_trends(enriched, n_resamples=200, bootstrap_seed=0)

    assert len(descriptive) == 405
    assert len(friedman) == 126
    assert len(wilcoxon) == 1215
    assert len(trends_first) == 66
    pd.testing.assert_frame_equal(trends_first, trends_second)
    assert descriptive["bootstrap_seed"].between(0, 29).all()
    assert set(descriptive["bootstrap_seed"]) == {0}
    assert descriptive.loc[
        descriptive["analysis"].eq("dp_edge_only"), "metric"
    ].eq("L_structure").all()
    assert wilcoxon["p_holm"].between(0.0, 1.0).all()
    assert wilcoxon["rank_biserial_first_minus_second"].between(-1.0, 1.0).all()
    assert not trends_first["strictly_decreasing"].isna().any()
    assert trends_first["p_method"].eq("exact_two_sided_permutation").all()
    assert trends_first["n_permutations"].eq(120).all()
    assert trends_first["p_raw"].dropna().ge(2.0 / 120.0).all()


def test_missing_pair_fails_in_current_phase(enriched: pd.DataFrame) -> None:
    target = enriched[
        enriched["method"].eq("DP-full")
        & enriched["dataset"].eq("BPIC2012")
        & enriched["budget_strategy"].eq("uniform")
        & enriched["epsilon"].eq(0.5)
        & enriched["seed"].eq(0)
    ].index
    broken = enriched.drop(target)

    with pytest.raises(
        StatisticalTestError,
        match="row counts changed|exactly seeds 0-29",
    ):
        build_friedman_results(broken)

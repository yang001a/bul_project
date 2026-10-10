"""Statistical inference and publication-figure interfaces."""

from .analysis_runner import (
    Phase7AnalysisError,
    StatisticalAnalysisError,
    assess_external_validity,
    run_phase7_analysis,
    run_statistical_analysis,
)
from .plotting import PlottingError
from .statistical_tests import (
    Phase7StatisticsError,
    StatisticalTestError,
    bootstrap_mean_ci,
    build_ablation_ranking_consistency,
    build_ablation_summary,
    build_descriptive_summary,
    build_friedman_results,
    build_sensitivity_summary,
    build_spearman_trends,
    build_stage_summary,
    build_wilcoxon_results,
    exact_spearman_permutation,
    friedman_test,
    holm_adjust,
    paired_wilcoxon,
)

__all__ = [
    "PlottingError",
    "StatisticalAnalysisError",
    "StatisticalTestError",
    "assess_external_validity",
    "bootstrap_mean_ci",
    "build_ablation_ranking_consistency",
    "build_ablation_summary",
    "build_descriptive_summary",
    "build_friedman_results",
    "build_sensitivity_summary",
    "build_spearman_trends",
    "build_stage_summary",
    "build_wilcoxon_results",
    "exact_spearman_permutation",
    "friedman_test",
    "holm_adjust",
    "paired_wilcoxon",
    "run_statistical_analysis",
    # Historical compatibility exports retained for frozen notebooks.
    "Phase7AnalysisError",
    "Phase7StatisticsError",
    "run_phase7_analysis",
]

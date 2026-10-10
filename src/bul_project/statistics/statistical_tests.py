"""Statistical analysis primitives for original Phase 7.

The functions in this module consume only frozen Phase 6 tabular results.  They
never read XES files, regenerate differential-privacy noise, or impute metrics
that are undefined for DP-edge-only.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


DATASET_ORDER = ("BPIC2012", "BPIC2017", "SEPSIS")
EPSILON_ORDER = (0.5, 1.0, 2.0, 4.0, 8.0)
STRATEGY_ORDER = (
    "uniform",
    "activity_focus",
    "edge_focus",
    "variant_focus",
    "time_focus",
)
MAIN_METRICS = (
    "L_structure",
    "L_temporal",
    "L_information",
    "BUL_equal",
)
EXPLORATORY_METRICS = ("BUL_entropy",)
ABLATION_METRICS = (
    "BUL_equal",
    "BUL_wo_structure",
    "BUL_wo_temporal",
    "BUL_wo_information",
)
SEEDS = tuple(range(30))


class Phase7StatisticsError(ValueError):
    """Raised when a statistical input violates the frozen pairing contract."""


def _numeric_vector(values: Iterable[object], role: str) -> np.ndarray:
    """Convert input values to a validated numeric vector.

    Inputs:
        values: Values processed by the operation.
        role: Input value used by this operation.
    Outputs:
        A value of type `np.ndarray` produced by the operation.
    Paper role:
        Statistical inference and publication-quality figure generation.
    """
    vector = np.asarray(list(values), dtype=float)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise Phase7StatisticsError(f"{role} must be a non-empty finite vector.")
    return vector


def bootstrap_mean_ci(
    values: Iterable[object],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Return a percentile bootstrap confidence interval for a sample mean.

    Input: one finite sample, confidence level, resample count, and seed.
    Output: n, mean, sample standard deviation, median, and CI endpoints.
    Paper step: Phase 7 random-noise stability reporting across experiment seeds.
    """

    vector = _numeric_vector(values, "bootstrap sample")
    if not 0.0 < float(confidence) < 1.0:
        raise Phase7StatisticsError("confidence must be in (0,1).")
    if int(n_resamples) < 1:
        raise Phase7StatisticsError("n_resamples must be positive.")
    rng = np.random.default_rng(int(seed))
    bootstrap_indices = rng.integers(
        0, vector.size, size=(int(n_resamples), vector.size)
    )
    bootstrap_means = vector[bootstrap_indices].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    lower, upper = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return {
        "n": int(vector.size),
        "mean": float(vector.mean()),
        "std": float(vector.std(ddof=1)) if vector.size > 1 else 0.0,
        "median": float(np.median(vector)),
        "ci_level": float(confidence),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "bootstrap_resamples": int(n_resamples),
        "bootstrap_seed": int(seed),
    }


def holm_adjust(
    p_values: Sequence[float], *, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Holm's step-down family-wise-error correction.

    Input: finite raw p-values in one explicitly defined comparison family.
    Output: adjusted p-values and rejection decisions in original row order.
    Paper step: Phase 7 multiple-comparison correction.
    """

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise Phase7StatisticsError("Holm p-values must be a finite vector.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise Phase7StatisticsError("Holm p-values must lie in [0,1].")
    if not 0.0 < float(alpha) < 1.0:
        raise Phase7StatisticsError("alpha must be in (0,1).")
    count = values.size
    if count == 0:
        return values.copy(), np.asarray([], dtype=bool)
    order = np.argsort(values, kind="mergesort")
    adjusted_sorted = np.empty(count, dtype=float)
    running = 0.0
    for index, original_index in enumerate(order):
        candidate = (count - index) * values[original_index]
        running = max(running, candidate)
        adjusted_sorted[index] = min(running, 1.0)
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted, adjusted <= float(alpha)


def paired_wilcoxon(
    first: Iterable[object], second: Iterable[object]
) -> dict[str, float | int | str]:
    """Run a two-sided paired Wilcoxon test with rank-biserial effect size.

    Input: two finite, equally sized paired samples in the stated first/second
    order.  Output: Wilcoxon statistic, raw p-value, paired differences, and
    matched-pairs rank-biserial correlation.  A positive effect means the first
    condition has larger loss.  Paper step: Phase 7 paired method comparisons.
    """

    left = _numeric_vector(first, "first paired sample")
    right = _numeric_vector(second, "second paired sample")
    if left.shape != right.shape:
        raise Phase7StatisticsError("Paired samples must have equal length.")
    differences = left - right
    nonzero = differences != 0.0
    effective = differences[nonzero]
    if effective.size == 0:
        statistic = 0.0
        p_value = 1.0
        effect = 0.0
        status = "all_zero_differences"
    else:
        result = stats.wilcoxon(
            left,
            right,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        ranks = stats.rankdata(np.abs(effective), method="average")
        positive = float(ranks[effective > 0.0].sum())
        negative = float(ranks[effective < 0.0].sum())
        effect = (positive - negative) / (positive + negative)
        status = "ok"
    return {
        "n_pairs": int(left.size),
        "n_nonzero": int(effective.size),
        "statistic": statistic,
        "p_raw": p_value,
        "mean_difference_first_minus_second": float(differences.mean()),
        "median_difference_first_minus_second": float(np.median(differences)),
        "rank_biserial_first_minus_second": float(effect),
        "status": status,
    }


def friedman_test(samples: Sequence[Iterable[object]]) -> dict[str, float | int | str]:
    """Run a Friedman repeated-measures test across three or more conditions.

    Input: ordered finite samples sharing the same pairing blocks.
    Output: Friedman chi-square statistic, p-value, block count, and condition
    count.  Paper step: Phase 7 omnibus comparison across epsilon or strategies.
    """

    vectors = [_numeric_vector(sample, "Friedman sample") for sample in samples]
    if len(vectors) < 3:
        raise Phase7StatisticsError("Friedman requires at least three conditions.")
    sizes = {vector.size for vector in vectors}
    if len(sizes) != 1:
        raise Phase7StatisticsError("Friedman samples must share pairing blocks.")
    matrix = np.column_stack(vectors)
    if np.all(matrix == matrix[0, 0]):
        statistic = 0.0
        p_value = 1.0
        status = "all_values_identical"
    else:
        result = stats.friedmanchisquare(*vectors)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        status = "ok"
    return {
        "n_blocks": int(matrix.shape[0]),
        "n_conditions": int(matrix.shape[1]),
        "statistic": statistic,
        "p_raw": p_value,
        "status": status,
    }


def _validate_seed_grid(
    frame: pd.DataFrame,
    *,
    treatment_column: str,
    treatments: Sequence[Any],
    metric: str,
    role: str,
) -> pd.DataFrame:
    """Validate seed grid.

    Inputs:
        frame: Tabular data processed by the operation.
        treatment_column: Input value used by this operation.
        treatments: Input value used by this operation.
        metric: Input value used by this operation.
        role: Input value used by this operation.
    Outputs:
        A value of type `pd.DataFrame` produced by the operation.
    Paper role:
        Statistical inference and publication-quality figure generation.
    """
    required = {"seed", treatment_column, metric}
    missing = required - set(frame.columns)
    if missing:
        raise Phase7StatisticsError(f"{role} is missing columns: {sorted(missing)!r}")
    if frame.duplicated(["seed", treatment_column]).any():
        raise Phase7StatisticsError(f"{role} contains duplicate paired cells.")
    wide = frame.pivot(index="seed", columns=treatment_column, values=metric)
    if set(wide.index) != set(SEEDS) or set(wide.columns) != set(treatments):
        raise Phase7StatisticsError(
            f"{role} must contain exactly seeds 0-29 and treatments {list(treatments)!r}."
        )
    wide = wide.reindex(index=SEEDS, columns=treatments)
    numeric = wide.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise Phase7StatisticsError(f"{role} contains missing/non-finite paired data.")
    return numeric


def _validate_main_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate main frame.

    Inputs:
        frame: Tabular data processed by the operation.
    Outputs:
        A value of type `pd.DataFrame` produced by the operation.
    Paper role:
        Statistical inference and publication-quality figure generation.
    """
    required = {
        "dataset",
        "epsilon",
        "method",
        "seed",
        "budget_strategy",
        "dp_edge_mode",
        "status",
        "BUL_entropy_eligible",
        *MAIN_METRICS,
        *EXPLORATORY_METRICS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise Phase7StatisticsError(
            f"Phase 6 enriched result is missing: {sorted(missing)!r}"
        )
    if set(frame["dataset"]) != set(DATASET_ORDER) or set(frame["status"]) != {"success"}:
        raise Phase7StatisticsError("Phase 6 dataset/status scope changed.")
    full = frame[frame["method"].eq("DP-full")].copy()
    edge = frame[frame["method"].eq("DP-edge-only")].copy()
    if len(full) != 2250 or len(edge) != 900:
        raise Phase7StatisticsError("Phase 6 DP method row counts changed.")
    return frame


def build_descriptive_summary(
    frame: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Build seed-level mean/std/median/bootstrap-CI tables.

    Input: Phase 6 enriched result table.  Output: summaries for DP-full main
    and exploratory metrics plus DP-edge-only structure loss.  Paper step:
    Phase 7 epsilon/BUL and random-noise stability analysis.
    """

    frame = _validate_main_frame(frame)
    rows: list[dict[str, Any]] = []
    full = frame[frame["method"].eq("DP-full")]
    group_columns = ["dataset", "epsilon", "budget_strategy"]
    for keys, group in full.groupby(group_columns, sort=False):
        dataset, epsilon, strategy = keys
        for metric in (*MAIN_METRICS, *EXPLORATORY_METRICS):
            metric_group = group
            role = "main" if metric in MAIN_METRICS else "exploratory_robustness_only"
            if metric == "BUL_entropy":
                metric_group = group[group["BUL_entropy_eligible"].astype(bool)]
            values = pd.to_numeric(metric_group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            summary = bootstrap_mean_ci(
                values,
                n_resamples=n_resamples,
                seed=bootstrap_seed,
            )
            rows.append(
                {
                    "analysis": "dp_full",
                    "analysis_role": role,
                    "dataset": dataset,
                    "method": "DP-full",
                    "epsilon": float(epsilon),
                    "budget_strategy": strategy,
                    "dp_edge_mode": pd.NA,
                    "metric": metric,
                    **summary,
                }
            )
    edge = frame[frame["method"].eq("DP-edge-only")]
    edge_columns = ["dataset", "epsilon", "dp_edge_mode"]
    for keys, group in edge.groupby(edge_columns, sort=False):
        dataset, epsilon, mode = keys
        summary = bootstrap_mean_ci(
            pd.to_numeric(group["L_structure"]),
            n_resamples=n_resamples,
            seed=bootstrap_seed,
        )
        rows.append(
            {
                "analysis": "dp_edge_only",
                "analysis_role": "structure_only",
                "dataset": dataset,
                "method": "DP-edge-only",
                "epsilon": float(epsilon),
                "budget_strategy": pd.NA,
                "dp_edge_mode": mode,
                "metric": "L_structure",
                **summary,
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["analysis", "dataset", "epsilon", "budget_strategy", "dp_edge_mode", "metric"],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _holm_by_family(frame: pd.DataFrame, family_column: str, alpha: float) -> pd.DataFrame:
    """Return holm by family.

    Inputs:
        frame: Tabular data processed by the operation.
        family_column: Input value used by this operation.
        alpha: Input value used by this operation.
    Outputs:
        A value of type `pd.DataFrame` produced by the operation.
    Paper role:
        Statistical inference and publication-quality figure generation.
    """
    result = frame.copy()
    result["p_holm"] = np.nan
    result["reject_holm"] = False
    result["holm_family_size"] = 0
    for _, indexes in result.groupby(family_column, sort=False).groups.items():
        positions = list(indexes)
        adjusted, rejected = holm_adjust(result.loc[positions, "p_raw"], alpha=alpha)
        result.loc[positions, "p_holm"] = adjusted
        result.loc[positions, "reject_holm"] = rejected
        result.loc[positions, "holm_family_size"] = len(positions)
    result["reject_holm"] = result["reject_holm"].astype(bool)
    result["holm_family_size"] = result["holm_family_size"].astype(int)
    return result


def build_friedman_results(frame: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    """Build omnibus repeated-measures tests for epsilon and strategy effects.

    Input: Phase 6 enriched results with complete seed grids.  Output: Friedman
    results for DP-full epsilon/strategy families and DP-edge-only epsilon
    families, with Holm correction across parallel omnibus tests.  Paper step:
    Phase 7 overall significance analysis.
    """

    frame = _validate_main_frame(frame)
    full = frame[frame["method"].eq("DP-full")]
    rows: list[dict[str, Any]] = []
    for (dataset, strategy), group in full.groupby(
        ["dataset", "budget_strategy"], sort=False
    ):
        for metric in MAIN_METRICS:
            wide = _validate_seed_grid(
                group,
                treatment_column="epsilon",
                treatments=EPSILON_ORDER,
                metric=metric,
                role=f"epsilon Friedman {dataset}/{strategy}/{metric}",
            )
            result = friedman_test([wide[value] for value in EPSILON_ORDER])
            rows.append(
                {
                    "comparison_scope": "epsilon",
                    "family_id": f"epsilon|{dataset}|{strategy}|{metric}",
                    "omnibus_holm_family": f"epsilon|{metric}",
                    "dataset": dataset,
                    "metric": metric,
                    "budget_strategy": strategy,
                    "dp_edge_mode": pd.NA,
                    "conditions": "|".join(str(value) for value in EPSILON_ORDER),
                    **result,
                }
            )
    for (dataset, epsilon), group in full.groupby(["dataset", "epsilon"], sort=False):
        for metric in MAIN_METRICS:
            wide = _validate_seed_grid(
                group,
                treatment_column="budget_strategy",
                treatments=STRATEGY_ORDER,
                metric=metric,
                role=f"strategy Friedman {dataset}/{epsilon}/{metric}",
            )
            result = friedman_test([wide[value] for value in STRATEGY_ORDER])
            rows.append(
                {
                    "comparison_scope": "budget_strategy",
                    "family_id": f"strategy|{dataset}|{epsilon}|{metric}",
                    "omnibus_holm_family": f"budget_strategy|{metric}",
                    "dataset": dataset,
                    "metric": metric,
                    "budget_strategy": pd.NA,
                    "dp_edge_mode": pd.NA,
                    "conditions": "|".join(STRATEGY_ORDER),
                    **result,
                }
            )
    edge = frame[frame["method"].eq("DP-edge-only")]
    for (dataset, mode), group in edge.groupby(["dataset", "dp_edge_mode"], sort=False):
        wide = _validate_seed_grid(
            group,
            treatment_column="epsilon",
            treatments=EPSILON_ORDER,
            metric="L_structure",
            role=f"edge epsilon Friedman {dataset}/{mode}",
        )
        result = friedman_test([wide[value] for value in EPSILON_ORDER])
        rows.append(
            {
                "comparison_scope": "edge_only_epsilon",
                "family_id": f"edge_epsilon|{dataset}|{mode}|L_structure",
                "omnibus_holm_family": "edge_only_epsilon|L_structure",
                "dataset": dataset,
                "metric": "L_structure",
                "budget_strategy": pd.NA,
                "dp_edge_mode": mode,
                "conditions": "|".join(str(value) for value in EPSILON_ORDER),
                **result,
            }
        )
    result = _holm_by_family(pd.DataFrame(rows), "omnibus_holm_family", alpha)
    result["alpha"] = float(alpha)
    return result.sort_values(
        ["comparison_scope", "dataset", "metric", "budget_strategy", "dp_edge_mode"],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _append_pairwise_family(
    rows: list[dict[str, Any]],
    *,
    wide: pd.DataFrame,
    conditions: Sequence[Any],
    base: dict[str, Any],
) -> None:
    """Append pairwise family.

    Inputs:
        rows: Input value used by this operation.
        wide: Input value used by this operation.
        conditions: Input value used by this operation.
        base: Input value used by this operation.
    Outputs:
        None.
    Paper role:
        Statistical inference and publication-quality figure generation.
    """
    for first, second in itertools.combinations(conditions, 2):
        test = paired_wilcoxon(wide[first], wide[second])
        rows.append(
            {
                **base,
                "condition_first": first,
                "condition_second": second,
                **test,
            }
        )


def build_wilcoxon_results(frame: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    """Build complete paired post-hoc comparisons with Holm correction.

    Input: Phase 6 enriched results.  Output: all pairwise epsilon and budget
    strategy comparisons for DP-full, plus component-versus-mechanism for
    DP-edge-only. Pairing is strictly by seed 0-29.  Paper step: Phase 7 paired
    significance and effect-size analysis.
    """

    frame = _validate_main_frame(frame)
    full = frame[frame["method"].eq("DP-full")]
    rows: list[dict[str, Any]] = []
    for (dataset, strategy), group in full.groupby(
        ["dataset", "budget_strategy"], sort=False
    ):
        for metric in MAIN_METRICS:
            family = f"epsilon|{dataset}|{strategy}|{metric}"
            wide = _validate_seed_grid(
                group,
                treatment_column="epsilon",
                treatments=EPSILON_ORDER,
                metric=metric,
                role=family,
            )
            _append_pairwise_family(
                rows,
                wide=wide,
                conditions=EPSILON_ORDER,
                base={
                    "comparison_scope": "epsilon",
                    "family_id": family,
                    "dataset": dataset,
                    "metric": metric,
                    "epsilon": pd.NA,
                    "budget_strategy": strategy,
                    "dp_edge_mode": pd.NA,
                },
            )
    for (dataset, epsilon), group in full.groupby(["dataset", "epsilon"], sort=False):
        for metric in MAIN_METRICS:
            family = f"strategy|{dataset}|{epsilon}|{metric}"
            wide = _validate_seed_grid(
                group,
                treatment_column="budget_strategy",
                treatments=STRATEGY_ORDER,
                metric=metric,
                role=family,
            )
            _append_pairwise_family(
                rows,
                wide=wide,
                conditions=STRATEGY_ORDER,
                base={
                    "comparison_scope": "budget_strategy",
                    "family_id": family,
                    "dataset": dataset,
                    "metric": metric,
                    "epsilon": float(epsilon),
                    "budget_strategy": pd.NA,
                    "dp_edge_mode": pd.NA,
                },
            )
    edge = frame[frame["method"].eq("DP-edge-only")]
    for (dataset, epsilon), group in edge.groupby(["dataset", "epsilon"], sort=False):
        wide = _validate_seed_grid(
            group,
            treatment_column="dp_edge_mode",
            treatments=("component", "mechanism"),
            metric="L_structure",
            role=f"edge mode {dataset}/{epsilon}",
        )
        _append_pairwise_family(
            rows,
            wide=wide,
            conditions=("component", "mechanism"),
            base={
                "comparison_scope": "edge_only_mode",
                "family_id": f"edge_mode|{dataset}|L_structure",
                "dataset": dataset,
                "metric": "L_structure",
                "epsilon": float(epsilon),
                "budget_strategy": pd.NA,
                "dp_edge_mode": "component_vs_mechanism",
            },
        )
    result = _holm_by_family(pd.DataFrame(rows), "family_id", alpha)
    result["alpha"] = float(alpha)
    return result.sort_values(
        ["comparison_scope", "family_id", "condition_first", "condition_second"],
        kind="mergesort",
    ).reset_index(drop=True)


def _rowwise_spearman_with_order(values: np.ndarray) -> np.ndarray:
    """Return rowwise spearman with order.

    Inputs:
        values: Values processed by the operation.
    Outputs:
        A value of type `np.ndarray` produced by the operation.
    Paper role:
        Statistical inference and publication-quality figure generation.
    """
    if values.ndim != 2 or values.shape[1] < 2:
        raise Phase7StatisticsError("Spearman matrix must be two-dimensional.")
    ranks = stats.rankdata(values, axis=1, method="average")
    ranks = ranks - ranks.mean(axis=1, keepdims=True)
    order = np.arange(1.0, values.shape[1] + 1.0)
    order -= order.mean()
    numerator = ranks @ order
    denominator = np.sqrt(np.sum(ranks * ranks, axis=1) * np.sum(order * order))
    output = np.full(values.shape[0], np.nan, dtype=float)
    valid = denominator > 0.0
    output[valid] = numerator[valid] / denominator[valid]
    return output


def exact_spearman_permutation(
    first: Iterable[object],
    second: Iterable[object],
) -> dict[str, float | int | str]:
    """Calculate an exact two-sided Spearman permutation test.

    Function:
        Enumerate every permutation of the second vector and calculate the
        exact two-sided p-value for a small, fixed number of ordered levels.
    Input:
        Two finite vectors of equal length.  The Phase 7 epsilon analysis uses
        the five epsilon levels and their five corresponding mean losses.
    Output:
        Spearman rho, exact p-value, number of enumerated permutations, and the
        recorded p-value method.
    Paper section:
        Statistical analysis of privacy-budget trends.
    Mathematical definition:
        p = |{pi: |rho(x, pi(y))| >= |rho_obs|}| / n!, evaluated exhaustively.
    """

    x = _numeric_vector(first, "first Spearman vector")
    y = _numeric_vector(second, "second Spearman vector")
    if x.size != y.size:
        raise Phase7StatisticsError("Spearman vectors must have equal length.")
    if x.size < 2:
        raise Phase7StatisticsError("Spearman vectors must contain at least two values.")
    observed = stats.spearmanr(x, y)
    rho = float(observed.statistic)
    if not math.isfinite(rho):
        return {
            "rho": np.nan,
            "p_exact": np.nan,
            "n_permutations": math.factorial(int(x.size)),
            "p_method": "exact_two_sided_permutation",
        }

    threshold = abs(rho) - 1e-12
    extreme = 0
    total = 0
    for permuted in itertools.permutations(y.tolist()):
        statistic = float(stats.spearmanr(x, np.asarray(permuted, dtype=float)).statistic)
        total += 1
        if math.isfinite(statistic) and abs(statistic) >= threshold:
            extreme += 1
    return {
        "rho": rho,
        "p_exact": float(extreme / total),
        "n_permutations": int(total),
        "p_method": "exact_two_sided_permutation",
    }


def _spearman_epsilon_trend(
    wide: pd.DataFrame,
    *,
    n_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Return spearman epsilon trend.

    Inputs:
        wide: Input value used by this operation.
        n_resamples: Input value used by this operation.
        bootstrap_seed: Input value used by this operation.
    Outputs:
        A value of type `dict[str, Any]` produced by the operation.
    Paper role:
        Statistical inference and publication-quality figure generation.
    """
    matrix = wide.loc[:, EPSILON_ORDER].to_numpy(dtype=float)
    means = matrix.mean(axis=0)
    exact = exact_spearman_permutation(np.asarray(EPSILON_ORDER), means)
    rho = float(exact["rho"])
    p_value = float(exact["p_exact"])
    if not math.isfinite(rho):
        return {
            "rho": np.nan,
            "p_raw": np.nan,
            "p_method": exact["p_method"],
            "n_permutations": int(exact["n_permutations"]),
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_epsilon_levels": len(EPSILON_ORDER),
            "n_seed_clusters": matrix.shape[0],
            "mean_nonincreasing": bool(np.all(np.diff(means) <= 0.0)),
            "strictly_decreasing": bool(np.all(np.diff(means) < 0.0)),
            "status": "undefined_constant_epsilon_means",
        }
    rng = np.random.default_rng(int(bootstrap_seed))
    indices = rng.integers(
        0, matrix.shape[0], size=(int(n_resamples), matrix.shape[0])
    )
    bootstrap_means = matrix[indices, :].mean(axis=1)
    bootstrap_rho = _rowwise_spearman_with_order(bootstrap_means)
    finite = bootstrap_rho[np.isfinite(bootstrap_rho)]
    if finite.size == 0:
        lower = upper = np.nan
        status = "bootstrap_undefined"
    else:
        lower, upper = np.quantile(finite, [0.025, 0.975])
        status = "ok"
    return {
        "rho": rho,
        "p_raw": p_value,
        "p_method": exact["p_method"],
        "n_permutations": int(exact["n_permutations"]),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_epsilon_levels": len(EPSILON_ORDER),
        "n_seed_clusters": int(matrix.shape[0]),
        "mean_nonincreasing": bool(np.all(np.diff(means) <= 0.0)),
        "strictly_decreasing": bool(np.all(np.diff(means) < 0.0)),
        "status": status,
    }


def build_spearman_trends(
    frame: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    bootstrap_seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Estimate epsilon-loss trends without pooling repeated seed observations.

    Input: Phase 6 enriched results.  Output: Spearman rho on the five
    epsilon-level means, exact two-sided permutation p-value, and seed-cluster
    bootstrap CI.  This avoids treating 150 repeated rows as independent,
    avoids an invalid small-sample asymptotic p-value, and records monotonicity
    only as an observed property.  Paper step: Phase 7 epsilon trend analysis.
    """

    frame = _validate_main_frame(frame)
    rows: list[dict[str, Any]] = []
    full = frame[frame["method"].eq("DP-full")]
    for (dataset, strategy), group in full.groupby(
        ["dataset", "budget_strategy"], sort=False
    ):
        for metric in MAIN_METRICS:
            wide = _validate_seed_grid(
                group,
                treatment_column="epsilon",
                treatments=EPSILON_ORDER,
                metric=metric,
                role=f"Spearman {dataset}/{strategy}/{metric}",
            )
            trend = _spearman_epsilon_trend(
                wide,
                n_resamples=n_resamples,
                bootstrap_seed=bootstrap_seed,
            )
            rows.append(
                {
                    "analysis": "dp_full_epsilon_trend",
                    "holm_family": f"dp_full_epsilon_trend|{metric}",
                    "dataset": dataset,
                    "metric": metric,
                    "budget_strategy": strategy,
                    "dp_edge_mode": pd.NA,
                    **trend,
                }
            )
    edge = frame[frame["method"].eq("DP-edge-only")]
    for (dataset, mode), group in edge.groupby(["dataset", "dp_edge_mode"], sort=False):
        wide = _validate_seed_grid(
            group,
            treatment_column="epsilon",
            treatments=EPSILON_ORDER,
            metric="L_structure",
            role=f"edge Spearman {dataset}/{mode}",
        )
        trend = _spearman_epsilon_trend(
            wide,
            n_resamples=n_resamples,
            bootstrap_seed=bootstrap_seed,
        )
        rows.append(
            {
                "analysis": "dp_edge_only_epsilon_trend",
                "holm_family": "dp_edge_only_epsilon_trend|L_structure",
                "dataset": dataset,
                "metric": "L_structure",
                "budget_strategy": pd.NA,
                "dp_edge_mode": mode,
                **trend,
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm"] = np.nan
    result["reject_holm"] = False
    for _, indexes in result.groupby("holm_family", sort=False).groups.items():
        positions = [index for index in indexes if math.isfinite(result.at[index, "p_raw"])]
        if positions:
            adjusted, rejected = holm_adjust(result.loc[positions, "p_raw"], alpha=alpha)
            result.loc[positions, "p_holm"] = adjusted
            result.loc[positions, "reject_holm"] = rejected
    result["reject_holm"] = result["reject_holm"].astype(bool)
    result["alpha"] = float(alpha)
    result["bootstrap_resamples"] = int(n_resamples)
    result["bootstrap_seed"] = int(bootstrap_seed)
    return result.sort_values(
        ["analysis", "dataset", "metric", "budget_strategy", "dp_edge_mode"],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def build_stage_summary(
    frame: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Summarize independent clipping, DP-noise, and total-loss comparisons.

    Input: Phase 6 stage result table.  Output: DP-full/uniform summaries across
    30 seeds for every dataset, epsilon, stage, and loss metric.  Paper step:
    Phase 7 RQ2 stage comparison; the output does not assert additivity.
    """

    required = {
        "dataset", "epsilon", "method", "seed", "budget_strategy", "stage",
        "BUL", "L_structure", "L_temporal", "L_information",
    }
    missing = required - set(frame.columns)
    if missing:
        raise Phase7StatisticsError(f"Stage result is missing: {sorted(missing)!r}")
    selected = frame[
        frame["method"].eq("DP-full") & frame["budget_strategy"].eq("uniform")
    ].copy()
    if len(selected) != 3 * 5 * 30 * 3:
        raise Phase7StatisticsError("Stage DP-full/uniform grid is incomplete.")
    selected = selected.rename(columns={"BUL": "BUL_equal"})
    rows: list[dict[str, Any]] = []
    keys_columns = ["dataset", "epsilon", "stage"]
    for keys, group in selected.groupby(keys_columns, sort=False):
        if set(pd.to_numeric(group["seed"]).astype(int)) != set(SEEDS) or len(group) != 30:
            raise Phase7StatisticsError(f"Stage seed grid is incomplete for {keys!r}.")
        for metric in MAIN_METRICS:
            summary = bootstrap_mean_ci(
                pd.to_numeric(group[metric]),
                n_resamples=n_resamples,
                seed=bootstrap_seed,
            )
            rows.append(
                {
                    "dataset": keys[0],
                    "epsilon": float(keys[1]),
                    "stage": keys[2],
                    "budget_strategy": "uniform",
                    "metric": metric,
                    "additive_decomposition_claim": False,
                    **summary,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "epsilon", "stage", "metric"], kind="mergesort"
    ).reset_index(drop=True)


def build_ablation_summary(
    frame: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Summarize full and dimension-ablated BUL scores across seeds.

    Input: Phase 6 dimension-ablation table.  Output: bootstrap summaries for
    full BUL and each one-dimension-removed score.  Paper step: Phase 7 RQ3
    dimension complementarity analysis.
    """

    required = {"dataset", "epsilon", "seed", "budget_strategy", *ABLATION_METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise Phase7StatisticsError(f"Ablation result is missing: {sorted(missing)!r}")
    if len(frame) != 2250:
        raise Phase7StatisticsError("Ablation result must contain 2250 rows.")
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["dataset", "epsilon", "budget_strategy"], sort=False
    ):
        if len(group) != 30 or set(pd.to_numeric(group["seed"]).astype(int)) != set(SEEDS):
            raise Phase7StatisticsError(f"Ablation seed grid is incomplete for {keys!r}.")
        for metric in ABLATION_METRICS:
            summary = bootstrap_mean_ci(
                pd.to_numeric(group[metric]),
                n_resamples=n_resamples,
                seed=bootstrap_seed,
            )
            rows.append(
                {
                    "dataset": keys[0],
                    "epsilon": float(keys[1]),
                    "budget_strategy": keys[2],
                    "metric": metric,
                    **summary,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "epsilon", "budget_strategy", "metric"], kind="mergesort"
    ).reset_index(drop=True)


def build_ablation_ranking_consistency(
    frame: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Measure strategy-ranking stability after removing each BUL dimension.

    Input: complete Phase 6 ablation table.  Output: Spearman correlation between
    five strategy means under full and ablated BUL, with seed-cluster bootstrap
    CI.  Paper step: Phase 7 ablation ranking-consistency analysis.
    """

    required = {"dataset", "epsilon", "seed", "budget_strategy", *ABLATION_METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise Phase7StatisticsError(f"Ablation result is missing: {sorted(missing)!r}")
    rows: list[dict[str, Any]] = []
    for (dataset, epsilon), group in frame.groupby(["dataset", "epsilon"], sort=False):
        expected = {(seed, strategy) for seed in SEEDS for strategy in STRATEGY_ORDER}
        actual = set(zip(pd.to_numeric(group["seed"]).astype(int), group["budget_strategy"]))
        if actual != expected or len(group) != len(expected):
            raise Phase7StatisticsError(
                f"Ablation strategy/seed grid is incomplete for {dataset}/{epsilon}."
            )
        full = group.pivot(index="seed", columns="budget_strategy", values="BUL_equal")
        full = full.reindex(index=SEEDS, columns=STRATEGY_ORDER)
        for ablated in ABLATION_METRICS[1:]:
            other = group.pivot(index="seed", columns="budget_strategy", values=ablated)
            other = other.reindex(index=SEEDS, columns=STRATEGY_ORDER)
            full_matrix = full.to_numpy(dtype=float)
            other_matrix = other.to_numpy(dtype=float)
            full_means = full_matrix.mean(axis=0)
            other_means = other_matrix.mean(axis=0)
            observed = exact_spearman_permutation(full_means, other_means)
            rng = np.random.default_rng(int(bootstrap_seed))
            indices = rng.integers(0, len(SEEDS), size=(int(n_resamples), len(SEEDS)))
            full_boot = full_matrix[indices, :].mean(axis=1)
            other_boot = other_matrix[indices, :].mean(axis=1)
            full_ranks = stats.rankdata(full_boot, axis=1, method="average")
            other_ranks = stats.rankdata(other_boot, axis=1, method="average")
            full_ranks -= full_ranks.mean(axis=1, keepdims=True)
            other_ranks -= other_ranks.mean(axis=1, keepdims=True)
            denominator = np.sqrt(
                np.sum(full_ranks * full_ranks, axis=1)
                * np.sum(other_ranks * other_ranks, axis=1)
            )
            correlations = np.full(int(n_resamples), np.nan, dtype=float)
            valid = denominator > 0.0
            correlations[valid] = np.sum(full_ranks[valid] * other_ranks[valid], axis=1) / denominator[valid]
            finite = correlations[np.isfinite(correlations)]
            lower, upper = np.quantile(finite, [0.025, 0.975])
            rows.append(
                {
                    "dataset": dataset,
                    "epsilon": float(epsilon),
                    "full_metric": "BUL_equal",
                    "ablated_metric": ablated,
                    "rho": float(observed["rho"]),
                    "p_raw": float(observed["p_exact"]),
                    "p_method": observed["p_method"],
                    "n_permutations": int(observed["n_permutations"]),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "n_strategies": len(STRATEGY_ORDER),
                    "n_seed_clusters": len(SEEDS),
                    "bootstrap_resamples": int(n_resamples),
                    "bootstrap_seed": int(bootstrap_seed),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "epsilon", "ablated_metric"], kind="mergesort"
    ).reset_index(drop=True)


def build_sensitivity_summary(
    frame: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Summarize the approved BPIC2017 Top-k/tau sensitivity grid.

    Input: Phase 6 sensitivity result with 810 complete rows.  Output: mean,
    spread, and bootstrap CI for all main loss metrics at every parameter cell.
    Paper step: Phase 7 RQ5 parameter-robustness analysis.
    """

    required = {
        "dataset", "method", "budget_strategy", "epsilon", "seed",
        "top_k_variants", "min_edge_support", "status", *MAIN_METRICS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise Phase7StatisticsError(f"Sensitivity result is missing: {sorted(missing)!r}")
    if (
        len(frame) != 810
        or set(frame["dataset"]) != {"BPIC2017"}
        or set(frame["method"]) != {"DP-full"}
        or set(frame["budget_strategy"]) != {"uniform"}
        or set(frame["status"]) != {"success"}
    ):
        raise Phase7StatisticsError("Sensitivity experiment scope changed.")
    rows: list[dict[str, Any]] = []
    keys_columns = ["top_k_variants", "min_edge_support", "epsilon"]
    for keys, group in frame.groupby(keys_columns, sort=False):
        if len(group) != 30 or set(pd.to_numeric(group["seed"]).astype(int)) != set(SEEDS):
            raise Phase7StatisticsError(f"Sensitivity seed grid is incomplete for {keys!r}.")
        for metric in MAIN_METRICS:
            summary = bootstrap_mean_ci(
                pd.to_numeric(group[metric]),
                n_resamples=n_resamples,
                seed=bootstrap_seed,
            )
            rows.append(
                {
                    "dataset": "BPIC2017",
                    "method": "DP-full",
                    "budget_strategy": "uniform",
                    "top_k_variants": int(keys[0]),
                    "min_edge_support": int(keys[1]),
                    "epsilon": float(keys[2]),
                    "metric": metric,
                    **summary,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["epsilon", "top_k_variants", "min_edge_support", "metric"],
        kind="mergesort",
    ).reset_index(drop=True)


# Natural public error name; the historical class remains for compatibility.
StatisticalTestError = Phase7StatisticsError

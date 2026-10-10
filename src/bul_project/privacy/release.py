"""Vector releases for DP-full and the two DP-edge-only fairness modes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal, TypeVar

import numpy as np

from bul_project.behavior.models import Edge, GRawStatistics, Variant
from bul_project.config import DEFAULT_DP_EDGE_MODE, DP_DENOMINATOR_DELTA, PRIVACY_COMPONENTS

from ._validation import validate_experiment_seed
from .budget import (
    allocate_privacy_budget,
    edge_only_epsilon,
    split_temporal_budget,
)
from .laplace_dp import laplace_release
from .models import DPReleaseStatistics
from .sensitivity import calculate_sensitivity


KeyT = TypeVar("KeyT")


def _validate_g_clip(g_clip: GRawStatistics) -> GRawStatistics:
    """Validate g clip.

    Inputs:
        g_clip: Input value used by this operation.
    Outputs:
        A value of type `GRawStatistics` produced by the operation.
    Paper role:
        Case-level differential-privacy release and temporal post-processing.
    """
    if not isinstance(g_clip, GRawStatistics):
        raise TypeError("g_clip must be a GRawStatistics object built from clipped traces.")
    if g_clip.trace_count <= 0:
        raise ValueError("g_clip must contain at least one case.")
    return g_clip


def _normalize_released_counts(counts: Mapping[KeyT, int]) -> dict[KeyT, float]:
    """Normalize post-processed counts, using a public uniform zero-mass fallback."""

    if any(value < 0 for value in counts.values()):
        raise RuntimeError("Post-processed DP counts must be non-negative.")
    if not counts:
        return {}
    total = sum(counts.values())
    if total == 0:
        probability = 1.0 / len(counts)
        return {key: probability for key in counts}
    return {key: value / total for key, value in counts.items()}


def _release_counts(
    counts: Mapping[KeyT, int],
    sensitivity: float,
    epsilon: float,
    rng: np.random.Generator,
) -> dict[KeyT, int]:
    """Release counts.

    Inputs:
        counts: Input value used by this operation.
        sensitivity: Input value used by this operation.
        epsilon: Differential-privacy budget for the condition.
        rng: Input value used by this operation.
    Outputs:
        A value of type `dict[KeyT, int]` produced by the operation.
    Paper role:
        Case-level differential-privacy release and temporal post-processing.
    """
    return {
        key: int(
            laplace_release(
                counts[key],
                sensitivity,
                epsilon,
                statistic="count",
                rng=rng,
            )
        )
        for key in sorted(counts)
    }


def _release_time_sums(
    sums: Mapping[Edge, float],
    sensitivity: float,
    epsilon: float,
    rng: np.random.Generator,
) -> dict[Edge, float]:
    """Release time sums.

    Inputs:
        sums: Input value used by this operation.
        sensitivity: Input value used by this operation.
        epsilon: Differential-privacy budget for the condition.
        rng: Input value used by this operation.
    Outputs:
        A value of type `dict[Edge, float]` produced by the operation.
    Paper role:
        Case-level differential-privacy release and temporal post-processing.
    """
    return {
        edge: float(
            laplace_release(
                sums[edge],
                sensitivity,
                epsilon,
                statistic="time",
                rng=rng,
            )
        )
        for edge in sorted(sums)
    }


def dp_full_release(
    g_clip: GRawStatistics,
    *,
    l_max: int,
    t_max: float,
    epsilon_total: float,
    strategy: str = "uniform",
    seed: int,
) -> DPReleaseStatistics:
    """Release all four clipped statistic families under sequential composition.

    Input
    -----
    g_clip:
        Phase 2 statistics rebuilt from ``clip_traces``. Its Top-k+OTHER and
        low-support edge views define one release domain before noise.
    l_max, t_max:
        The exact bounds used to construct ``g_clip``.
    epsilon_total:
        One of the configured main experiment values ``[0.5,1,2,4,8]`` (the
        mechanism remains valid for any finite positive epsilon).
    strategy:
        Uniform or one of the four configured focus strategies.
    seed:
        Reproducibility seed in the configured range 0 through 29.

    Output
    ------
    DPReleaseStatistics
        ``G_DP`` with noisy activity, edge and variant counts, noisy first-
        and second-moment time sums, normalized distributions, and means
        derived only from noisy first-moment sums and noisy edge counts.

    Paper experiment step
    ---------------------
    Overall Steps 6-7 plus Phase 6 temporal extension: release the four
    statistic families with independent Laplace draws. Inside the time family,
    release ``S'_T`` and ``Q'_T`` with ``epsilon_T/2`` each under sequential
    composition. Top-k and edge support are never reselected after noise.
    """

    clipped = _validate_g_clip(g_clip)
    experiment_seed = validate_experiment_seed(seed)
    sensitivities = calculate_sensitivity(l_max, t_max)
    budget = allocate_privacy_budget(epsilon_total, strategy)
    temporal_epsilons = split_temporal_budget(budget.time)
    rng = np.random.default_rng(experiment_seed)

    activity_counts = _release_counts(
        clipped.activity.counts,
        sensitivities.activity,
        budget.activity,
        rng,
    )

    # Phase 2 always keeps the full DFG separately.  Only the clipped, frozen
    # filtered view is a release/evaluation domain when tau is configured.
    edge_domain = clipped.dfg.filtered_counts
    edge_counts = _release_counts(
        edge_domain,
        sensitivities.edge,
        budget.edge,
        rng,
    )

    variant_counts = _release_counts(
        clipped.variants.top_counts,
        sensitivities.variant,
        budget.variant,
        rng,
    )
    variant_other_count: int | None = None
    if clipped.variants.top_k is not None:
        variant_other_count = int(
            laplace_release(
                clipped.variants.other_count,
                sensitivities.variant,
                budget.variant,
                statistic="count",
                rng=rng,
            )
        )

    combined_variant_total = sum(variant_counts.values()) + (
        variant_other_count if variant_other_count is not None else 0
    )
    combined_variant_size = len(variant_counts) + (
        1 if variant_other_count is not None else 0
    )
    if combined_variant_total > 0:
        variant_probabilities = {
            key: value / combined_variant_total
            for key, value in variant_counts.items()
        }
        variant_other_probability = (
            variant_other_count / combined_variant_total
            if variant_other_count is not None
            else None
        )
    elif combined_variant_size > 0:
        fallback = 1.0 / combined_variant_size
        variant_probabilities = {key: fallback for key in variant_counts}
        variant_other_probability = (
            fallback if variant_other_count is not None else None
        )
    else:
        variant_probabilities = {}
        variant_other_probability = None

    time_input = {
        edge: clipped.temporal.sums.get(edge, 0.0) for edge in edge_domain
    }
    time_squared_input = {
        edge: sum(
            interval**2
            for interval in clipped.temporal.samples.get(edge, ())
        )
        for edge in edge_domain
    }
    time_sums = _release_time_sums(
        time_input,
        sensitivities.time,
        temporal_epsilons["time_sum"],
        rng,
    )
    time_squared_sums = _release_time_sums(
        time_squared_input,
        sensitivities.time_squared,
        temporal_epsilons["time_squared_sum"],
        rng,
    )
    time_means = {
        edge: time_sums[edge] / max(edge_counts[edge], DP_DENOMINATOR_DELTA)
        for edge in time_sums
    }

    component_epsilons = {
        "activity": budget.activity,
        "edge": budget.edge,
        "variant": budget.variant,
        "time": budget.time,
    }
    if not math.isclose(
        sum(temporal_epsilons.values()),
        budget.time,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Temporal first- and second-moment budgets do not compose to epsilon_T."
        )
    if not math.isclose(
        sum(component_epsilons.values()),
        budget.epsilon_total,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("DP-full component budgets do not compose to epsilon_total.")

    return DPReleaseStatistics(
        method="DP-full",
        epsilon_total=budget.epsilon_total,
        epsilon_spent=budget.composed_epsilon,
        component_epsilons=component_epsilons,
        temporal_epsilons=temporal_epsilons,
        budget_strategy=budget.strategy,
        dp_edge_mode=None,
        seed=experiment_seed,
        sensitivities=sensitivities,
        top_k_variants=clipped.top_k_variants,
        min_edge_support=clipped.min_edge_support,
        activity_counts=activity_counts,
        activity_probabilities=_normalize_released_counts(activity_counts),
        edge_counts=edge_counts,
        edge_probabilities=_normalize_released_counts(edge_counts),
        variant_counts=variant_counts,
        variant_probabilities=variant_probabilities,
        variant_other_count=variant_other_count,
        variant_other_probability=variant_other_probability,
        time_sums=time_sums,
        time_squared_sums=time_squared_sums,
        time_means=time_means,
    )


def dp_edge_only_release(
    g_clip: GRawStatistics,
    *,
    l_max: int,
    t_max: float,
    epsilon_total: float,
    seed: int,
    mode: str = DEFAULT_DP_EDGE_MODE,
) -> DPReleaseStatistics:
    """Release only the clipped DFG edge-count vector in one fairness mode.

    Input
    -----
    g_clip:
        Statistics built from clipped traces; its filtered edge set is frozen
        before noise and is the only released component.
    l_max, t_max:
        Clipping bounds. ``t_max`` is retained in reproducibility metadata but
        does not affect the edge-count sensitivity.
    epsilon_total:
        Experiment-level privacy budget.
    seed:
        Reproducibility seed in the configured range 0 through 29.
    mode:
        ``component`` (main experiment) assigns ``epsilon_total/4`` to edges;
        ``mechanism`` (appendix) assigns the full ``epsilon_total``.

    Output
    ------
    DPReleaseStatistics
        A ``G_DP`` result containing only noisy edge counts/probabilities. Raw
        activity, variant, and temporal statistics are not exposed as part of
        the DP-edge-only release.

    Paper experiment step
    ---------------------
    Overall Steps 6-8: implement the component-fair main baseline and the
    mechanism-fair appendix baseline as distinct, recorded modes.
    """

    clipped = _validate_g_clip(g_clip)
    experiment_seed = validate_experiment_seed(seed)
    sensitivities = calculate_sensitivity(l_max, t_max)
    epsilon_edge = edge_only_epsilon(epsilon_total, mode)
    rng = np.random.default_rng(experiment_seed)
    edge_counts = _release_counts(
        clipped.dfg.filtered_counts,
        sensitivities.edge,
        epsilon_edge,
        rng,
    )
    component_epsilons = {name: 0.0 for name in PRIVACY_COMPONENTS}
    component_epsilons["edge"] = epsilon_edge

    return DPReleaseStatistics(
        method="DP-edge-only",
        epsilon_total=float(epsilon_total),
        epsilon_spent=epsilon_edge,
        component_epsilons=component_epsilons,
        temporal_epsilons=None,
        budget_strategy=None,
        dp_edge_mode=mode,
        seed=experiment_seed,
        sensitivities=sensitivities,
        top_k_variants=clipped.top_k_variants,
        min_edge_support=clipped.min_edge_support,
        activity_counts=None,
        activity_probabilities=None,
        edge_counts=edge_counts,
        edge_probabilities=_normalize_released_counts(edge_counts),
        variant_counts=None,
        variant_probabilities=None,
        variant_other_count=None,
        variant_other_probability=None,
        time_sums=None,
        time_squared_sums=None,
        time_means=None,
    )


def release_g_dp(
    g_clip: GRawStatistics,
    *,
    method: Literal["DP-full", "DP-edge-only"],
    l_max: int,
    t_max: float,
    epsilon_total: float,
    seed: int,
    strategy: str = "uniform",
    dp_edge_mode: str = DEFAULT_DP_EDGE_MODE,
) -> DPReleaseStatistics:
    """Dispatch one explicit DP-full or DP-edge-only Phase 3 release.

    Input
    -----
    g_clip, l_max, t_max, epsilon_total, seed:
        Shared clipped-statistics and reproducibility inputs.
    method:
        Exactly ``DP-full`` or ``DP-edge-only``.
    strategy:
        DP-full budget allocation; ignored by DP-edge-only, whose fairness
        allocation is controlled only by ``dp_edge_mode``.
    dp_edge_mode:
        Component or mechanism fairness for DP-edge-only.

    Output
    ------
    DPReleaseStatistics
        Typed ``G_DP`` output with the selected mechanism recorded.

    Paper experiment step
    ---------------------
    Overall Step 8: keep the DP-full and DP-edge-only experimental groups
    explicit and reproducible without mixing their budget definitions.
    """

    if method == "DP-full":
        return dp_full_release(
            g_clip,
            l_max=l_max,
            t_max=t_max,
            epsilon_total=epsilon_total,
            strategy=strategy,
            seed=seed,
        )
    if method == "DP-edge-only":
        return dp_edge_only_release(
            g_clip,
            l_max=l_max,
            t_max=t_max,
            epsilon_total=epsilon_total,
            seed=seed,
            mode=dp_edge_mode,
        )
    raise ValueError("method must be either 'DP-full' or 'DP-edge-only'.")

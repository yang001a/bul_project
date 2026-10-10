"""Sequential privacy-budget allocation for Phase 3."""

from __future__ import annotations

import math

from bul_project.config import (
    DEFAULT_DP_EDGE_MODE,
    DP_EDGE_COMPONENT_SHARE,
    DP_EDGE_MODES,
    PRIVACY_BUDGET_RATIOS,
    PRIVACY_COMPONENTS,
    TEMPORAL_BUDGET_COMPONENTS,
    TEMPORAL_BUDGET_RATIOS,
)

from ._validation import validate_epsilon
from .models import PrivacyBudget


def allocate_privacy_budget(
    epsilon_total: float, strategy: str = "uniform"
) -> PrivacyBudget:
    """Allocate ``epsilon_total`` by the configured sequential strategy.

    Input
    -----
    epsilon_total:
        Positive global privacy budget for a DP-full release.
    strategy:
        One of ``uniform``, ``activity_focus``, ``edge_focus``,
        ``variant_focus``, or ``time_focus``.

    Output
    ------
    PrivacyBudget
        Four component budgets satisfying
        ``epsilon_A + epsilon_E + epsilon_V + epsilon_T = epsilon_total``.

    Paper experiment step
    ---------------------
    Overall Step 6/16: apply statistic-family-level sequential composition.
    Uniform assigns 25% each; a focus strategy assigns 40% to its named
    family and 20% to every other family.
    """

    total = validate_epsilon(epsilon_total, "epsilon_total")
    try:
        ratios = PRIVACY_BUDGET_RATIOS[strategy]
    except KeyError as exc:
        choices = ", ".join(PRIVACY_BUDGET_RATIOS)
        raise ValueError(f"Unknown budget strategy {strategy!r}; expected {choices}.") from exc
    if tuple(ratios) != PRIVACY_COMPONENTS or not math.isclose(
        sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("Configured privacy-budget ratios must cover four families and sum to 1.")

    allocated = {name: total * ratios[name] for name in PRIVACY_COMPONENTS}
    budget = PrivacyBudget(
        epsilon_total=total,
        strategy=strategy,
        activity=allocated["activity"],
        edge=allocated["edge"],
        variant=allocated["variant"],
        time=allocated["time"],
    )
    if not math.isclose(
        budget.composed_epsilon, total, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("Sequential privacy-budget composition failed.")
    return budget


def split_temporal_budget(epsilon_time: float) -> dict[str, float]:
    """Split the time-family budget between first and second moments.

    Input
    -----
    epsilon_time:
        Positive family-level time budget already allocated from epsilon_total.

    Output
    ------
    dict[str, float]
        Budgets for time_sum and time_squared_sum. Each receives one half of
        epsilon_time and their sequential sum equals epsilon_time.

    Paper experiment step
    ---------------------
    Phase 6 temporal DP release: apply sequential composition inside the time
    family, with epsilon_ST=epsilon_T/2 and epsilon_QT=epsilon_T/2. This is not
    parallel composition.
    """

    time_budget = validate_epsilon(epsilon_time, "epsilon_time")
    configured_components_match = (
        tuple(TEMPORAL_BUDGET_RATIOS) == TEMPORAL_BUDGET_COMPONENTS
    )
    configured_ratios_sum_to_one = math.isclose(
        sum(TEMPORAL_BUDGET_RATIOS.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    if not configured_components_match or not configured_ratios_sum_to_one:
        raise RuntimeError(
            "Configured temporal-budget ratios must cover both moments and sum to 1."
        )
    allocated = {
        name: time_budget * TEMPORAL_BUDGET_RATIOS[name]
        for name in TEMPORAL_BUDGET_COMPONENTS
    }
    if not math.isclose(
        sum(allocated.values()),
        time_budget,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Sequential temporal-budget composition failed.")
    return allocated


def edge_only_epsilon(
    epsilon_total: float, mode: str = DEFAULT_DP_EDGE_MODE
) -> float:
    """Return the edge budget for either DP-edge-only fairness mode.

    Input
    -----
    epsilon_total:
        Positive experiment-level epsilon.
    mode:
        ``component`` for the main experiment or ``mechanism`` for the
        appendix mechanism-level comparison.

    Output
    ------
    float
        ``epsilon_total/4`` in component mode and ``epsilon_total`` in
        mechanism mode.

    Paper experiment step
    ---------------------
    Overall Step 6: keep DP-edge-only fairness definitions separate and use
    component mode as the configured default.
    """

    total = validate_epsilon(epsilon_total, "epsilon_total")
    if mode not in DP_EDGE_MODES:
        choices = ", ".join(DP_EDGE_MODES)
        raise ValueError(f"Unknown DP-edge-only mode {mode!r}; expected {choices}.")
    if mode == "component":
        return total * DP_EDGE_COMPONENT_SHARE
    return total

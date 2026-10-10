"""Edge-weighted normalized EMD for the temporal behavior dimension."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from bul_project.behavior.models import Edge
from bul_project.metrics import calculate_normalized_emd

from ._utils import validate_weights


def calculate_temporal_loss(
    reference_times: Mapping[Edge, Sequence[float] | np.ndarray],
    comparison_times: Mapping[Edge, Sequence[float] | np.ndarray],
    *,
    edge_weights: Mapping[Edge, float],
    normalization_scales: Mapping[Edge, float],
) -> float:
    """Calculate weighted transition-level nEMD on a frozen effective edge set.

    Input
    -----
    reference_times, comparison_times:
        Per-edge transition-time samples. Every edge selected by
        ``edge_weights`` must exist with non-empty samples in both mappings.
    edge_weights:
        Fixed ``w_e`` values, normally reference edge probabilities
        renormalized over ``E_m``; keys define ``E_m`` and values must sum to 1.
    normalization_scales:
        Pre-fixed, epsilon-independent ``Z_e`` for every edge in ``E_m``.

    Output
    ------
    float
        ``L_temporal = sum(w_e*nEMD_e)`` in ``[0,1]``.

    Paper experiment step
    ---------------------
    Overall Step 9.2/10: compare waiting-time distributions separately per
    transition edge and aggregate with fixed reference weights. The effective
    edge set and normalization scales cannot be reselected per noisy result.
    """

    if not edge_weights:
        raise ValueError("edge_weights must define at least one effective edge.")
    edges = tuple(edge_weights)
    checked_weights = validate_weights(edge_weights, edges, "edge_weights")

    missing_reference = set(edges) - set(reference_times)
    missing_comparison = set(edges) - set(comparison_times)
    missing_scales = set(edges) - set(normalization_scales)
    if missing_reference:
        raise ValueError(f"reference_times is missing effective edges: {missing_reference!r}")
    if missing_comparison:
        raise ValueError(f"comparison_times is missing effective edges: {missing_comparison!r}")
    if missing_scales:
        raise ValueError(f"normalization_scales is missing edges: {missing_scales!r}")

    normalized_losses = {
        edge: calculate_normalized_emd(
            reference_times[edge],
            comparison_times[edge],
            normalization_scales[edge],
        )
        for edge in edges
    }
    total = sum(
        checked_weights[edge] * normalized_losses[edge] for edge in edges
    )
    return float(np.clip(total, 0.0, 1.0))

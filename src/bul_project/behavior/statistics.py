"""Orchestration of the four Phase 2 behavior-statistics families."""

from __future__ import annotations

from collections.abc import Sequence

from bul_project.preprocessing.models import Trace

from ._utils import validate_traces
from .activity import activity_statistics
from .dfg import dfg_statistics
from .models import GRawStatistics
from .temporal import temporal_statistics
from .variants import variant_statistics


def build_g_raw(
    traces: Sequence[Trace],
    *,
    top_k_variants: int | None = None,
    min_edge_support: int | None = None,
) -> GRawStatistics:
    """Assemble Activity, DFG, Variant, and Temporal statistics as ``G_raw``.

    Input
    -----
    traces:
        Phase 1 trace collection.
    top_k_variants:
        Optional Top-k candidate-space size; remaining variants become OTHER.
    min_edge_support:
        Optional low-frequency edge threshold for filtered DFG/time views.

    Output
    ------
    GRawStatistics
        Full raw statistics plus non-destructive constrained-space views.

    Paper experiment step
    ---------------------
    Overall Step 3: build ``G_raw`` before clipping or differential privacy.
    This function does not import or calculate DP, JS, Cosine, EMD, Entropy,
    process-model quality, or BUL.
    """

    checked = validate_traces(traces)
    activity = activity_statistics(checked)
    dfg = dfg_statistics(checked, min_support=min_edge_support)
    variants = variant_statistics(checked, top_k=top_k_variants)
    temporal = temporal_statistics(
        checked, min_edge_support=min_edge_support
    )

    expected_events = sum(trace.length for trace in checked)
    expected_edges = sum(max(trace.length - 1, 0) for trace in checked)
    if activity.total_events != expected_events:
        raise RuntimeError("Activity-count conservation failed while building G_raw.")
    if dfg.total_edge_occurrences != expected_edges:
        raise RuntimeError("DFG edge-count conservation failed while building G_raw.")
    if temporal.total_transition_pairs != expected_edges:
        raise RuntimeError("Temporal pair-count conservation failed while building G_raw.")
    if temporal.edge_support_counts != dfg.counts:
        raise RuntimeError("Temporal edge support does not match DFG counts in G_raw.")
    if temporal.valid_interval_pairs + temporal.missing_interval_pairs != expected_edges:
        raise RuntimeError("Temporal valid/missing interval conservation failed.")
    if sum(temporal.sample_counts.values()) != temporal.valid_interval_pairs:
        raise RuntimeError("Temporal sample-count conservation failed.")
    if variants.total_cases != len(checked):
        raise RuntimeError("Variant case-count conservation failed while building G_raw.")

    return GRawStatistics(
        activity=activity,
        dfg=dfg,
        variants=variants,
        temporal=temporal,
        trace_count=len(checked),
        top_k_variants=top_k_variants,
        min_edge_support=min_edge_support,
    )

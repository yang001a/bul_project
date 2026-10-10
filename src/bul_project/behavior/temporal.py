"""Transition-level empirical time distributions for Phase 2."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from statistics import median

from bul_project.preprocessing.models import Trace

from ._utils import validate_positive_optional, validate_traces
from .models import Edge, TemporalStatistics


def temporal_statistics(
    traces: Sequence[Trace], *, min_edge_support: int | None = None
) -> TemporalStatistics:
    """Collect transition-time samples separately for each DFG edge.

    Input
    -----
    traces:
        Validated Phase 1 traces whose adjacent intervals are in seconds.
    min_edge_support:
        Optional threshold applied to the raw DFG occurrence count, not to the
        number of available time samples. ``None`` keeps all edge types.

    Output
    ------
    TemporalStatistics
        Full per-edge samples, sum/mean/median, missing-pair counts, and a
        separate effective-edge view for the requested support threshold.

    Paper experiment step
    ---------------------
    Overall Step 3 constructs transition time samples ``T_e`` and time sums
    ``S_T(a,b)``. Step 9.2/Step 16 uses support threshold ``tau`` to stabilize
    the later time-loss comparison. No EMD or other metric is calculated here.
    """

    validate_positive_optional(min_edge_support, "min_edge_support")
    checked = validate_traces(traces)
    support: Counter[Edge] = Counter()
    sample_lists: defaultdict[Edge, list[float]] = defaultdict(list)
    missing_intervals = 0

    for trace in checked:
        for source, target, interval in zip(
            trace.activities,
            trace.activities[1:],
            trace.transition_seconds,
        ):
            edge = (source, target)
            support[edge] += 1
            sample_lists[edge]  # retain edges even when every time is missing
            if interval is None:
                missing_intervals += 1
                continue
            value = float(interval)
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite transition interval for edge {edge}: {value}"
                )
            if value < 0:
                raise ValueError(
                    f"Negative transition interval for edge {edge}: {value}; "
                    "Phase 2 never applies abs()."
                )
            sample_lists[edge].append(value)

    edge_support_counts = dict(sorted(support.items()))
    samples = {
        edge: tuple(sample_lists[edge]) for edge in edge_support_counts
    }
    sample_counts = {edge: len(values) for edge, values in samples.items()}
    sums = {edge: float(sum(values)) for edge, values in samples.items()}
    observed_means = {
        edge: (sums[edge] / len(values) if values else None)
        for edge, values in samples.items()
    }
    edge_normalized_means = {
        edge: sums[edge] / edge_support_counts[edge] for edge in samples
    }
    medians = {
        edge: (float(median(values)) if values else None)
        for edge, values in samples.items()
    }
    effective_samples = {
        edge: values
        for edge, values in samples.items()
        if min_edge_support is None
        or edge_support_counts[edge] >= min_edge_support
    }
    total_pairs = sum(edge_support_counts.values())
    valid_pairs = sum(sample_counts.values())
    return TemporalStatistics(
        samples=samples,
        edge_support_counts=edge_support_counts,
        sample_counts=sample_counts,
        sums=sums,
        observed_means=observed_means,
        edge_normalized_means=edge_normalized_means,
        medians=medians,
        total_transition_pairs=total_pairs,
        valid_interval_pairs=valid_pairs,
        missing_interval_pairs=missing_intervals,
        min_edge_support=min_edge_support,
        effective_samples=effective_samples,
    )

"""Directly-follows graph statistics and support-space filtering."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from bul_project.preprocessing.models import Trace

from ._utils import normalize_counts, validate_positive_optional, validate_traces
from .models import DFGStatistics, Edge


def dfg_statistics(
    traces: Sequence[Trace], *, min_support: int | None = None
) -> DFGStatistics:
    """Count directly-follows edges and create a low-frequency-filtered view.

    Input
    -----
    traces:
        Validated case-level traces produced by Phase 1.
    min_support:
        Optional raw edge-count threshold ``tau``. ``None`` keeps every edge.

    Output
    ------
    DFGStatistics
        Full edge counts/probabilities and a separately renormalized filtered
        view. Full raw counts are never overwritten by the constraint.

    Paper experiment step
    ---------------------
    Overall Step 3 constructs ``C_E(a,b)``; Step 16 evaluates ``tau`` in
    ``{5,10,20}``. Filtering restricts the statistical space only and does not
    define or alter JS, EMD, Entropy, or any other metric formula.
    """

    validate_positive_optional(min_support, "min_support")
    checked = validate_traces(traces)
    counter: Counter[Edge] = Counter()
    for trace in checked:
        counter.update(zip(trace.activities, trace.activities[1:]))

    counts = dict(sorted(counter.items()))
    probabilities = normalize_counts(counts)
    if min_support is None:
        filtered_counts = dict(counts)
    else:
        filtered_counts = {
            edge: count for edge, count in counts.items() if count >= min_support
        }
    filtered_probabilities = normalize_counts(filtered_counts)
    dropped = set(counts) - set(filtered_counts)
    return DFGStatistics(
        counts=counts,
        probabilities=probabilities,
        total_edge_occurrences=sum(counts.values()),
        min_support=min_support,
        filtered_counts=filtered_counts,
        filtered_probabilities=filtered_probabilities,
        dropped_edge_types=len(dropped),
        dropped_edge_occurrences=sum(counts[edge] for edge in dropped),
    )

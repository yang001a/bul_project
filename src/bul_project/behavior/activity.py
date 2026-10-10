"""Activity-count statistics for Phase 2 ``G_raw`` construction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from bul_project.preprocessing.models import Trace

from ._utils import normalize_counts, validate_traces
from .models import ActivityStatistics


def activity_statistics(traces: Sequence[Trace]) -> ActivityStatistics:
    """Count activity occurrences across Phase 1 traces.

    Input
    -----
    traces:
        Validated case-level traces produced by Phase 1.

    Output
    ------
    ActivityStatistics
        Full activity counts, normalized probabilities, and total event count.

    Paper experiment step
    ---------------------
    Overall Step 3: construct ``G_raw`` activity count ``C_A(a)``. This function
    performs no JS, Entropy, DP noise, or BUL calculation.
    """

    checked = validate_traces(traces)
    counter: Counter[str] = Counter()
    for trace in checked:
        counter.update(trace.activities)
    counts = dict(sorted(counter.items()))
    return ActivityStatistics(
        counts=counts,
        probabilities=normalize_counts(counts),
        total_events=sum(counts.values()),
    )

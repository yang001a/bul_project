"""Trace-variant distributions with a fixed Top-k plus ``OTHER`` view."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from bul_project.preprocessing.models import Trace

from ._utils import normalize_counts, validate_positive_optional, validate_traces
from .models import Variant, VariantStatistics


def variant_statistics(
    traces: Sequence[Trace], *, top_k: int | None = None
) -> VariantStatistics:
    """Count complete trace variants and construct Top-k plus ``OTHER``.

    Input
    -----
    traces:
        Validated case-level traces produced by Phase 1.
    top_k:
        Number of most frequent variants retained in the constrained view.
        Ties are resolved deterministically by the activity tuple. ``None``
        retains the complete variant space.

    Output
    ------
    VariantStatistics
        Full counts/distribution together with Top-k counts and the aggregated
        ``OTHER`` count/probability.

    Paper experiment step
    ---------------------
    Overall Step 3 constructs ``C_V(sigma)`` and Top-k candidate variants plus
    ``OTHER``; Step 16 evaluates k in ``{50,100,200}``. This statistical-space
    operation does not change any later evaluation-metric formula.
    """

    validate_positive_optional(top_k, "top_k")
    checked = validate_traces(traces)
    counter: Counter[Variant] = Counter(trace.activities for trace in checked)
    full_counts = dict(sorted(counter.items()))
    full_probabilities = normalize_counts(full_counts)

    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    selected = ranked if top_k is None else ranked[:top_k]
    top_counts = dict(selected)
    total_cases = len(checked)
    top_probabilities = {
        variant: count / total_cases for variant, count in top_counts.items()
    }
    other_count = total_cases - sum(top_counts.values())
    return VariantStatistics(
        full_counts=full_counts,
        full_probabilities=full_probabilities,
        total_cases=total_cases,
        unique_variants=len(full_counts),
        top_k=top_k,
        top_counts=top_counts,
        top_probabilities=top_probabilities,
        other_count=other_count,
        other_probability=other_count / total_cases,
    )

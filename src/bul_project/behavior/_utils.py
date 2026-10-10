"""Internal validation and normalization helpers for Phase 2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar

from bul_project.preprocessing.models import Trace


KeyT = TypeVar("KeyT")


def validate_traces(traces: Sequence[Trace]) -> tuple[Trace, ...]:
    """Validate the Phase 1 trace contract and return an immutable view.

    Input
    -----
    traces:
        Non-empty Phase 1 trace sequence, with exactly one trace per case.

    Output
    ------
    tuple[Trace, ...]
        Validated immutable trace collection.

    Paper experiment step
    ---------------------
    Overall Step 2: generate case-level traces before constructing ``G_raw``.
    """

    materialized = tuple(traces)
    if not materialized:
        raise ValueError("Behavior statistics require at least one trace.")

    case_ids: set[str] = set()
    for trace in materialized:
        if not isinstance(trace, Trace):
            raise TypeError("All behavior-statistics inputs must be Phase 1 Trace objects.")
        if not trace.activities:
            raise ValueError(f"Trace {trace.case_id!r} contains no activities.")
        if any(not isinstance(activity, str) or not activity for activity in trace.activities):
            raise ValueError(
                f"Trace {trace.case_id!r} contains an invalid activity label."
            )
        if trace.case_id in case_ids:
            raise ValueError(f"Duplicate case_id in trace input: {trace.case_id!r}")
        case_ids.add(trace.case_id)
        if len(trace.timestamps) != trace.length:
            raise ValueError(
                f"Trace {trace.case_id!r} timestamp length does not match activities."
            )
        if len(trace.transition_seconds) != max(trace.length - 1, 0):
            raise ValueError(
                f"Trace {trace.case_id!r} transition length does not match activities."
            )
    return materialized


def normalize_counts(counts: Mapping[KeyT, int]) -> dict[KeyT, float]:
    """Normalize non-negative counts without applying any evaluation metric.

    Input
    -----
    counts:
        Mapping of statistical-space keys to non-negative integer counts.

    Output
    ------
    dict
        Probability mapping whose values sum to one when the total is positive.

    Paper experiment step
    ---------------------
    Overall Step 3: derive ``P_A``, ``P_E``, and ``P_V`` from ``G_raw`` counts.
    This is statistical normalization, not JS, EMD, or Entropy calculation.
    """

    if any(value < 0 for value in counts.values()):
        raise ValueError("Counts must be non-negative.")
    total = sum(counts.values())
    if total == 0:
        return {key: 0.0 for key in counts}
    return {key: value / total for key, value in counts.items()}


def validate_positive_optional(value: int | None, name: str) -> None:
    """Validate an optional positive-integer statistical-space parameter.

    Input
    -----
    value:
        ``None`` for the full space or a positive integer constraint.
    name:
        Parameter name used in validation messages.

    Output
    ------
    None
        Returns normally when the value is valid; otherwise raises ValueError.

    Paper experiment step
    ---------------------
    Overall Steps 3 and 16: configure Top-k variants or low-support edge tau.
    """

    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
    ):
        raise ValueError(f"{name} must be None or a positive integer.")

"""Case-level trace contribution clipping for Phase 3."""

from __future__ import annotations

import math
from collections.abc import Sequence

from bul_project.preprocessing.models import Trace

from ._validation import validate_l_max, validate_t_max


def clip_trace(trace: Trace, l_max: int, t_max: float) -> Trace:
    """Clip one case's length and adjacent transition-time contributions.

    Input
    -----
    trace:
        One Phase 1 case-level ``Trace`` in stable event order.
    l_max:
        Maximum number of events contributed by this case.
    t_max:
        Maximum non-negative seconds contributed by one retained transition.

    Output
    ------
    Trace
        A new trace containing at most the first ``l_max`` events. Each
        available retained interval is ``min(interval, t_max)``; ``None``
        remains missing, zero remains zero, and the input object is unchanged.

    Paper experiment step
    ---------------------
    Overall Step 4: bound each case by ``L_max`` and every adjacent interval
    by ``T_max`` before constructing ``G_clip`` and before adding DP noise.
    """

    if not isinstance(trace, Trace):
        raise TypeError("trace must be a Phase 1 Trace object.")
    length_bound = validate_l_max(l_max)
    time_bound = validate_t_max(t_max)
    if not trace.activities:
        raise ValueError("trace must contain at least one activity.")
    if len(trace.timestamps) != trace.length:
        raise ValueError("trace timestamp length does not match activities.")
    if len(trace.transition_seconds) != max(trace.length - 1, 0):
        raise ValueError("trace transition length does not match activities.")

    retained_length = min(trace.length, length_bound)
    retained_intervals: list[float | None] = []
    for interval in trace.transition_seconds[: max(retained_length - 1, 0)]:
        if interval is None:
            retained_intervals.append(None)
            continue
        if isinstance(interval, bool):
            raise ValueError("transition intervals must be numeric seconds, not bool.")
        try:
            value = float(interval)
        except (TypeError, ValueError) as exc:
            raise ValueError("transition intervals must be numeric seconds or None.") from exc
        if not math.isfinite(value):
            raise ValueError("transition intervals must be finite or None.")
        if value < 0:
            raise ValueError(
                "transition intervals must be non-negative; clipping never applies abs()."
            )
        retained_intervals.append(min(value, time_bound))

    return Trace(
        case_id=trace.case_id,
        activities=trace.activities[:retained_length],
        timestamps=trace.timestamps[:retained_length],
        transition_seconds=tuple(retained_intervals),
    )


def clip_traces(
    traces: Sequence[Trace], l_max: int, t_max: float
) -> tuple[Trace, ...]:
    """Clip a complete case collection without merging contributions.

    Input
    -----
    traces:
        Non-empty Phase 1 trace collection with one unique trace per case.
    l_max, t_max:
        Dataset-level contribution bounds fixed before DP release.

    Output
    ------
    tuple[Trace, ...]
        Independently clipped traces suitable for Phase 2 ``build_g_raw`` to
        construct ``G_clip``.

    Paper experiment step
    ---------------------
    Overall Step 4: apply case-level contribution clipping to the whole log.
    """

    materialized = tuple(traces)
    if not materialized:
        raise ValueError("at least one trace is required for clipping.")
    clipped = tuple(clip_trace(trace, l_max, t_max) for trace in materialized)
    case_ids = [trace.case_id for trace in clipped]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("trace input contains duplicate case_id values.")
    return clipped

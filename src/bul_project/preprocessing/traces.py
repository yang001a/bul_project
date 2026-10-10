"""Construction of case-level traces from the canonical event table."""

from __future__ import annotations

import pandas as pd

from bul_project.config import CORE_COLUMNS
from .cleaning import SchemaError
from .models import Trace


def _as_optional_timestamp(value: object) -> pd.Timestamp | None:
    """Return as optional timestamp.

    Inputs:
        value: Candidate value to validate or transform.
    Outputs:
        A value of type `pd.Timestamp | None` produced by the operation.
    Paper role:
        Event-log ingestion, cleaning, and trace construction.
    """
    if pd.isna(value):
        return None
    return pd.Timestamp(value)


def _transition_seconds(
    timestamps: tuple[pd.Timestamp | None, ...],
) -> tuple[float | None, ...]:
    """Calculate the transition interval between two timestamps in seconds.

    Inputs:
        timestamps: Input value used by this operation.
    Outputs:
        A value of type `tuple[float | None, ...]` produced by the operation.
    Paper role:
        Event-log ingestion, cleaning, and trace construction.
    """
    intervals: list[float | None] = []
    for current, following in zip(timestamps, timestamps[1:]):
        if current is None or following is None:
            intervals.append(None)
        else:
            # Deliberately preserve a negative value if unsorted caller input is
            # supplied; silently taking abs() would hide a data-quality problem.
            intervals.append(float((following - current).total_seconds()))
    return tuple(intervals)


def build_traces(
    events: pd.DataFrame,
    *,
    sort_events: bool = True,
    transition_time_unit: str = "seconds",
) -> list[Trace]:
    """Build one immutable :class:`Trace` per case.

    Parameters
    ----------
    events:
        Canonical event table with ``case_id``, ``activity``, and ``timestamp``.
        Phase 1 normally passes the already stably sorted cleaning output.
    sort_events:
        When true, apply the same stable timestamp ordering before grouping.
        The safe public default applies stable sorting. Set false only when the
        caller has already used ``normalize_and_clean`` or is deliberately
        auditing unsorted input.
    transition_time_unit:
        Phase 1 is fixed to seconds. Other values are rejected so the manifest
        cannot advertise a time unit that the trace values do not use.
    """

    missing = [column for column in CORE_COLUMNS if column not in events]
    if missing:
        raise SchemaError(f"Canonical event table is missing columns: {missing}")

    work = events.loc[:, list(CORE_COLUMNS)].copy()
    if work["case_id"].isna().any() or work["activity"].isna().any():
        raise SchemaError("Canonical traces cannot contain missing case_id/activity.")
    if transition_time_unit != "seconds":
        raise ValueError("Phase 1 supports transition_time_unit='seconds' only.")

    if sort_events:
        work["_source_order"] = range(len(work))
        work = work.sort_values(
            ["case_id", "timestamp", "_source_order"],
            kind="mergesort",
            na_position="last",
            ignore_index=True,
        ).drop(columns="_source_order")

    traces: list[Trace] = []
    for case_id, group in work.groupby("case_id", sort=False, observed=True):
        activities = tuple(group["activity"].astype("string").tolist())
        timestamps = tuple(_as_optional_timestamp(value) for value in group["timestamp"])
        traces.append(
            Trace(
                case_id=str(case_id),
                activities=activities,
                timestamps=timestamps,
                transition_seconds=_transition_seconds(timestamps),
            )
        )
    return traces

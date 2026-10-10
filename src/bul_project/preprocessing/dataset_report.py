"""Dataset-level descriptive statistics required by the experiment plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .models import CleaningReport, Trace


def _percentile(values: Iterable[float], quantile: float) -> float | None:
    """Return percentile.

    Inputs:
        values: Values processed by the operation.
        quantile: Input value used by this operation.
    Outputs:
        A value of type `float | None` produced by the operation.
    Paper role:
        Event-log ingestion, cleaning, and trace construction.
    """
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return None
    return float(np.quantile(array, quantile, method="linear"))


def _mean(values: list[float]) -> float | None:
    """Return mean.

    Inputs:
        values: Values processed by the operation.
    Outputs:
        A value of type `float | None` produced by the operation.
    Paper role:
        Event-log ingestion, cleaning, and trace construction.
    """
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    """Return median.

    Inputs:
        values: Values processed by the operation.
    Outputs:
        A value of type `float | None` produced by the operation.
    Paper role:
        Event-log ingestion, cleaning, and trace construction.
    """
    return float(np.median(values)) if values else None


@dataclass(frozen=True)
class DatasetReport:
    """Reproducible descriptive summary of one canonical event log."""

    dataset: str
    cases: int
    events: int
    activities: int
    variants: int
    start_time: str | None
    end_time: str | None
    trace_length: dict[str, float | int | None]
    transition_seconds: dict[str, float | int | None]
    unavailable_timestamp_events: int
    negative_transition_intervals: int
    zero_transition_intervals: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Inputs:
            None.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Event-log ingestion, cleaning, and trace construction.
        """
        return asdict(self)


def build_dataset_report(
    dataset: str,
    events: pd.DataFrame,
    traces: list[Trace],
    cleaning_report: CleaningReport,
) -> DatasetReport:
    """Compute the Phase 1 dataset table without altering the event log."""

    trace_lengths = [float(trace.length) for trace in traces]
    variants = {trace.activities for trace in traces}
    all_intervals = [
        interval
        for trace in traces
        for interval in trace.transition_seconds
        if interval is not None
    ]
    non_negative_intervals = [value for value in all_intervals if value >= 0]
    negative_count = sum(value < 0 for value in all_intervals)
    zero_count = sum(value == 0 for value in all_intervals)

    valid_timestamps = events["timestamp"].dropna()
    start_time = (
        pd.Timestamp(valid_timestamps.min()).isoformat()
        if not valid_timestamps.empty
        else None
    )
    end_time = (
        pd.Timestamp(valid_timestamps.max()).isoformat()
        if not valid_timestamps.empty
        else None
    )

    lengths_as_int = [int(value) for value in trace_lengths]
    trace_summary: dict[str, float | int | None] = {
        "mean": _mean(trace_lengths),
        "median": _median(trace_lengths),
        "max": max(lengths_as_int) if lengths_as_int else None,
        "p75": _percentile(trace_lengths, 0.75),
        "p90": _percentile(trace_lengths, 0.90),
        "p95": _percentile(trace_lengths, 0.95),
    }
    transition_summary: dict[str, float | int | None] = {
        "count": len(non_negative_intervals),
        "median": _median(non_negative_intervals),
        "p90": _percentile(non_negative_intervals, 0.90),
        "p95": _percentile(non_negative_intervals, 0.95),
        "p99": _percentile(non_negative_intervals, 0.99),
    }

    return DatasetReport(
        dataset=dataset,
        cases=len(traces),
        events=len(events),
        activities=int(events["activity"].nunique(dropna=True)),
        variants=len(variants),
        start_time=start_time,
        end_time=end_time,
        trace_length=trace_summary,
        transition_seconds=transition_summary,
        unavailable_timestamp_events=cleaning_report.unavailable_timestamp_events,
        negative_transition_intervals=int(negative_count),
        zero_transition_intervals=int(zero_count),
    )

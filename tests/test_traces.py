from __future__ import annotations

import pandas as pd
import pytest

from bul_project.preprocessing.cleaning import SchemaError
from bul_project.preprocessing.traces import build_traces


def test_trace_intervals_keep_zero_and_missing_values() -> None:
    events = pd.DataFrame(
        {
            "case_id": ["c", "c", "c"],
            "activity": ["A", "B", "C"],
            "timestamp": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", None],
                utc=True,
            ),
        }
    )

    trace = build_traces(events)[0]

    assert trace.activities == ("A", "B", "C")
    assert trace.transition_seconds == (0.0, None)


def test_unsorted_negative_interval_is_not_converted_to_absolute_value() -> None:
    events = pd.DataFrame(
        {
            "case_id": ["c", "c"],
            "activity": ["late", "early"],
            "timestamp": pd.to_datetime(
                ["2024-01-01T11:00:00Z", "2024-01-01T10:00:00Z"], utc=True
            ),
        }
    )

    trace = build_traces(events, sort_events=False)[0]

    assert trace.transition_seconds == (-3600.0,)


def test_trace_builder_requires_canonical_schema() -> None:
    with pytest.raises(SchemaError, match="timestamp"):
        build_traces(pd.DataFrame({"case_id": ["c"], "activity": ["A"]}))

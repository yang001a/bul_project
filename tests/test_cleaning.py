from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from bul_project.config import DEFAULT_PREPROCESSING, DatasetSpec
from bul_project.preprocessing.cleaning import SchemaError, normalize_and_clean
from bul_project.preprocessing.xes_reader import read_xes_dataframe


def test_demo_cleaning_is_auditable_and_stable(demo_spec: DatasetSpec) -> None:
    raw = read_xes_dataframe(demo_spec.path)
    events, report = normalize_and_clean(raw, demo_spec, DEFAULT_PREPROCESSING)

    assert list(events.columns) == ["case_id", "activity", "timestamp"]
    assert len(events) == 7
    assert report.input_events == 8
    assert report.output_events == 7
    assert report.missing_activity_events == 1
    assert report.dropped_core_events == 1
    assert report.unavailable_timestamp_events == 1
    assert report.negative_intervals_before_sort == 1
    assert report.timestamp_tie_events == 2
    assert str(events["timestamp"].dtype) == "datetime64[ns, UTC]"

    case_one = events.loc[events["case_id"] == "case-1", "activity"].tolist()
    case_three = events.loc[events["case_id"] == "case-3", "activity"].tolist()
    assert case_one == ["Start", "Review", "End"]
    assert case_three == ["A", "B"]  # source order survives the timestamp tie
    register_time = events.loc[
        (events["case_id"] == "case-2") & (events["activity"] == "Register"),
        "timestamp",
    ].iloc[0]
    assert register_time.isoformat() == "2024-01-02T08:00:00+00:00"


def test_schema_error_lists_missing_source_columns() -> None:
    spec = DatasetSpec(name="BROKEN", path=pd.NA)  # path is unused here
    raw = pd.DataFrame({"concept:name": ["A"]})

    with pytest.raises(SchemaError, match="case:concept:name"):
        normalize_and_clean(raw, spec, DEFAULT_PREPROCESSING)


def test_case_ids_remain_strings_and_timezones_normalize_to_utc() -> None:
    spec = DatasetSpec(name="TZ", path=pd.NA)
    raw = pd.DataFrame(
        {
            "case:concept:name": [2012, 2012],
            "concept:name": ["A", "B"],
            "time:timestamp": [
                "2024-03-31T01:30:00+01:00",
                "2024-03-31T03:30:00+02:00",
            ],
        }
    )

    events, _ = normalize_and_clean(raw, spec, DEFAULT_PREPROCESSING)

    assert events["case_id"].tolist() == ["2012", "2012"]
    assert str(events["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert events["timestamp"].iloc[0].isoformat() == "2024-03-31T00:30:00+00:00"
    assert events["timestamp"].iloc[1].isoformat() == "2024-03-31T01:30:00+00:00"


def test_duplicates_are_reported_but_preserved_unless_explicitly_dropped() -> None:
    spec = DatasetSpec(name="DUP", path=pd.NA)
    raw = pd.DataFrame(
        {
            "case:concept:name": ["c", "c"],
            "concept:name": ["A", "A"],
            "time:timestamp": ["2024-01-01Z", "2024-01-01Z"],
            "org:resource": ["r", "r"],
        }
    )

    preserved, report = normalize_and_clean(raw, spec, DEFAULT_PREPROCESSING)
    assert len(preserved) == 2
    assert report.exact_duplicate_events == 1
    assert report.dropped_exact_duplicates == 0

    policy = replace(
        DEFAULT_PREPROCESSING,
        drop_exact_duplicates=True,
        keep_extensions=True,
    )
    deduplicated, dropped_report = normalize_and_clean(raw, spec, policy)
    assert len(deduplicated) == 1
    assert deduplicated["org:resource"].tolist() == ["r"]
    assert dropped_report.dropped_exact_duplicates == 1

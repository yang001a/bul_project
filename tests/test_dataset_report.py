from __future__ import annotations

import pytest

from bul_project.config import DEFAULT_PREPROCESSING, DatasetSpec
from bul_project.preprocessing.cleaning import normalize_and_clean
from bul_project.preprocessing.dataset_report import build_dataset_report
from bul_project.preprocessing.traces import build_traces
from bul_project.preprocessing.xes_reader import read_xes_dataframe


def test_demo_dataset_report_matches_hand_calculation(
    demo_spec: DatasetSpec,
) -> None:
    events, cleaning = normalize_and_clean(
        read_xes_dataframe(demo_spec.path), demo_spec, DEFAULT_PREPROCESSING
    )
    traces = build_traces(events)
    report = build_dataset_report("DEMO", events, traces, cleaning)

    assert report.cases == 3
    assert report.events == 7
    assert report.activities == 7
    assert report.variants == 3
    assert report.trace_length["mean"] == pytest.approx(7 / 3)
    assert report.trace_length["median"] == 2
    assert report.trace_length["max"] == 3
    assert report.trace_length["p95"] == pytest.approx(2.9)
    assert report.transition_seconds["count"] == 3
    assert report.transition_seconds["median"] == 3600
    assert report.zero_transition_intervals == 1
    assert report.negative_transition_intervals == 0
    assert report.unavailable_timestamp_events == 1

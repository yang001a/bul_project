from __future__ import annotations

import os

import pytest

from bul_project.config import DATASETS
from bul_project.preprocessing.pipeline import preprocess_dataset


EXPECTED = {
    "BPIC2017": {
        "events": 1_202_267,
        "cases": 31_509,
        "activities": 26,
        "start": "2016-01-01T09:51:15.304000+00:00",
        "end": "2017-02-01T14:11:03.499000+00:00",
        "tie_events": 0,
    },
    "BPIC2012": {
        "events": 262_200,
        "cases": 13_087,
        "activities": 24,
        "start": "2011-09-30T22:38:44.546000+00:00",
        "end": "2012-03-14T15:04:54.681000+00:00",
        "tie_events": 25_369,
    },
    "SEPSIS": {
        "events": 15_214,
        "cases": 1_050,
        "activities": 16,
        "start": "2013-11-07T07:18:29+00:00",
        "end": "2015-06-05T10:25:11+00:00",
        "tie_events": 7_526,
    },
}


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_REAL_XES_TESTS") != "1",
    reason="Set RUN_REAL_XES_TESTS=1 to read all real XES datasets.",
)
@pytest.mark.parametrize("dataset_name", tuple(EXPECTED))
def test_real_dataset_counts_and_core_invariants(dataset_name: str) -> None:
    result = preprocess_dataset(DATASETS[dataset_name])
    expected = EXPECTED[dataset_name]

    assert result.dataset_report.events == expected["events"]
    assert result.dataset_report.cases == expected["cases"]
    assert result.dataset_report.activities == expected["activities"]
    assert result.dataset_report.start_time == expected["start"]
    assert result.dataset_report.end_time == expected["end"]
    assert result.cleaning_report.timestamp_tie_events == expected["tie_events"]
    assert result.cleaning_report.dropped_core_events == 0
    assert result.cleaning_report.unavailable_timestamp_events == 0
    assert result.cleaning_report.negative_intervals_before_sort == 0
    assert result.dataset_report.negative_transition_intervals == 0
    assert len(result.traces) == expected["cases"]
    assert sum(trace.length for trace in result.traces) == expected["events"]
    assert list(result.events.columns) == ["case_id", "activity", "timestamp"]

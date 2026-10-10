from __future__ import annotations

import pandas as pd
import pytest

from bul_project.validation.evaluation_sample import (
    deterministic_case_sample,
    expected_evaluation_sample_metadata,
    expected_evaluation_sampling_policy,
)


def _events() -> pd.DataFrame:
    rows = []
    for case_index in range(12):
        for event_index, activity in enumerate(("A", "B", "C")):
            rows.append(
                {
                    "case_id": f"case-{case_index:02d}",
                    "activity": activity,
                    "timestamp": pd.Timestamp("2026-01-01")
                    + pd.Timedelta(seconds=event_index),
                }
            )
    return pd.DataFrame(rows)


def test_hash_rank_sample_is_reproducible_and_keeps_whole_cases() -> None:
    events = _events()
    first = deterministic_case_sample(events, sample_size_cases=5, seed=27)
    second = deterministic_case_sample(events, sample_size_cases=5, seed=27)

    pd.testing.assert_frame_equal(first, second)
    assert first["case_id"].nunique() == 5
    assert len(first) == 15
    for case_id in first["case_id"].unique():
        assert first.loc[first["case_id"] == case_id, "activity"].tolist() == [
            "A",
            "B",
            "C",
        ]


def test_hash_rank_sample_validates_size_and_columns() -> None:
    events = _events()
    with pytest.raises(ValueError, match="cannot exceed"):
        deterministic_case_sample(events, sample_size_cases=13, seed=27)
    with pytest.raises(ValueError, match="canonical columns"):
        deterministic_case_sample(
            events.drop(columns="timestamp"), sample_size_cases=3, seed=27
        )


def test_frozen_bpic2017_sample_contract_is_resource_bounded() -> None:
    metadata = expected_evaluation_sample_metadata()
    policy = expected_evaluation_sampling_policy()

    assert metadata["case_count"] == 3000
    assert metadata["event_count"] == 114955
    assert policy["population_case_count"] == 31509
    assert policy["same_sample_for_all_conditions"] is True
    assert policy["role"] == "external_scoring_only_not_DP_generation"


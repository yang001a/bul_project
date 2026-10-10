from __future__ import annotations

import inspect
import json

import pandas as pd
import pytest

from bul_project.validation.synthetic_log import (
    MissingTemporalEdgeError,
    SYNTHETIC_START_TIMESTAMP,
    SyntheticLogGenerator,
)
from bul_project.privacy.temporal_generator import TemporalDistributionGenerator


EDGE_AB = ("A", "B")
EDGE_AC = ("A", "C")


def _temporal_result(seed: int = 0):
    return TemporalDistributionGenerator(t_max=10.0, seed=seed).generate(
        edge_counts={EDGE_AB: 4, EDGE_AC: 3},
        time_sums={EDGE_AB: 8.0, EDGE_AC: 9.0},
        time_squared_sums={EDGE_AB: 20.0, EDGE_AC: 33.0},
    )


def test_synthetic_log_uses_concrete_dp_variants_and_excludes_other() -> None:
    result = SyntheticLogGenerator(seed=0).generate(
        {("A", "B"): 2, ("A", "C"): 1},
        _temporal_result(seed=0),
        other_count=9,
    )

    assert result.case_count == 3
    assert result.events.columns.tolist() == ["case_id", "activity", "timestamp"]
    assert result.events["case_id"].nunique() == 3
    assert len(result.events) == 6
    assert result.metadata["excluded_other_count"] == 9
    assert result.metadata["case_count"] == 3
    probabilities = {
        tuple(row["activities"]): row["synthetic_probability"]
        for row in result.metadata["concrete_variant_distribution"]
    }
    assert probabilities == pytest.approx({("A", "B"): 2 / 3, ("A", "C"): 1 / 3})
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_synthetic_timestamps_start_fixed_and_accumulate_forward() -> None:
    temporal = _temporal_result(seed=1)
    result = SyntheticLogGenerator(seed=1).generate(
        {("A", "B"): 2, ("A", "C"): 2}, temporal
    )

    for _, case in result.events.groupby("case_id", sort=False):
        timestamps = case["timestamp"].tolist()
        assert timestamps[0] == SYNTHETIC_START_TIMESTAMP
        differences = case["timestamp"].diff().dropna().dt.total_seconds()
        assert (differences >= 0).all()
        edge = tuple(case["activity"].tolist())
        pool = temporal.samples[(edge[0], edge[1])]
        assert any(
            differences.iloc[0] == pytest.approx(value, abs=1e-9)
            for value in pool
        )


def test_synthetic_generation_is_exactly_reproducible_for_same_seed() -> None:
    first = SyntheticLogGenerator(seed=5).generate(
        {("A", "B"): 5, ("A", "C"): 5}, _temporal_result(seed=5)
    )
    second = SyntheticLogGenerator(seed=5).generate(
        {("A", "B"): 5, ("A", "C"): 5}, _temporal_result(seed=5)
    )

    pd.testing.assert_frame_equal(first.events, second.events)
    assert first.metadata == second.metadata
    assert first.seed == second.seed == 5


def test_missing_variant_edge_fails_and_records_missing_edge_metadata() -> None:
    temporal = TemporalDistributionGenerator(t_max=10.0, seed=2).generate(
        edge_counts={EDGE_AB: 1},
        time_sums={EDGE_AB: 1.0},
        time_squared_sums={EDGE_AB: 1.0},
    )

    with pytest.raises(MissingTemporalEdgeError) as captured:
        SyntheticLogGenerator(seed=2).generate(
            {("A", "C"): 1}, temporal
        )

    error = captured.value
    assert error.missing_edges == (EDGE_AC,)
    assert error.metadata["status"] == "failed_missing_temporal_edges"
    assert error.metadata["missing_edges"] == [{"source": "A", "target": "C"}]
    assert error.metadata["raw_data_access"] is False
    json.dumps(error.metadata)


def test_single_activity_variant_needs_no_temporal_edge() -> None:
    result = SyntheticLogGenerator(seed=3).generate(
        {("A",): 2}, _temporal_result(seed=3), other_count=1
    )

    assert result.case_count == 2
    assert len(result.events) == 2
    assert set(result.events["activity"]) == {"A"}
    assert set(result.events["timestamp"]) == {SYNTHETIC_START_TIMESTAMP}


def test_synthetic_metadata_is_json_serializable_and_dp_only() -> None:
    result = SyntheticLogGenerator(seed=4).generate(
        {("A", "B"): 1}, _temporal_result(seed=4), other_count=2
    )
    metadata = json.loads(result.metadata_json())

    assert metadata["status"] == "success"
    assert metadata["dp_post_processing_only"] is True
    assert metadata["raw_data_access"] is False
    assert metadata["timestamp_rule"] == "timestamp[i+1] = timestamp[i] + T'_e"
    assert metadata["other_policy"] == "excluded_from_synthetic_generation"


@pytest.mark.parametrize(
    "call",
    [
        lambda: SyntheticLogGenerator(seed=30),
        lambda: SyntheticLogGenerator(seed=0).generate(
            {}, _temporal_result(seed=0)
        ),
        lambda: SyntheticLogGenerator(seed=0).generate(
            {"OTHER": 1},  # type: ignore[dict-item]
            _temporal_result(seed=0),
        ),
        lambda: SyntheticLogGenerator(seed=0).generate(
            {("A",): 0}, _temporal_result(seed=0), other_count=10
        ),
        lambda: SyntheticLogGenerator(seed=0).generate(
            {("A",): 1.5}, _temporal_result(seed=0)
        ),
        lambda: SyntheticLogGenerator(seed=0).generate(
            {("A",): 1}, _temporal_result(seed=0), other_count=-1
        ),
        lambda: SyntheticLogGenerator(seed=0).generate(
            {("A",): 1}, _temporal_result(seed=1)
        ),
        lambda: SyntheticLogGenerator(seed=0).generate(
            {("A",): 1},  # type: ignore[arg-type]
            {EDGE_AB: [1.0]},  # raw sample mappings are forbidden
        ),
    ],
)
def test_invalid_synthetic_inputs_fail_explicitly(call: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()  # type: ignore[operator]


def test_synthetic_generate_documents_input_output_and_method_step() -> None:
    documentation = inspect.getdoc(SyntheticLogGenerator.generate) or ""
    assert "Input" in documentation
    assert "Output" in documentation
    assert "Paper experiment step" in documentation


def test_synthetic_interface_has_no_raw_log_trace_or_time_parameter() -> None:
    parameters = inspect.signature(SyntheticLogGenerator.generate).parameters
    assert tuple(parameters) == (
        "self",
        "variant_counts",
        "temporal_distributions",
        "other_count",
    )
    assert all("raw" not in name and "trace" not in name for name in parameters)

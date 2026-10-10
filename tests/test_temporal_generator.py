from __future__ import annotations

import inspect
import json

import pytest

from bul_project.privacy.temporal_generator import TemporalDistributionGenerator


EDGE_AB = ("A", "B")
EDGE_BC = ("B", "C")


def test_temporal_generator_reconstructs_dp_mean_variance_and_count() -> None:
    result = TemporalDistributionGenerator(t_max=10.0, seed=0).generate(
        edge_counts={EDGE_AB: 4},
        time_sums={EDGE_AB: 8.0},
        time_squared_sums={EDGE_AB: 20.0},
    )

    assert result.means[EDGE_AB] == pytest.approx(2.0)
    assert result.variances[EDGE_AB] == pytest.approx(1.0)
    assert result.standard_deviations[EDGE_AB] == pytest.approx(1.0)
    assert result.sample_counts[EDGE_AB] == 4
    assert len(result.samples[EDGE_AB]) == 4
    assert all(0.0 <= value <= 10.0 for value in result.samples[EDGE_AB])


def test_temporal_generator_floors_negative_noisy_variance_without_raw_data() -> None:
    delta = 1e-8
    result = TemporalDistributionGenerator(
        t_max=10.0, seed=1, delta=delta
    ).generate(
        edge_counts={EDGE_AB: 2},
        time_sums={EDGE_AB: 4.0},
        time_squared_sums={EDGE_AB: -100.0},
    )

    assert result.means[EDGE_AB] == pytest.approx(2.0)
    assert result.variances[EDGE_AB] == pytest.approx(delta)
    assert result.metadata["raw_data_access"] is False
    edge_metadata = result.metadata["edges"][0]
    assert edge_metadata["raw_variance_seconds_squared"] < 0
    assert edge_metadata["reconstructed_variance_seconds_squared"] == pytest.approx(
        delta
    )


def test_zero_dp_edge_count_still_generates_one_bounded_sample() -> None:
    result = TemporalDistributionGenerator(t_max=5.0, seed=2).generate(
        edge_counts={EDGE_AB: 0},
        time_sums={EDGE_AB: 0.0},
        time_squared_sums={EDGE_AB: 0.0},
    )

    assert result.sample_counts[EDGE_AB] == 1
    assert result.means[EDGE_AB] == 0.0
    assert result.variances[EDGE_AB] == pytest.approx(1e-8)
    assert 0.0 <= result.samples[EDGE_AB][0] <= 5.0


def test_temporal_generator_uses_round_rule_and_zero_tmax_bound() -> None:
    result = TemporalDistributionGenerator(t_max=0.0, seed=3).generate(
        edge_counts={EDGE_AB: 1.5},
        time_sums={EDGE_AB: 0.0},
        time_squared_sums={EDGE_AB: 0.0},
    )

    assert result.sample_counts[EDGE_AB] == 2
    assert result.samples[EDGE_AB] == (0.0, 0.0)


def test_temporal_generator_is_exactly_reproducible_for_same_seed() -> None:
    inputs = {
        "edge_counts": {EDGE_AB: 50, EDGE_BC: 20},
        "time_sums": {EDGE_AB: 250.0, EDGE_BC: 40.0},
        "time_squared_sums": {EDGE_AB: 1450.0, EDGE_BC: 100.0},
    }
    first = TemporalDistributionGenerator(t_max=10.0, seed=7).generate(**inputs)
    second = TemporalDistributionGenerator(t_max=10.0, seed=7).generate(**inputs)
    different = TemporalDistributionGenerator(t_max=10.0, seed=8).generate(**inputs)

    assert first == second
    assert first.samples != different.samples
    assert len(set(first.samples[EDGE_AB])) > 1


def test_temporal_metadata_is_json_serializable_and_records_only_dp_inputs() -> None:
    result = TemporalDistributionGenerator(t_max=10.0, seed=4).generate(
        edge_counts={EDGE_AB: 2},
        time_sums={EDGE_AB: 4.0},
        time_squared_sums={EDGE_AB: 10.0},
    )
    payload = json.loads(json.dumps(result.to_dict()))
    metadata = json.loads(result.metadata_json())

    assert payload["name"] == "DP_temporal_distributions"
    assert metadata["seed"] == 4
    assert metadata["dp_post_processing_only"] is True
    assert metadata["raw_data_access"] is False
    assert metadata["bounds_seconds"] == [0.0, 10.0]


def test_temporal_generator_requires_identical_frozen_edge_domains() -> None:
    generator = TemporalDistributionGenerator(t_max=10.0, seed=0)
    with pytest.raises(ValueError, match="same frozen DP edge domain"):
        generator.generate(
            edge_counts={EDGE_AB: 1},
            time_sums={EDGE_AB: 1.0, EDGE_BC: 2.0},
            time_squared_sums={EDGE_AB: 1.0},
        )


@pytest.mark.parametrize(
    "call",
    [
        lambda: TemporalDistributionGenerator(t_max=-1.0, seed=0),
        lambda: TemporalDistributionGenerator(t_max=1.0, seed=30),
        lambda: TemporalDistributionGenerator(t_max=1.0, seed=0, delta=0),
        lambda: TemporalDistributionGenerator(t_max=1.0, seed=0).generate(
            {}, {}, {}
        ),
        lambda: TemporalDistributionGenerator(t_max=1.0, seed=0).generate(
            {EDGE_AB: -1}, {EDGE_AB: 0}, {EDGE_AB: 0}
        ),
        lambda: TemporalDistributionGenerator(t_max=1.0, seed=0).generate(
            {EDGE_AB: 1}, {EDGE_AB: -1}, {EDGE_AB: 0}
        ),
        lambda: TemporalDistributionGenerator(t_max=1.0, seed=0).generate(
            {EDGE_AB: 1}, {EDGE_AB: 0}, {EDGE_AB: float("nan")}
        ),
        lambda: TemporalDistributionGenerator(t_max=1.0, seed=0).generate(
            {("A", ""): 1}, {("A", ""): 0}, {("A", ""): 0}
        ),
    ],
)
def test_invalid_temporal_generator_inputs_fail_explicitly(call: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()  # type: ignore[operator]


def test_temporal_generate_documents_input_output_and_method_step() -> None:
    documentation = inspect.getdoc(TemporalDistributionGenerator.generate) or ""
    assert "Input" in documentation
    assert "Output" in documentation
    assert "Paper experiment step" in documentation


def test_temporal_interface_has_no_raw_trace_or_raw_sample_parameter() -> None:
    parameters = inspect.signature(TemporalDistributionGenerator.generate).parameters
    assert tuple(parameters) == (
        "self",
        "edge_counts",
        "time_sums",
        "time_squared_sums",
    )
    assert all("raw" not in name and "trace" not in name for name in parameters)

from __future__ import annotations

import copy
import inspect

import pandas as pd
import pytest

from bul_project.validation.release import (
    ProcessQualityReleaseError,
    generate_process_quality_release,
)
from bul_project.privacy.temporal_generator import TemporalDistributionGenerator
from bul_project.validation.synthetic_log import MissingTemporalEdgeError


EDGE_AB = ("A", "B")
EDGE_BC = ("B", "C")


def _stored_temporal_generation(
    *,
    edge_counts: dict[tuple[str, str], int],
    time_sums: dict[tuple[str, str], float],
    time_squared_sums: dict[tuple[str, str], float],
    seed: int,
    t_max: float,
) -> dict[str, object]:
    result = TemporalDistributionGenerator(t_max=t_max, seed=seed).generate(
        edge_counts=edge_counts,
        time_sums=time_sums,
        time_squared_sums=time_squared_sums,
    )
    return {
        "seed": result.seed,
        "t_max": result.t_max,
        "delta": result.delta,
        "sample_counts": [
            {
                "source": source,
                "target": target,
                "sample_count": result.sample_counts[(source, target)],
            }
            for source, target in sorted(result.sample_counts)
        ],
        "metadata": result.metadata,
    }


def _payload(*, seed: int = 7, epsilon: float = 2.0) -> dict[str, object]:
    t_max = 10.0
    edge_counts = {EDGE_AB: 4, EDGE_BC: 3}
    time_sums = {EDGE_AB: 8.0, EDGE_BC: 9.0}
    time_squared_sums = {EDGE_AB: 20.0, EDGE_BC: 33.0}
    return {
        "name": "small_authenticated_DP_checkpoint",
        "config": {
            "dataset": "DEMO",
            "method": "DP-full",
            "budget_strategy": "uniform",
            "dp_edge_mode": None,
            "epsilon": epsilon,
            "seed": seed,
            "t_max": t_max,
        },
        "release": {
            "name": "G_DP",
            "method": "DP-full",
            "budget_strategy": "uniform",
            "dp_edge_mode": None,
            "epsilon_total": epsilon,
            "epsilon_spent": epsilon,
            "seed": seed,
            "dfg": [
                {
                    "source": "A",
                    "target": "B",
                    "count": 4,
                    "probability": 4 / 7,
                },
                {
                    "source": "B",
                    "target": "C",
                    "count": 3,
                    "probability": 3 / 7,
                },
            ],
            "temporal": [
                {
                    "source": "A",
                    "target": "B",
                    "mean_seconds": 2.0,
                    "sum_seconds": 8.0,
                    "squared_sum_seconds2": 20.0,
                },
                {
                    "source": "B",
                    "target": "C",
                    "mean_seconds": 3.0,
                    "sum_seconds": 9.0,
                    "squared_sum_seconds2": 33.0,
                },
            ],
            "variants": [
                {
                    "activities": ["A", "B", "C"],
                    "count": 2,
                    "probability": 2 / 12,
                },
                {
                    "activities": ["A", "B"],
                    "count": 1,
                    "probability": 1 / 12,
                },
            ],
            "variant_other": {
                "label": "OTHER",
                "count": 9,
                "probability": 9 / 12,
            },
        },
        "temporal_generation": _stored_temporal_generation(
            edge_counts=edge_counts,
            time_sums=time_sums,
            time_squared_sums=time_squared_sums,
            seed=seed,
            t_max=t_max,
        ),
    }


def _generate(payload: dict[str, object]):
    return generate_process_quality_release(
        payload,
        expected_epsilon=2.0,
        expected_seed=7,
        frozen_t_max=10.0,
    )


def test_bridge_strictly_extracts_dp_release_and_excludes_other() -> None:
    result = _generate(_payload())

    assert result.edge_counts == {EDGE_AB: 4, EDGE_BC: 3}
    assert result.time_sums == {EDGE_AB: 8.0, EDGE_BC: 9.0}
    assert result.time_squared_sums == {EDGE_AB: 20.0, EDGE_BC: 33.0}
    assert result.variant_counts == {("A", "B", "C"): 2, ("A", "B"): 1}
    assert result.other_count == 9
    assert result.synthetic_log.case_count == 3
    assert result.synthetic_log.events["case_id"].nunique() == 3
    assert result.synthetic_log.metadata["excluded_other_count"] == 9
    assert result.metadata["concrete_case_count"] == 3
    assert result.metadata["temporal_checkpoint_science_verified"] is True
    assert result.metadata["raw_data_access"] is False


def test_bridge_is_exactly_reproducible_for_same_seed() -> None:
    first = _generate(_payload())
    second = _generate(_payload())

    assert first.temporal_distributions == second.temporal_distributions
    pd.testing.assert_frame_equal(
        first.synthetic_log.events, second.synthetic_log.events
    )
    assert first.synthetic_log.metadata == second.synthetic_log.metadata
    assert first.metadata == second.metadata


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["config"].update(method="DP-edge-only"), "method"),
        (
            lambda payload: payload["release"].update(budget_strategy="edge_focus"),
            "budget_strategy",
        ),
        (lambda payload: payload["release"].update(epsilon_total=1.0), "epsilon_total"),
        (lambda payload: payload["release"].update(seed=8), "release.seed"),
        (lambda payload: payload["config"].update(t_max=9.0), "config.t_max"),
    ],
)
def test_bridge_rejects_wrong_formal_condition_identity(
    mutation: object, message: str
) -> None:
    payload = _payload()
    mutation(payload)  # type: ignore[operator]

    with pytest.raises(ProcessQualityReleaseError, match=message):
        _generate(payload)


def test_bridge_requires_identical_dfg_and_temporal_edge_domains() -> None:
    payload = _payload()
    payload["release"]["temporal"] = payload["release"]["temporal"][:1]

    with pytest.raises(ProcessQualityReleaseError, match="same frozen DP edge domain"):
        _generate(payload)


def test_bridge_rejects_other_inside_concrete_variants() -> None:
    payload = _payload()
    payload["release"]["variants"][0]["activities"] = ["OTHER"]

    with pytest.raises(ProcessQualityReleaseError, match="OTHER must remain separate"):
        _generate(payload)


def test_bridge_rejects_temporal_checkpoint_science_mismatch() -> None:
    payload = _payload()
    payload["temporal_generation"]["metadata"]["edges"][0][
        "reconstructed_mean_seconds"
    ] = 999.0

    with pytest.raises(
        ProcessQualityReleaseError,
        match="temporal_generation.metadata.edges",
    ):
        _generate(payload)


def test_missing_variant_edge_propagates_established_error_unchanged() -> None:
    payload = _payload()
    payload["release"]["variants"] = [
        {"activities": ["A", "C"], "count": 1, "probability": 1.0}
    ]
    payload["release"]["variant_other"] = {
        "label": "OTHER",
        "count": 0,
        "probability": 0.0,
    }

    with pytest.raises(MissingTemporalEdgeError) as captured:
        _generate(payload)

    assert captured.value.missing_edges == (("A", "C"),)
    assert captured.value.metadata["status"] == "failed_missing_temporal_edges"
    assert captured.value.metadata["missing_edges"] == [
        {"source": "A", "target": "C"}
    ]


def test_bridge_rejects_non_json_release_records_and_duplicate_edges() -> None:
    malformed = _payload()
    malformed["release"]["dfg"][0]["unexpected"] = "raw"
    with pytest.raises(ProcessQualityReleaseError, match="formal checkpoint schema"):
        _generate(malformed)

    duplicate = _payload()
    duplicate["release"]["dfg"].append(
        copy.deepcopy(duplicate["release"]["dfg"][0])
    )
    with pytest.raises(ProcessQualityReleaseError, match="Duplicate DP DFG edge"):
        _generate(duplicate)


def test_bridge_interface_has_no_raw_trace_log_or_evaluation_parameter() -> None:
    parameters = inspect.signature(generate_process_quality_release).parameters
    assert tuple(parameters) == (
        "experiment_metadata",
        "expected_epsilon",
        "expected_seed",
        "frozen_t_max",
    )
    forbidden = ("raw", "trace", "log", "evaluation", "path")
    assert all(
        not any(token in parameter.lower() for token in forbidden)
        for parameter in parameters
    )


def test_bridge_documents_input_output_and_paper_step() -> None:
    documentation = inspect.getdoc(generate_process_quality_release) or ""
    assert "Input" in documentation
    assert "Output" in documentation
    assert "Paper experiment step" in documentation

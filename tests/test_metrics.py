from __future__ import annotations

import inspect
import math

import pytest

from bul_project.config import DELTA_DENOM, DELTA_PROB, NEMD_ZERO_SCALE
from bul_project.metrics import (
    calculate_cosine_loss,
    calculate_cosine_similarity,
    calculate_emd,
    calculate_entropy_loss,
    calculate_js,
    calculate_normalized_emd,
)


def test_js_is_divergence_not_square_root_distance() -> None:
    value = calculate_js([1.0, 0.0], [0.5, 0.5])

    assert value == pytest.approx(0.311278124, abs=2e-8)
    assert value != pytest.approx(math.sqrt(0.311278124), abs=1e-4)


def test_js_is_symmetric_bounded_and_zero_for_identical_inputs() -> None:
    left = {"A": 2, "B": 1}
    right = {"A": 1, "C": 2}

    assert calculate_js(left, left) == pytest.approx(0.0, abs=1e-15)
    assert calculate_js(left, right) == pytest.approx(calculate_js(right, left))
    assert 0.0 <= calculate_js(left, right) <= 1.0
    assert calculate_js([1, 0], [0, 1]) == pytest.approx(1.0, abs=1e-8)


def test_js_aligns_mapping_union_and_normalizes_counts() -> None:
    mapping_loss = calculate_js({"A": 10, "B": 0}, {"B": 5, "A": 0})
    vector_loss = calculate_js([10, 0], [0, 5])

    assert mapping_loss == pytest.approx(vector_loss)
    assert calculate_js([2, 2], [100, 100]) == pytest.approx(0.0, abs=1e-15)


def test_cosine_similarity_is_explicitly_converted_to_loss() -> None:
    identical_similarity = calculate_cosine_similarity([1, 2], [1, 2])
    identical_loss = calculate_cosine_loss([1, 2], [1, 2])
    orthogonal_similarity = calculate_cosine_similarity([1, 0], [0, 1])
    orthogonal_loss = calculate_cosine_loss([1, 0], [0, 1])

    assert identical_similarity == pytest.approx(1.0)
    assert identical_loss == pytest.approx(0.0)
    assert orthogonal_similarity == pytest.approx(0.0)
    assert orthogonal_loss == pytest.approx(1.0)
    assert identical_loss == pytest.approx(1.0 - identical_similarity)
    assert orthogonal_loss == pytest.approx(1.0 - orthogonal_similarity)


def test_cosine_direction_for_partial_distortion_and_mapping_alignment() -> None:
    similarity = calculate_cosine_similarity({"A": 1, "B": 1}, {"A": 1})
    loss = calculate_cosine_loss({"A": 1, "B": 1}, {"A": 1})

    assert similarity == pytest.approx(1.0 / math.sqrt(2.0))
    assert loss == pytest.approx(1.0 - 1.0 / math.sqrt(2.0))
    assert calculate_cosine_loss([1, 2], [10, 20]) == pytest.approx(0.0)


def test_cosine_zero_vector_protection_has_consistent_direction() -> None:
    assert calculate_cosine_similarity([0, 0], [0, 0]) == 1.0
    assert calculate_cosine_loss([0, 0], [0, 0]) == 0.0
    assert calculate_cosine_similarity([0, 0], [1, 0]) == 0.0
    assert calculate_cosine_loss([0, 0], [1, 0]) == 1.0


def test_raw_emd_is_known_wasserstein_distance_and_loss_directed() -> None:
    assert calculate_emd([0, 2], [0, 2]) == pytest.approx(0.0)
    assert calculate_emd([0, 2], [1, 3]) == pytest.approx(1.0)
    assert calculate_emd([0, 2], [2, 4]) == pytest.approx(2.0)
    assert calculate_emd([1, 3], [0, 2]) == pytest.approx(1.0)


def test_normalized_emd_uses_caller_fixed_scale_and_clips_to_unit_interval() -> None:
    assert calculate_normalized_emd([0, 2], [1, 3], 2.0) == pytest.approx(0.5)
    assert calculate_normalized_emd([0, 2], [2, 4], 2.0) == pytest.approx(1.0)
    assert calculate_normalized_emd([0], [10], 2.0) == pytest.approx(1.0)
    assert calculate_normalized_emd([5], [5], 0.0) == pytest.approx(0.0)
    assert calculate_normalized_emd([0], [1], 0.0) == pytest.approx(1.0)


def test_entropy_loss_uses_reference_entropy_denominator_and_loss_direction() -> None:
    uniform = [1, 1]
    concentrated = [1, 0]

    assert calculate_entropy_loss(uniform, uniform) == pytest.approx(0.0)
    assert calculate_entropy_loss(uniform, concentrated) == pytest.approx(1.0)
    assert calculate_entropy_loss(concentrated, uniform) == pytest.approx(1.0)
    assert calculate_entropy_loss([1, 0], [0, 1]) == pytest.approx(0.0)
    assert calculate_entropy_loss([2, 2], [100, 100]) == pytest.approx(0.0)


def test_all_bul_facing_metric_outputs_are_higher_is_worse_and_bounded() -> None:
    identical_losses = (
        calculate_js([0.7, 0.3], [0.7, 0.3]),
        calculate_cosine_loss([0.7, 0.3], [0.7, 0.3]),
        calculate_normalized_emd([0, 1], [0, 1], 10.0),
        calculate_entropy_loss([0.7, 0.3], [0.7, 0.3]),
    )
    distorted_losses = (
        calculate_js([1, 0], [0, 1]),
        calculate_cosine_loss([1, 0], [0, 1]),
        calculate_normalized_emd([0, 1], [5, 6], 10.0),
        calculate_entropy_loss([1, 1], [1, 0]),
    )

    assert identical_losses == pytest.approx((0.0, 0.0, 0.0, 0.0))
    assert all(0.0 <= value <= 1.0 for value in distorted_losses)
    assert all(distorted > identical for distorted, identical in zip(distorted_losses, identical_losses))


@pytest.mark.parametrize(
    "call",
    [
        lambda: calculate_js([], []),
        lambda: calculate_js([0, 0], [1, 0]),
        lambda: calculate_js([1, -1], [1, 0]),
        lambda: calculate_js([1, float("nan")], [1, 0]),
        lambda: calculate_js({"A": 1}, [1]),
        lambda: calculate_cosine_loss([1, 2], [1]),
        lambda: calculate_cosine_loss([1, -1], [1, 0]),
        lambda: calculate_emd([], [1]),
        lambda: calculate_emd([0], [float("inf")]),
        lambda: calculate_emd([-1], [0]),
        lambda: calculate_normalized_emd([0], [1], -1),
        lambda: calculate_normalized_emd([0], [1], float("nan")),
        lambda: calculate_entropy_loss([0, 0], [1, 0]),
    ],
)
def test_invalid_metric_inputs_fail_explicitly(call: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()  # type: ignore[operator]


@pytest.mark.parametrize(
    "function",
    [
        calculate_js,
        calculate_cosine_similarity,
        calculate_cosine_loss,
        calculate_emd,
        calculate_normalized_emd,
        calculate_entropy_loss,
    ],
)
def test_public_phase4_functions_document_input_output_and_paper_step(
    function: object,
) -> None:
    documentation = inspect.getdoc(function) or ""
    assert "Input" in documentation
    assert "Output" in documentation
    assert "Paper experiment step" in documentation


def test_metric_stability_constants_match_project_specification() -> None:
    assert DELTA_PROB == 1e-10
    assert DELTA_DENOM == 1e-8
    assert NEMD_ZERO_SCALE == 1.0

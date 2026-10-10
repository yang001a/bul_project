from __future__ import annotations

import inspect

import pytest

import bul_project.bul as bul
from bul_project.bul import (
    calculate_BUL,
    calculate_bul,
    calculate_fixed_entropy_weights,
    calculate_information_loss,
    calculate_structure_loss,
    calculate_temporal_loss,
)
from bul_project.config import (
    BUL_DIMENSION_WEIGHTS,
    INFORMATION_COMPONENT_WEIGHTS,
    STRUCTURE_COMPONENT_WEIGHTS,
)
from bul_project.metrics import (
    calculate_cosine_loss,
    calculate_entropy_loss,
    calculate_js,
)


def test_structure_loss_uses_exactly_three_stated_loss_components() -> None:
    reference_edges = {("A", "B"): 1, ("A", "C"): 0}
    comparison_edges = {("A", "B"): 0, ("A", "C"): 1}
    variants = {("A", "B"): 1}

    result = calculate_structure_loss(
        reference_edges,
        comparison_edges,
        variants,
        variants,
    )
    expected = (
        calculate_js(reference_edges, comparison_edges)
        + calculate_cosine_loss(reference_edges, comparison_edges)
        + calculate_js(variants, variants)
    ) / 3.0

    assert result == pytest.approx(expected)
    assert result == pytest.approx(2.0 / 3.0, abs=1e-8)


def test_structure_loss_is_zero_for_identical_spaces_and_loss_directed() -> None:
    edges = {("A", "B"): 3, ("B", "C"): 1}
    variants = {("A", "B", "C"): 4}
    identical = calculate_structure_loss(edges, edges, variants, variants)
    distorted = calculate_structure_loss(
        edges,
        {("A", "B"): 0, ("B", "C"): 4},
        variants,
        {("A", "D"): 4},
    )

    assert identical == pytest.approx(0.0)
    assert 0.0 < distorted <= 1.0


def test_structure_weights_support_explicit_renormalized_ablation() -> None:
    reference_edges = {"e1": 1, "e2": 0}
    comparison_edges = {"e1": 0, "e2": 1}
    variant = {"v": 1}
    weights = {
        "edge_js": 1.0,
        "edge_cosine_loss": 0.0,
        "variant_js": 0.0,
    }

    result = calculate_structure_loss(
        reference_edges,
        comparison_edges,
        variant,
        variant,
        weights=weights,
    )
    assert result == pytest.approx(
        calculate_js(reference_edges, comparison_edges)
    )


def test_information_loss_uses_activity_entropy_edge_entropy_and_activity_js() -> None:
    reference_activity = {"A": 1, "B": 1}
    comparison_activity = {"A": 1, "B": 0}
    reference_edges = {"e1": 1, "e2": 1}
    comparison_edges = {"e1": 1, "e2": 0}

    result = calculate_information_loss(
        reference_activity,
        comparison_activity,
        reference_edges,
        comparison_edges,
    )
    expected = (
        calculate_entropy_loss(reference_activity, comparison_activity)
        + calculate_entropy_loss(reference_edges, comparison_edges)
        + calculate_js(reference_activity, comparison_activity)
    ) / 3.0

    assert result == pytest.approx(expected)
    assert 0.0 < result <= 1.0


def test_information_loss_does_not_accept_or_double_count_variant_js() -> None:
    parameters = inspect.signature(calculate_information_loss).parameters
    assert all("variant" not in name for name in parameters)
    same = calculate_information_loss(
        {"A": 1}, {"A": 1}, {"e": 1}, {"e": 1}
    )
    assert same == pytest.approx(0.0)


def test_temporal_loss_matches_hand_calculated_edge_weighted_nemd() -> None:
    edge_ab = ("A", "B")
    edge_bc = ("B", "C")
    reference = {edge_ab: [0.0, 2.0], edge_bc: [0.0]}
    comparison = {edge_ab: [1.0, 3.0], edge_bc: [2.0]}

    result = calculate_temporal_loss(
        reference,
        comparison,
        edge_weights={edge_ab: 0.75, edge_bc: 0.25},
        normalization_scales={edge_ab: 2.0, edge_bc: 4.0},
    )

    # AB: EMD=1, nEMD=0.5; BC: EMD=2, nEMD=0.5.
    assert result == pytest.approx(0.75 * 0.5 + 0.25 * 0.5)


def test_temporal_loss_is_zero_for_identical_samples_and_uses_fixed_edge_set() -> None:
    edge = ("A", "B")
    reference = {edge: [0.0, 1.0], ("X", "Y"): [100.0]}
    comparison = {edge: [0.0, 1.0], ("X", "Y"): [0.0]}

    result = calculate_temporal_loss(
        reference,
        comparison,
        edge_weights={edge: 1.0},
        normalization_scales={edge: 1.0},
    )
    assert result == pytest.approx(0.0)


def test_temporal_zero_scale_uses_one_and_remains_bounded() -> None:
    edge = ("A", "B")
    assert calculate_temporal_loss(
        {edge: [0.0]},
        {edge: [1.0]},
        edge_weights={edge: 1.0},
        normalization_scales={edge: 0.0},
    ) == pytest.approx(1.0)


def test_bul_equal_is_arithmetic_mean_and_preserves_loss_direction() -> None:
    assert calculate_BUL(0.0, 0.0, 0.0) == pytest.approx(0.0)
    assert calculate_BUL(1.0, 1.0, 1.0) == pytest.approx(1.0)
    assert calculate_BUL(0.2, 0.4, 0.6) == pytest.approx(0.4)
    assert calculate_bul(0.2, 0.4, 0.6) == pytest.approx(0.4)


def test_bul_accepts_only_explicit_fixed_weights_that_sum_to_one() -> None:
    weights = {"structure": 0.5, "temporal": 0.25, "information": 0.25}
    result = calculate_BUL(0.2, 0.4, 0.8, weights=weights)
    assert result == pytest.approx(0.2 * 0.5 + 0.4 * 0.25 + 0.8 * 0.25)


def test_fixed_entropy_weights_ignore_constant_column_and_are_reusable() -> None:
    reference_matrix = [
        [0.0, 0.5, 0.2],
        [1.0, 0.0, 0.2],
        [0.5, 1.0, 0.2],
    ]
    weights = calculate_fixed_entropy_weights(reference_matrix)

    assert weights["structure"] == pytest.approx(0.5)
    assert weights["temporal"] == pytest.approx(0.5)
    assert weights["information"] == pytest.approx(0.0)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert calculate_BUL(0.2, 0.6, 1.0, weights=weights) == pytest.approx(0.4)
    assert calculate_BUL(0.4, 0.8, 0.0, weights=weights) == pytest.approx(0.6)


def test_fixed_entropy_weights_fall_back_to_equal_when_all_columns_constant() -> None:
    weights = calculate_fixed_entropy_weights(
        [[0.2, 0.4, 0.6], [0.2, 0.4, 0.6]]
    )
    assert weights == pytest.approx(
        {"structure": 1 / 3, "temporal": 1 / 3, "information": 1 / 3}
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: calculate_structure_loss(
            [1, 0],
            [0, 1],
            [1],
            [1],
            weights={"edge_js": 0.5, "edge_cosine_loss": 0.5},
        ),
        lambda: calculate_information_loss(
            [1],
            [1],
            [1],
            [1],
            weights={
                "activity_entropy_loss": 0.5,
                "edge_entropy_loss": 0.5,
                "activity_js": 0.5,
            },
        ),
        lambda: calculate_temporal_loss(
            {}, {}, edge_weights={}, normalization_scales={}
        ),
        lambda: calculate_temporal_loss(
            {("A", "B"): [0]},
            {},
            edge_weights={("A", "B"): 1.0},
            normalization_scales={("A", "B"): 1.0},
        ),
        lambda: calculate_temporal_loss(
            {("A", "B"): [0]},
            {("A", "B"): [1]},
            edge_weights={("A", "B"): 0.5},
            normalization_scales={("A", "B"): 1.0},
        ),
        lambda: calculate_BUL(-0.1, 0.2, 0.3),
        lambda: calculate_BUL(0.1, 1.2, 0.3),
        lambda: calculate_BUL(
            0.1,
            0.2,
            0.3,
            weights={"structure": 0.5, "temporal": 0.5},
        ),
        lambda: calculate_fixed_entropy_weights([[0.1, 0.2, 0.3]]),
        lambda: calculate_fixed_entropy_weights([[0.1, 0.2], [0.2, 0.3]]),
        lambda: calculate_fixed_entropy_weights(
            [[0.1, 0.2, 0.3], [0.4, float("nan"), 0.6]]
        ),
        lambda: calculate_fixed_entropy_weights(
            [[0.1, 0.2, 0.3], [0.4, 1.2, 0.6]]
        ),
    ],
)
def test_invalid_phase5_inputs_fail_without_silent_renormalization(
    call: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()  # type: ignore[operator]


@pytest.mark.parametrize(
    "function",
    [
        calculate_structure_loss,
        calculate_temporal_loss,
        calculate_information_loss,
        calculate_BUL,
        calculate_fixed_entropy_weights,
    ],
)
def test_public_phase5_functions_document_input_output_and_paper_step(
    function: object,
) -> None:
    documentation = inspect.getdoc(function) or ""
    assert "Input" in documentation
    assert "Output" in documentation
    assert "Paper experiment step" in documentation


def test_phase5_public_name_is_information_not_distribution_loss() -> None:
    assert hasattr(bul, "calculate_information_loss")
    assert not hasattr(bul, "calculate_distribution_loss")


def test_default_weights_are_centralized_and_sum_to_one() -> None:
    assert sum(STRUCTURE_COMPONENT_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(INFORMATION_COMPONENT_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(BUL_DIMENSION_WEIGHTS.values()) == pytest.approx(1.0)

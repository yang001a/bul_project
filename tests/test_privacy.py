from __future__ import annotations

import inspect
import json
import math

import pytest

import bul_project.privacy.release as release_module
from bul_project.behavior import build_g_raw
from bul_project.config import (
    BUDGET_STRATEGIES,
    DEFAULT_DP_EDGE_MODE,
    DP_EDGE_MODE,
    DP_EDGE_MODES,
    EPS_LIST,
    EPSILON_VALUES,
    PRIVACY_BUDGET_RATIOS,
    TEMPORAL_BUDGET_RATIOS,
)
from bul_project.preprocessing.models import Trace
from bul_project.privacy import (
    allocate_privacy_budget,
    calculate_sensitivity,
    clip_trace,
    clip_traces,
    dp_edge_only_release,
    dp_full_release,
    edge_only_epsilon,
    laplace_release,
    release_g_dp,
    split_temporal_budget,
)
from bul_project.privacy.temporal_generator import TemporalDistributionGenerator


def _trace(
    case_id: str,
    activities: tuple[str, ...],
    intervals: tuple[float | None, ...],
) -> Trace:
    return Trace(
        case_id=case_id,
        activities=activities,
        timestamps=(None,) * len(activities),
        transition_seconds=intervals,
    )


def _g_clip():
    raw = [
        _trace("c1", ("A", "B", "C", "X"), (2.0, 10.0, 20.0)),
        _trace("c2", ("A", "B"), (None,)),
        _trace("c3", ("A", "D"), (4.0,)),
    ]
    clipped = clip_traces(raw, l_max=3, t_max=5.0)
    return build_g_raw(clipped, top_k_variants=1, min_edge_support=2)


class _FixedRNG:
    def __init__(self, draw: float) -> None:
        self.draw = draw
        self.calls: list[tuple[float, float]] = []

    def laplace(self, loc: float, scale: float) -> float:
        self.calls.append((loc, scale))
        return self.draw


def test_clip_trace_bounds_length_and_time_without_mutating_input() -> None:
    trace = _trace("c", ("A", "B", "C", "D"), (0.0, 8.0, 100.0))

    clipped = clip_trace(trace, l_max=3, t_max=5.0)

    assert clipped.activities == ("A", "B", "C")
    assert clipped.timestamps == (None, None, None)
    assert clipped.transition_seconds == (0.0, 5.0)
    assert trace.activities == ("A", "B", "C", "D")
    assert trace.transition_seconds == (0.0, 8.0, 100.0)


def test_clip_trace_preserves_missing_and_rejects_invalid_intervals() -> None:
    missing = clip_trace(_trace("m", ("A", "B"), (None,)), 2, 5.0)
    assert missing.transition_seconds == (None,)

    with pytest.raises(ValueError, match="non-negative"):
        clip_trace(_trace("n", ("A", "B"), (-1.0,)), 2, 5.0)
    with pytest.raises(ValueError, match="finite"):
        clip_trace(_trace("i", ("A", "B"), (float("inf"),)), 2, 5.0)
    with pytest.raises(ValueError, match="finite"):
        clip_trace(_trace("nan", ("A", "B"), (float("nan"),)), 2, 5.0)


def test_clip_traces_rejects_empty_or_duplicate_cases() -> None:
    with pytest.raises(ValueError, match="at least one"):
        clip_traces([], 2, 5.0)
    duplicate = [
        _trace("same", ("A",), ()),
        _trace("same", ("B",), ()),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        clip_traces(duplicate, 2, 5.0)


@pytest.mark.parametrize(
    ("l_max", "t_max", "expected"),
    [
        (1, 10.0, (1.0, 0.0, 1.0, 0.0)),
        (3, 5.0, (3.0, 2.0, 1.0, 10.0)),
        (10, 2.5, (10.0, 9.0, 1.0, 22.5)),
    ],
)
def test_calculate_sensitivity_uses_only_paper_formulas(
    l_max: int,
    t_max: float,
    expected: tuple[float, float, float, float],
) -> None:
    result = calculate_sensitivity(l_max, t_max)

    assert (result.activity, result.edge, result.variant, result.time) == expected


@pytest.mark.parametrize(
    ("l_max", "t_max", "expected"),
    [
        (1, 10.0, 0.0),
        (3, 5.0, 50.0),
        (10, 2.5, 56.25),
    ],
)
def test_calculate_sensitivity_includes_paper_squared_time_bound(
    l_max: int,
    t_max: float,
    expected: float,
) -> None:
    result = calculate_sensitivity(l_max, t_max)

    assert result.time_sum == result.time
    assert result.time_squared == expected
    assert result.time_squared_sum == expected
    assert result.to_dict()["time_squared_sum"] == expected


def test_main_epsilon_values_and_all_budget_strategies_are_fixed() -> None:
    assert EPS_LIST == (0.5, 1.0, 2.0, 4.0, 8.0)
    assert EPSILON_VALUES == (0.5, 1.0, 2.0, 4.0, 8.0)
    assert BUDGET_STRATEGIES == (
        "uniform",
        "activity_focus",
        "edge_focus",
        "variant_focus",
        "time_focus",
    )

    for strategy, ratios in PRIVACY_BUDGET_RATIOS.items():
        budget = allocate_privacy_budget(2.0, strategy)
        assert budget.composed_epsilon == pytest.approx(2.0)
        assert budget.activity == pytest.approx(2.0 * ratios["activity"])
        assert budget.edge == pytest.approx(2.0 * ratios["edge"])
        assert budget.variant == pytest.approx(2.0 * ratios["variant"])
        assert budget.time == pytest.approx(2.0 * ratios["time"])
        assert budget.to_dict()["composition"] == "sequential"


def test_temporal_budget_is_half_split_under_sequential_composition() -> None:
    assert TEMPORAL_BUDGET_RATIOS == {
        "time_sum": 0.5,
        "time_squared_sum": 0.5,
    }
    for epsilon_time in EPS_LIST:
        allocated = split_temporal_budget(epsilon_time)
        assert allocated["time_sum"] == pytest.approx(epsilon_time / 2.0)
        assert allocated["time_squared_sum"] == pytest.approx(epsilon_time / 2.0)
        assert sum(allocated.values()) == pytest.approx(epsilon_time)


def test_dp_edge_only_modes_have_distinct_fairness_budgets() -> None:
    assert DP_EDGE_MODES == ("component", "mechanism")
    assert DP_EDGE_MODE == "component"
    assert DEFAULT_DP_EDGE_MODE == "component"
    assert edge_only_epsilon(2.0) == pytest.approx(0.5)
    assert edge_only_epsilon(2.0, "component") == pytest.approx(0.5)
    assert edge_only_epsilon(2.0, "mechanism") == pytest.approx(2.0)


def test_laplace_release_uses_delta_over_epsilon_and_required_postprocessing() -> None:
    rng = _FixedRNG(0.6)
    count = laplace_release(3, 4, 2, statistic="count", rng=rng)
    assert rng.calls == [(0.0, 2.0)]
    assert count == 4

    negative_rng = _FixedRNG(-100.0)
    assert laplace_release(3, 4, 2, statistic="count", rng=negative_rng) == 0
    assert laplace_release(3.0, 4, 2, statistic="time", rng=_FixedRNG(-100.0)) == 0.0

    zero_delta_rng = _FixedRNG(99.0)
    assert laplace_release(3, 0, 2, statistic="count", rng=zero_delta_rng) == 3
    assert zero_delta_rng.calls == []


def test_laplace_release_is_seed_reproducible() -> None:
    first = laplace_release(10, 3, 1, statistic="time", seed=7)
    second = laplace_release(10, 3, 1, statistic="time", seed=7)
    assert first == second


def test_dp_full_release_freezes_domains_composes_budget_and_is_reproducible() -> None:
    g_clip = _g_clip()

    first = dp_full_release(
        g_clip,
        l_max=3,
        t_max=5.0,
        epsilon_total=2.0,
        strategy="uniform",
        seed=0,
    )
    second = dp_full_release(
        g_clip,
        l_max=3,
        t_max=5.0,
        epsilon_total=2.0,
        strategy="uniform",
        seed=0,
    )

    assert first == second
    assert first.method == "DP-full"
    assert first.epsilon_spent == pytest.approx(2.0)
    assert first.component_epsilons == {
        "activity": 0.5,
        "edge": 0.5,
        "variant": 0.5,
        "time": 0.5,
    }
    assert set(first.edge_counts) == {("A", "B")}
    assert set(first.time_sums or {}) == {("A", "B")}
    assert set(first.time_squared_sums or {}) == {("A", "B")}
    assert first.temporal_epsilons == {
        "time_sum": 0.25,
        "time_squared_sum": 0.25,
    }
    assert set(first.variant_counts or {}) == {("A", "B")}
    assert first.variant_other_count is not None
    assert first.activity_probabilities is not None
    assert sum(first.activity_probabilities.values()) == pytest.approx(1.0)
    assert sum(first.edge_probabilities.values()) == pytest.approx(1.0)
    assert first.variant_probabilities is not None
    assert (
        sum(first.variant_probabilities.values())
        + (first.variant_other_probability or 0.0)
    ) == pytest.approx(1.0)
    assert all(value >= 0 for value in (first.activity_counts or {}).values())
    assert all(value >= 0 for value in first.edge_counts.values())
    assert all(value >= 0 for value in (first.variant_counts or {}).values())
    assert all(value >= 0 for value in (first.time_sums or {}).values())
    assert all(value >= 0 for value in (first.time_squared_sums or {}).values())
    payload = json.loads(json.dumps(first.to_dict()))
    assert payload["name"] == "G_DP"
    assert payload["temporal"][0]["squared_sum_seconds2"] >= 0
    assert payload["temporal_epsilons"]["composition"] == "sequential"
    assert payload["temporal_epsilons"]["composed_epsilon_time"] == pytest.approx(
        0.5
    )


@pytest.mark.parametrize(
    ("strategy", "focused"),
    [
        ("activity_focus", "activity"),
        ("edge_focus", "edge"),
        ("variant_focus", "variant"),
        ("time_focus", "time"),
    ],
)
def test_dp_full_focus_strategy_gives_40_percent_to_named_family(
    strategy: str, focused: str
) -> None:
    release = dp_full_release(
        _g_clip(),
        l_max=3,
        t_max=5.0,
        epsilon_total=2.0,
        strategy=strategy,
        seed=1,
    )

    assert release.component_epsilons[focused] == pytest.approx(0.8)
    assert sum(release.component_epsilons.values()) == pytest.approx(2.0)
    assert all(
        epsilon == pytest.approx(0.4)
        for component, epsilon in release.component_epsilons.items()
        if component != focused
    )


def test_dp_edge_only_releases_no_raw_non_edge_components() -> None:
    component = dp_edge_only_release(
        _g_clip(),
        l_max=3,
        t_max=5.0,
        epsilon_total=2.0,
        seed=2,
    )
    mechanism = dp_edge_only_release(
        _g_clip(),
        l_max=3,
        t_max=5.0,
        epsilon_total=2.0,
        seed=2,
        mode="mechanism",
    )

    assert component.dp_edge_mode == "component"
    assert component.epsilon_spent == pytest.approx(0.5)
    assert mechanism.dp_edge_mode == "mechanism"
    assert mechanism.epsilon_spent == pytest.approx(2.0)
    for release in (component, mechanism):
        assert release.method == "DP-edge-only"
        assert set(release.edge_counts) == {("A", "B")}
        assert release.activity_counts is None
        assert release.variant_counts is None
        assert release.time_sums is None
        assert release.time_squared_sums is None
        assert release.temporal_epsilons is None
        payload = release.to_dict()
        assert payload["activity"] is None
        assert payload["variants"] is None
        assert payload["temporal"] is None


def test_release_dispatch_keeps_full_and_edge_only_modes_explicit() -> None:
    g_clip = _g_clip()
    full = release_g_dp(
        g_clip,
        method="DP-full",
        l_max=3,
        t_max=5.0,
        epsilon_total=1.0,
        seed=3,
    )
    edge = release_g_dp(
        g_clip,
        method="DP-edge-only",
        l_max=3,
        t_max=5.0,
        epsilon_total=1.0,
        seed=3,
        dp_edge_mode="mechanism",
    )
    assert full.method == "DP-full"
    assert edge.method == "DP-edge-only"
    assert edge.dp_edge_mode == "mechanism"

    with pytest.raises(ValueError, match="method"):
        release_g_dp(
            g_clip,
            method="invalid",  # type: ignore[arg-type]
            l_max=3,
            t_max=5.0,
            epsilon_total=1.0,
            seed=3,
        )


@pytest.mark.parametrize(
    "call",
    [
        lambda: calculate_sensitivity(0, 1.0),
        lambda: calculate_sensitivity(2, -1.0),
        lambda: allocate_privacy_budget(0.0),
        lambda: split_temporal_budget(0.0),
        lambda: allocate_privacy_budget(1.0, "unknown"),
        lambda: edge_only_epsilon(1.0, "unknown"),
        lambda: laplace_release(1, 1, 1),
        lambda: laplace_release(1, -1, 1, seed=0),
        lambda: laplace_release(1, 1, 0, seed=0),
        lambda: laplace_release(1, 1, 1, seed=30),
    ],
)
def test_invalid_privacy_inputs_fail_explicitly(call: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()  # type: ignore[operator]


@pytest.mark.parametrize(
    "function",
    [
        clip_trace,
        calculate_sensitivity,
        laplace_release,
        allocate_privacy_budget,
        split_temporal_budget,
        edge_only_epsilon,
        dp_full_release,
        dp_edge_only_release,
        release_g_dp,
    ],
)
def test_public_phase3_functions_document_input_output_and_paper_step(
    function: object,
) -> None:
    documentation = inspect.getdoc(function) or ""
    assert "Input" in documentation
    assert "Output" in documentation
    assert "Paper experiment step" in documentation


def test_time_mean_uses_noisy_sum_and_noisy_edge_count() -> None:
    release = dp_full_release(
        _g_clip(),
        l_max=3,
        t_max=5.0,
        epsilon_total=8.0,
        seed=4,
    )
    assert release.time_sums is not None
    assert release.time_means is not None
    for edge, time_sum in release.time_sums.items():
        expected = time_sum / max(release.edge_counts[edge], 1e-8)
        assert math.isclose(release.time_means[edge], expected)


def test_dp_release_temporal_outputs_feed_phase6_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time_calls: list[tuple[float, float, float]] = []

    def deterministic_release(
        value: float,
        sensitivity: float,
        epsilon: float,
        *,
        statistic: str,
        rng: object,
    ) -> float | int:
        if statistic == "time":
            time_calls.append((float(value), sensitivity, epsilon))
            return float(value)
        return int(round(value))

    monkeypatch.setattr(
        release_module,
        "laplace_release",
        deterministic_release,
    )
    release = dp_full_release(
        _g_clip(),
        l_max=3,
        t_max=5.0,
        epsilon_total=2.0,
        strategy="uniform",
        seed=0,
    )

    assert time_calls == [
        (2.0, 10.0, 0.25),
        (4.0, 50.0, 0.25),
    ]
    assert release.time_sums == {("A", "B"): 2.0}
    assert release.time_squared_sums == {("A", "B"): 4.0}

    generated = TemporalDistributionGenerator(
        t_max=5.0,
        seed=release.seed,
    ).generate(
        edge_counts=release.edge_counts,
        time_sums=release.time_sums,
        time_squared_sums=release.time_squared_sums,
    )
    assert set(generated.samples) == {("A", "B")}
    assert generated.seed == release.seed
    assert all(
        0.0 <= value <= 5.0
        for value in generated.samples[("A", "B")]
    )

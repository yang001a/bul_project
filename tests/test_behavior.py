from __future__ import annotations

import json
import inspect

import pytest

from bul_project.behavior import (
    activity_statistics,
    build_g_raw,
    dfg_statistics,
    temporal_statistics,
    variant_statistics,
)
from bul_project.preprocessing.models import Trace


def _trace(
    case_id: str,
    activities: tuple[str, ...],
    intervals: tuple[float | None, ...] | None = None,
) -> Trace:
    """Create a compact contract-valid Trace for hand-calculated tests."""

    if intervals is None:
        intervals = (None,) * max(len(activities) - 1, 0)
    timestamps = (None,) * len(activities)
    return Trace(
        case_id=case_id,
        activities=activities,
        timestamps=timestamps,
        transition_seconds=intervals,
    )


def test_activity_statistics_matches_demo_hand_count(
    demo_traces: list[Trace],
) -> None:
    result = activity_statistics(demo_traces)

    assert result.total_events == 7
    assert result.counts == {
        "A": 1,
        "B": 1,
        "End": 1,
        "Register": 1,
        "Review": 1,
        "Start": 1,
        "Treat": 1,
    }
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_dfg_statistics_keeps_full_counts_and_filters_a_separate_view() -> None:
    traces = [
        _trace("c1", ("A", "B", "C")),
        _trace("c2", ("A", "B")),
        _trace("c3", ("A", "D")),
    ]

    result = dfg_statistics(traces, min_support=2)

    assert result.counts == {("A", "B"): 2, ("A", "D"): 1, ("B", "C"): 1}
    assert result.total_edge_occurrences == 4
    assert result.probabilities[("A", "B")] == pytest.approx(0.5)
    assert result.filtered_counts == {("A", "B"): 2}
    assert result.filtered_probabilities == {("A", "B"): 1.0}
    assert result.dropped_edge_types == 2
    assert result.dropped_edge_occurrences == 2


def test_dfg_filter_may_be_empty_without_falling_back_to_full_space() -> None:
    result = dfg_statistics([_trace("c", ("A", "B"))], min_support=2)

    assert result.counts == {("A", "B"): 1}
    assert result.filtered_counts == {}
    assert result.filtered_probabilities == {}


def test_variant_top_k_plus_other_is_deterministic_and_conservative() -> None:
    traces = [
        _trace("c1", ("A", "B")),
        _trace("c2", ("A", "B")),
        _trace("c3", ("A", "C")),
        _trace("c4", ("B",)),
    ]

    result = variant_statistics(traces, top_k=1)

    assert result.full_counts == {("A", "B"): 2, ("A", "C"): 1, ("B",): 1}
    assert result.top_counts == {("A", "B"): 2}
    assert result.top_probabilities == {("A", "B"): 0.5}
    assert result.other_count == 2
    assert result.other_probability == 0.5
    assert sum(result.top_probabilities.values()) + result.other_probability == 1.0

    tied = variant_statistics(traces, top_k=2)
    assert list(tied.top_counts) == [("A", "B"), ("A", "C")]


def test_variant_full_space_has_zero_other_bucket() -> None:
    result = variant_statistics(
        [_trace("c1", ("A",)), _trace("c2", ("B",))], top_k=None
    )

    assert result.top_counts == result.full_counts
    assert result.other_count == 0
    assert result.other_probability == 0.0

    oversized = variant_statistics(
        [_trace("c1", ("A",)), _trace("c2", ("B",))], top_k=50
    )
    assert oversized.top_counts == oversized.full_counts
    assert oversized.other_count == 0
    assert oversized.other_probability == 0.0


def test_temporal_statistics_uses_edge_support_for_filter_and_both_means() -> None:
    traces = [
        _trace("c1", ("A", "B", "C"), (5.0, 10.0)),
        _trace("c2", ("A", "B"), (None,)),
        _trace("c3", ("A", "B"), (7.0,)),
    ]

    result = temporal_statistics(traces, min_edge_support=2)

    assert result.edge_support_counts == {("A", "B"): 3, ("B", "C"): 1}
    assert result.samples == {("A", "B"): (5.0, 7.0), ("B", "C"): (10.0,)}
    assert result.sample_counts == {("A", "B"): 2, ("B", "C"): 1}
    assert result.sums[("A", "B")] == 12.0
    assert result.observed_means[("A", "B")] == 6.0
    assert result.edge_normalized_means[("A", "B")] == 4.0
    assert result.medians[("A", "B")] == 6.0
    assert result.total_transition_pairs == 4
    assert result.valid_interval_pairs == 3
    assert result.missing_interval_pairs == 1
    assert result.effective_samples == {("A", "B"): (5.0, 7.0)}


def test_temporal_statistics_preserves_zero_and_rejects_negative_or_nonfinite() -> None:
    zero = temporal_statistics([_trace("c0", ("A", "B"), (0.0,))])
    assert zero.samples[("A", "B")] == (0.0,)

    with pytest.raises(ValueError, match="Negative transition interval"):
        temporal_statistics([_trace("cn", ("A", "B"), (-1.0,))])
    with pytest.raises(ValueError, match="Non-finite transition interval"):
        temporal_statistics([_trace("ci", ("A", "B"), (float("inf"),))])
    with pytest.raises(ValueError, match="Non-finite transition interval"):
        temporal_statistics([_trace("cnan", ("A", "B"), (float("nan"),))])
    with pytest.raises(ValueError, match="Non-finite transition interval"):
        temporal_statistics([_trace("cni", ("A", "B"), (float("-inf"),))])


def test_all_missing_temporal_edge_keeps_support_and_empty_distribution() -> None:
    result = temporal_statistics([_trace("c", ("A", "B"), (None,))])

    assert result.edge_support_counts == {("A", "B"): 1}
    assert result.samples == {("A", "B"): ()}
    assert result.sample_counts == {("A", "B"): 0}
    assert result.sums == {("A", "B"): 0.0}
    assert result.observed_means == {("A", "B"): None}
    assert result.edge_normalized_means == {("A", "B"): 0.0}
    assert result.medians == {("A", "B"): None}
    assert result.valid_interval_pairs == 0
    assert result.missing_interval_pairs == 1


def test_single_event_trace_contributes_activity_and_variant_but_no_edge() -> None:
    trace = _trace("single", ("A",), ())
    raw = build_g_raw([trace], top_k_variants=50, min_edge_support=5)

    assert raw.activity.counts == {"A": 1}
    assert raw.variants.full_counts == {("A",): 1}
    assert raw.dfg.counts == {}
    assert raw.temporal.samples == {}


def test_build_g_raw_conserves_all_four_statistical_families(
    demo_traces: list[Trace],
) -> None:
    raw = build_g_raw(
        demo_traces, top_k_variants=1, min_edge_support=1
    )

    assert raw.trace_count == 3
    assert raw.activity.total_events == sum(trace.length for trace in demo_traces)
    assert raw.dfg.total_edge_occurrences == 4
    assert raw.temporal.total_transition_pairs == 4
    assert raw.temporal.edge_support_counts == raw.dfg.counts
    assert raw.variants.total_cases == 3
    assert raw.variants.other_count == 2
    assert json.loads(json.dumps(raw.to_dict()))["name"] == "G_raw"


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: activity_statistics([]), "at least one trace"),
        (
            lambda: variant_statistics([_trace("c", ("A",))], top_k=0),
            "positive integer",
        ),
        (
            lambda: variant_statistics([_trace("c", ("A",))], top_k=True),
            "positive integer",
        ),
        (
            lambda: dfg_statistics([_trace("c", ("A",))], min_support=1.5),
            "positive integer",
        ),
    ],
)
def test_invalid_behavior_inputs_fail_explicitly(call: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        call()  # type: ignore[operator]


def test_duplicate_case_ids_are_rejected() -> None:
    traces = [_trace("same", ("A",)), _trace("same", ("B",))]

    with pytest.raises(ValueError, match="Duplicate case_id"):
        build_g_raw(traces)


@pytest.mark.parametrize(
    "function",
    [
        activity_statistics,
        dfg_statistics,
        variant_statistics,
        temporal_statistics,
        build_g_raw,
    ],
)
def test_public_phase2_functions_document_input_output_and_paper_step(
    function: object,
) -> None:
    documentation = inspect.getdoc(function) or ""

    assert "Input" in documentation
    assert "Output" in documentation
    assert "Paper experiment step" in documentation

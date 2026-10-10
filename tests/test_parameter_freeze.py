from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from bul_project.experiments.parameter_freeze import (
    FrozenConfigError,
    Phase7CDatasetInput,
    Phase7CPolicy,
    ProcessQualityFreezeSettings,
    analyze_phase7c_dataset,
    build_evaluation_log_metadata,
    build_frozen_dataset_inputs,
    finalize_phase7c_parameters,
    run_phase7c_smoke_test,
    select_phase7c_parameters,
    write_phase7c_artifacts,
)
from bul_project.preprocessing.models import Trace


def _trace(
    case_id: str,
    activities: tuple[str, ...],
    intervals: tuple[float | None, ...],
) -> Trace:
    return Trace(
        case_id=case_id,
        activities=activities,
        timestamps=tuple(None for _ in activities),
        transition_seconds=intervals,
    )


def _manual_traces(*, missing_ae_time: bool = False) -> tuple[Trace, ...]:
    traces: list[Trace] = []
    ab_values = (0.0, 0.5, 0.25, 0.5, None)
    for index, ab in enumerate(ab_values):
        traces.append(
            _trace(f"v1_{index}", ("A", "B", "C"), (ab, 10.0))
        )
    v2_intervals = ((0.25, 20.0), (0.5, None), (0.25, 20.0))
    for index, intervals in enumerate(v2_intervals):
        traces.append(
            _trace(f"v2_{index}", ("A", "B", "D"), intervals)
        )
    traces.append(
        _trace(
            "v3_0",
            ("A", "E"),
            (None if missing_ae_time else 5.0,),
        )
    )
    traces.append(_trace("v4_0", ("F",), ()))
    return tuple(traces)


def _evaluation_events(traces: tuple[Trace, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    origin = pd.Timestamp("2025-01-01T00:00:00Z")
    for case_index, trace in enumerate(traces):
        for event_index, activity in enumerate(trace.activities):
            rows.append(
                {
                    "case_id": trace.case_id,
                    "activity": activity,
                    "timestamp": origin
                    + pd.Timedelta(days=case_index, seconds=event_index),
                    "ignored": "not part of the canonical fingerprint",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def manual_input() -> Phase7CDatasetInput:
    traces = _manual_traces()
    return Phase7CDatasetInput(
        dataset="DEMO",
        traces=traces,
        source_sha256="0" * 64,
        evaluation_events=_evaluation_events(traces),
        evaluation_log_id="DEMO_fixed_complete_log",
    )


@pytest.fixture(scope="module")
def manual_policy() -> Phase7CPolicy:
    return Phase7CPolicy(
        top_k_candidates=(1, 2, 4),
        edge_tau_candidates=(1, 3, 5),
        reference_top_k=2,
        reference_edge_tau=3,
        variant_coverage_threshold=0.95,
        edge_mass_threshold=0.90,
        time_sample_threshold=0.90,
        selection_scope="global",
    )


@pytest.fixture(scope="module")
def manual_analysis(manual_input, manual_policy):
    return analyze_phase7c_dataset(manual_input, manual_policy)


def test_profile_bounds_use_trace_lengths_and_adjacent_intervals(
    manual_analysis,
) -> None:
    profile = manual_analysis.profile

    assert profile.case_count == 10
    assert profile.event_count == 27
    assert profile.activity_count == 6
    assert profile.raw_variant_count == 4
    assert profile.minimum_k_for_variant_threshold == 4
    assert profile.p95_trace_length == 3.0
    assert profile.l_max == 3
    assert profile.valid_transition_samples == 15
    assert profile.missing_transition_samples == 2
    assert profile.zero_transition_samples == 1
    assert profile.t_max_seconds == 20.0


def test_variant_coverage_uses_case_mass_and_other_is_complement(
    manual_analysis,
) -> None:
    rows = {row.k: row for row in manual_analysis.variant_coverage}

    assert rows[1].coverage == pytest.approx(0.5)
    assert rows[2].coverage == pytest.approx(0.8)
    assert rows[4].coverage == pytest.approx(1.0)
    assert rows[1].other_mass == pytest.approx(0.5)
    assert rows[2].other_count == 2
    assert [rows[k].coverage for k in (1, 2, 4)] == sorted(
        rows[k].coverage for k in (1, 2, 4)
    )


def test_variant_coverage_is_computed_after_lmax_clipping() -> None:
    traces = [
        _trace(f"common_{index}", ("A", "B"), (1.0,))
        for index in range(20)
    ]
    traces.append(
        _trace(
            "long_outlier",
            ("A", "B", "C", "D", "E", "F"),
            (1.0, 1.0, 1.0, 1.0, 1.0),
        )
    )
    policy = Phase7CPolicy(
        top_k_candidates=(1,),
        edge_tau_candidates=(1,),
        reference_top_k=1,
        reference_edge_tau=1,
        variant_coverage_threshold=1.0,
        edge_mass_threshold=1.0,
        time_sample_threshold=1.0,
    )
    analysis = analyze_phase7c_dataset(
        Phase7CDatasetInput("CLIP", traces, "1" * 64),
        policy,
    )

    assert analysis.profile.raw_variant_count == 2
    assert analysis.profile.l_max == 2
    assert analysis.profile.clipped_variant_count == 1
    assert analysis.coverage_for(1).coverage == 1.0


def test_tau_report_distinguishes_type_mass_and_time_ratios(
    manual_analysis,
) -> None:
    rows = {row.tau: row for row in manual_analysis.edge_thresholds}

    assert rows[1].edge_type_retention == pytest.approx(1.0)
    assert rows[3].edge_type_retention == pytest.approx(3 / 4)
    assert rows[5].edge_type_retention == pytest.approx(1 / 2)
    assert rows[3].occurrence_mass_retention == pytest.approx(16 / 17)
    assert rows[5].occurrence_mass_retention == pytest.approx(13 / 17)
    assert rows[3].valid_time_sample_retention == pytest.approx(14 / 15)
    assert rows[5].valid_time_sample_retention == pytest.approx(12 / 15)


def test_variant_edge_consistency_checks_every_concrete_top_k_variant(
    manual_analysis,
) -> None:
    tau_three = manual_analysis.consistency_for(4, 3)
    tau_one = manual_analysis.consistency_for(4, 1)

    assert tau_three.missing_release_edges == (("A", "E"),)
    assert tau_three.missing_temporal_edges == (("A", "E"),)
    assert tau_three.feasible is False
    assert tau_one.feasible is True

    missing_analysis = analyze_phase7c_dataset(
        Phase7CDatasetInput(
            "MISSING_TIME",
            _manual_traces(missing_ae_time=True),
            "2" * 64,
        ),
        Phase7CPolicy(
            top_k_candidates=(4,),
            edge_tau_candidates=(1,),
            reference_top_k=4,
            reference_edge_tau=1,
            variant_coverage_threshold=1.0,
            edge_mass_threshold=1.0,
            time_sample_threshold=1.0,
        ),
    )
    row = missing_analysis.consistency_for(4, 1)
    assert row.missing_release_edges == ()
    assert row.missing_temporal_edges == (("A", "E"),)
    assert row.feasible is False


def test_normalization_fallback_applies_only_to_exact_zero(
    manual_analysis,
) -> None:
    factors = {row.edge: row for row in manual_analysis.normalization_factors}

    assert factors[("A", "B")].raw_spread_seconds == pytest.approx(0.425)
    assert factors[("A", "B")].normalization_factor_seconds == pytest.approx(
        0.425
    )
    assert factors[("A", "B")].zero_fallback_applied is False
    assert factors[("B", "C")].raw_spread_seconds == 0.0
    assert factors[("B", "C")].normalization_factor_seconds == 1.0
    assert factors[("B", "C")].zero_fallback_applied is True


def test_selection_uses_reference_and_coverage_is_diagnostic(
    manual_analysis,
    manual_policy,
) -> None:
    result = select_phase7c_parameters(
        (manual_analysis,),
        manual_policy,
        process_quality=ProcessQualityFreezeSettings(datasets=("DEMO",)),
    )

    assert result.status == "ready_for_smoke"
    parameters = result.parameter_for("DEMO")
    assert parameters.top_k_variants == 2
    assert parameters.min_edge_support == 3
    assert manual_analysis.coverage_for(2).meets_threshold is False
    assert result.blocking_reasons == ()
    assert set(parameters.normalization_scales) == {
        ("A", "B"),
        ("B", "C"),
        ("B", "D"),
    }


def test_low_coverage_is_reported_without_blocking(
    manual_input: Phase7CDatasetInput,
    tmp_path: Path,
) -> None:
    policy = Phase7CPolicy(
        top_k_candidates=(1, 2, 3),
        edge_tau_candidates=(1, 3, 5),
        reference_top_k=2,
        reference_edge_tau=3,
        variant_coverage_threshold=0.95,
        edge_mass_threshold=0.90,
        time_sample_threshold=0.90,
    )
    analysis = analyze_phase7c_dataset(manual_input, policy)
    result = select_phase7c_parameters((analysis,), policy)
    artifacts = write_phase7c_artifacts(result, tmp_path)

    assert analysis.coverage_for(2).meets_threshold is False
    assert result.status == "ready_for_smoke"
    assert result.blocking_reasons == ()
    assert "experiment_config_final" not in artifacts
    assert not (tmp_path / "experiment_config_final.json").exists()
    assert (tmp_path / "variant_coverage.csv").exists()
    assert (tmp_path / "edge_normalization.json").exists()
    coverage = pd.read_csv(tmp_path / "variant_coverage.csv")
    selected = coverage.loc[coverage["k"] == 2].iloc[0]
    assert selected["selection_role"] == "reference"
    assert selected["coverage_role"] == "diagnostic_only"


def test_reference_tau_failure_is_blocked_and_cannot_write_final_config(
    manual_input: Phase7CDatasetInput,
    tmp_path: Path,
) -> None:
    policy = Phase7CPolicy(
        top_k_candidates=(1, 2, 4),
        edge_tau_candidates=(1, 3, 5),
        reference_top_k=2,
        reference_edge_tau=5,
        variant_coverage_threshold=0.95,
        edge_mass_threshold=0.90,
        time_sample_threshold=0.90,
    )
    analysis = analyze_phase7c_dataset(manual_input, policy)
    result = select_phase7c_parameters((analysis,), policy)
    artifacts = write_phase7c_artifacts(result, tmp_path)

    assert result.status == "blocked"
    assert any("occurrence mass" in reason for reason in result.blocking_reasons)
    assert any("valid-time samples" in reason for reason in result.blocking_reasons)
    assert "experiment_config_final" not in artifacts
    assert not (tmp_path / "experiment_config_final.json").exists()


def test_variant_edge_consistency_blocks_only_process_quality_datasets(
    tmp_path: Path,
) -> None:
    policy = Phase7CPolicy(
        top_k_candidates=(4,),
        edge_tau_candidates=(1,),
        reference_top_k=4,
        reference_edge_tau=1,
        variant_coverage_threshold=1.0,
        edge_mass_threshold=1.0,
        time_sample_threshold=1.0,
    )
    demo_traces = _manual_traces()
    missing_traces = _manual_traces(missing_ae_time=True)
    demo_input = Phase7CDatasetInput(
        "DEMO",
        demo_traces,
        "3" * 64,
        evaluation_events=_evaluation_events(demo_traces),
        evaluation_log_id="DEMO_complete",
    )
    missing_input = Phase7CDatasetInput(
        "MISSING_TIME",
        missing_traces,
        "4" * 64,
    )
    demo_analysis = analyze_phase7c_dataset(demo_input, policy)
    missing_analysis = analyze_phase7c_dataset(missing_input, policy)

    result = select_phase7c_parameters(
        (demo_analysis, missing_analysis),
        policy,
        process_quality=ProcessQualityFreezeSettings(datasets=("DEMO",)),
    )
    assert missing_analysis.consistency_for(4, 1).feasible is False
    assert result.status == "ready_for_smoke"

    write_phase7c_artifacts(result, tmp_path)
    consistency = pd.read_csv(tmp_path / "variant_edge_consistency.csv")
    missing_row = consistency.loc[
        (consistency["dataset"] == "MISSING_TIME")
        & (consistency["k"] == 4)
        & (consistency["tau"] == 1)
    ].iloc[0]
    assert bool(missing_row["required_for_process_quality"]) is False
    assert missing_row["selection_status"] == "not_applicable"

    missing_quality_input = replace(
        missing_input,
        evaluation_events=_evaluation_events(missing_traces),
        evaluation_log_id="MISSING_TIME_complete",
    )
    missing_quality_analysis = analyze_phase7c_dataset(
        missing_quality_input,
        policy,
    )
    blocked = select_phase7c_parameters(
        (demo_analysis, missing_quality_analysis),
        policy,
        process_quality=ProcessQualityFreezeSettings(
            datasets=("MISSING_TIME",)
        ),
    )
    assert blocked.status == "blocked"
    assert any(
        "Variant-edge consistency" in reason
        for reason in blocked.blocking_reasons
    )


def test_evaluation_log_fingerprint_is_canonical_and_stable(
    manual_input: Phase7CDatasetInput,
) -> None:
    events = manual_input.evaluation_events
    assert events is not None
    first = build_evaluation_log_metadata(events, "fixed")
    shuffled = events.sample(frac=1.0, random_state=7).reset_index(drop=True)
    second = build_evaluation_log_metadata(shuffled, "fixed")
    changed = events.copy()
    changed.loc[0, "activity"] = "CHANGED"
    third = build_evaluation_log_metadata(changed, "fixed")

    assert first == second
    assert len(first.sha256) == 64
    assert first.sha256 != third.sha256
    assert first.event_count == 27
    assert first.case_count == 10


def test_phase7c_bridge_and_smoke_freeze_are_reproducible_and_immutable(
    manual_input: Phase7CDatasetInput,
    manual_analysis,
    manual_policy,
    tmp_path: Path,
) -> None:
    preflight = select_phase7c_parameters(
        (manual_analysis,),
        manual_policy,
        process_quality=ProcessQualityFreezeSettings(datasets=("DEMO",)),
    )
    bridged = build_frozen_dataset_inputs(preflight, (manual_input,))
    assert bridged[0].top_k_variants == 2
    assert bridged[0].min_edge_support == 3
    assert bridged[0].evaluation_log_id == "DEMO_fixed_complete_log"

    _, evidence = run_phase7c_smoke_test(
        preflight,
        (manual_input,),
        tmp_path / "smoke",
    )
    frozen = finalize_phase7c_parameters(preflight, evidence)
    artifacts = write_phase7c_artifacts(frozen, tmp_path / "freeze")
    first_bytes = {
        filename: (tmp_path / "freeze" / filename).read_bytes()
        for filename in artifacts.values()
    }
    repeated = write_phase7c_artifacts(frozen, tmp_path / "freeze")

    assert frozen.status == "frozen"
    assert evidence.epsilons == (0.5, 8.0)
    assert evidence.seed == 0
    assert evidence.process_quality_conditions == 1
    assert "experiment_config_final" in artifacts
    assert artifacts == repeated
    assert {
        filename: (tmp_path / "freeze" / filename).read_bytes()
        for filename in repeated.values()
    } == first_bytes

    final_config = json.loads(
        (tmp_path / "freeze" / "experiment_config_final.json").read_text(
            encoding="utf-8"
        )
    )
    assert final_config["policy"]["reference_top_k"] == 2
    assert final_config["policy"]["reference_edge_tau"] == 3
    assert final_config["policy"]["variant_coverage_role"] == "diagnostic_only"
    assert (
        final_config["policy"]["synthetic_consistency_scope"]
        == "process_quality_datasets_only"
    )
    assert final_config["policy"]["sensitivity_analysis_required"] is True
    assert final_config["parameter_robustness"]["full_cross_product_required"] is True

    manifest = json.loads(
        (tmp_path / "freeze" / "phase7c_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for filename, expected_hash in manifest["artifact_sha256"].items():
        assert hashlib.sha256(
            (tmp_path / "freeze" / filename).read_bytes()
        ).hexdigest() == expected_hash

    changed_evidence = replace(
        evidence,
        artifact_sha256={"changed": "f" * 64},
    )
    changed_frozen = finalize_phase7c_parameters(preflight, changed_evidence)
    with pytest.raises(FrozenConfigError, match="different bytes"):
        write_phase7c_artifacts(changed_frozen, tmp_path / "freeze")


def test_bridge_rejects_evaluation_log_changed_after_selection(
    manual_input: Phase7CDatasetInput,
    manual_analysis,
    manual_policy,
) -> None:
    preflight = select_phase7c_parameters(
        (manual_analysis,),
        manual_policy,
        process_quality=ProcessQualityFreezeSettings(datasets=("DEMO",)),
    )
    events = manual_input.evaluation_events
    assert events is not None
    changed = events.copy()
    changed.loc[0, "activity"] = "CHANGED"
    changed_input = replace(manual_input, evaluation_events=changed)

    with pytest.raises(FrozenConfigError, match="changed after freeze"):
        build_frozen_dataset_inputs(preflight, (changed_input,))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"top_k_candidates": ()}, "at least one"),
        ({"top_k_candidates": (1, 1)}, "duplicate"),
        ({"top_k_candidates": (2, 1)}, "increasing"),
        ({"edge_tau_candidates": (True,)}, "positive integers"),
        (
            {"reference_top_k": 25},
            "reference_top_k must be one of",
        ),
        (
            {"reference_edge_tau": 15},
            "reference_edge_tau must be one of",
        ),
        ({"variant_coverage_threshold": 0.0}, r"in \(0,1\]"),
        ({"quantile_method": "nearest"}, "linear"),
        ({"zero_scale_fallback": 0.5}, "fallback 1.0"),
    ],
)
def test_phase7c_policy_validation(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        Phase7CPolicy(**kwargs)

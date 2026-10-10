from __future__ import annotations

import csv
import inspect
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from bul_project.config import DatasetSpec
from bul_project.experiments import (
    SingleExperimentConfig,
    SingleExperimentRunner,
    run_single_experiment,
    write_single_experiment_artifacts,
)
from bul_project.preprocessing import preprocess_dataset
from bul_project.preprocessing.models import Trace


FIXED_DEMO_SCALES = {
    ("A", "B"): 3600.0,
    ("Review", "End"): 3600.0,
    ("Start", "Review"): 3600.0,
}


def _full_config(
    *,
    seed: int = 0,
    evaluate_process_quality: bool = False,
) -> SingleExperimentConfig:
    return SingleExperimentConfig(
        dataset="DEMO",
        method="DP-full",
        epsilon=8.0,
        seed=seed,
        l_max=3,
        t_max=10.0,
        top_k_variants=3,
        min_edge_support=1,
        budget_strategy="uniform",
        dp_edge_mode=None,
        evaluate_process_quality=evaluate_process_quality,
        conformance_variant=(
            "token_based" if evaluate_process_quality else None
        ),
        process_model_noise_threshold=(
            0.0 if evaluate_process_quality else None
        ),
    )


def _edge_config(
    *,
    mode: str = "component",
) -> SingleExperimentConfig:
    return SingleExperimentConfig(
        dataset="DEMO",
        method="DP-edge-only",
        epsilon=8.0,
        seed=0,
        l_max=3,
        t_max=10.0,
        top_k_variants=3,
        min_edge_support=1,
        budget_strategy=None,
        dp_edge_mode=mode,
        evaluate_process_quality=False,
        conformance_variant=None,
        process_model_noise_threshold=None,
    )


def test_dp_full_single_run_records_three_independent_loss_stages(
    demo_traces: list[Trace],
) -> None:
    result = run_single_experiment(
        demo_traces,
        _full_config(),
        normalization_scales=FIXED_DEMO_SCALES,
    )

    assert tuple(result.stage_losses) == (
        "clipping",
        "dp_noise",
        "total",
    )
    for losses in result.stage_losses.values():
        assert 0.0 <= losses.l_structure <= 1.0
        assert losses.l_temporal is not None
        assert losses.l_information is not None
        assert losses.bul is not None
        assert 0.0 <= losses.l_temporal <= 1.0
        assert 0.0 <= losses.l_information <= 1.0
        assert losses.bul == pytest.approx(
            (
                losses.l_structure
                + losses.l_temporal
                + losses.l_information
            )
            / 3.0
        )

    assert result.temporal_distributions is not None
    assert result.synthetic_log is None
    assert result.process_quality is None
    assert result.release.time_squared_sums is not None
    assert (
        result.analysis_metadata["comparison_stages"]["additivity_claimed"]
        is False
    )
    assert result.analysis_metadata["normalization_scale_policy"] == (
        "caller_supplied_fixed_before_epsilon_run"
    )
    assert {
        (row["source"], row["target"])
        for row in result.analysis_metadata["temporal_domain"]
    } == set(FIXED_DEMO_SCALES)

    record = result.to_record()
    assert set(
        (
            "dataset",
            "epsilon",
            "method",
            "seed",
            "L_structure",
            "L_temporal",
            "L_information",
            "BUL",
            "fitness",
            "precision",
            "generalization",
            "runtime",
        )
    ).issubset(record)
    assert record["method"] == "DP-full"
    assert record["L_structure"] == result.total_losses.l_structure
    assert record["fitness"] is None
    assert result.runtime_seconds >= 0
    json.loads(result.metadata_json())


def test_dp_full_single_run_is_seed_reproducible(
    demo_traces: list[Trace],
) -> None:
    config = _full_config(seed=4)

    first = run_single_experiment(
        demo_traces,
        config,
        normalization_scales=FIXED_DEMO_SCALES,
    )
    second = run_single_experiment(
        demo_traces,
        config,
        normalization_scales=FIXED_DEMO_SCALES,
    )

    assert first.stage_losses == second.stage_losses
    assert first.release == second.release
    assert first.temporal_distributions == second.temporal_distributions
    first_record = first.to_record()
    second_record = second.to_record()
    first_record.pop("runtime")
    second_record.pop("runtime")
    assert first_record == second_record


def test_dp_full_optional_process_quality_uses_explicit_evaluation_log(
    demo_spec: DatasetSpec,
) -> None:
    phase1 = preprocess_dataset(demo_spec)
    evaluation_events = phase1.events.dropna(
        subset=["timestamp"]
    ).reset_index(drop=True)

    result = run_single_experiment(
        phase1.traces,
        _full_config(evaluate_process_quality=True),
        normalization_scales=FIXED_DEMO_SCALES,
        evaluation_events=evaluation_events,
        evaluation_log_id="DEMO_complete_timestamp_events",
    )

    assert result.synthetic_log is not None
    assert result.synthetic_log.metadata["raw_data_access"] is False
    assert result.process_quality is not None
    assert 0.0 <= result.process_quality.fitness <= 1.0
    assert 0.0 <= result.process_quality.precision <= 1.0
    assert 0.0 <= result.process_quality.generalization <= 1.0
    assert result.process_quality.metadata["evaluation_log_id"] == (
        "DEMO_complete_timestamp_events"
    )
    assert result.to_record()["fitness"] == result.process_quality.fitness


@pytest.mark.parametrize("mode", ["component", "mechanism"])
def test_dp_edge_only_keeps_unavailable_dimensions_na(
    demo_traces: list[Trace],
    mode: str,
) -> None:
    result = run_single_experiment(
        demo_traces,
        _edge_config(mode=mode),
    )

    assert result.release.dp_edge_mode == mode
    for losses in result.stage_losses.values():
        assert 0.0 <= losses.l_structure <= 1.0
        assert losses.l_temporal is None
        assert losses.l_information is None
        assert losses.bul is None
    assert result.temporal_distributions is None
    assert result.synthetic_log is None
    assert result.process_quality is None
    assert result.release.activity_counts is None
    assert result.release.variant_counts is None
    assert result.release.time_sums is None
    assert result.release.time_squared_sums is None

    record = result.to_record()
    assert record["L_temporal"] is None
    assert record["L_information"] is None
    assert record["BUL"] is None
    assert record["fitness"] is None
    assert record["precision"] is None
    assert record["generalization"] is None
    weights = result.analysis_metadata["edge_only_structure_components"]
    assert weights == pytest.approx(
        {"edge_js": 0.5, "edge_cosine_loss": 0.5}
    )
    assert result.analysis_metadata["raw_variant_or_time_substitution"] is False


def test_single_run_artifacts_write_required_na_and_metadata(
    demo_traces: list[Trace],
    tmp_path: Path,
) -> None:
    result = run_single_experiment(demo_traces, _edge_config())

    artifacts = write_single_experiment_artifacts(result, tmp_path)
    first_csv = (tmp_path / "experiment_result.csv").read_bytes()
    first_metadata = (tmp_path / "experiment_metadata.json").read_bytes()
    repeated = write_single_experiment_artifacts(result, tmp_path)

    assert artifacts == repeated == {
        "experiment_result": "experiment_result.csv",
        "experiment_metadata": "experiment_metadata.json",
    }
    assert (tmp_path / "experiment_result.csv").read_bytes() == first_csv
    assert (
        tmp_path / "experiment_metadata.json"
    ).read_bytes() == first_metadata
    assert {path.name for path in tmp_path.iterdir()} == {
        "experiment_result.csv",
        "experiment_metadata.json",
    }

    with (tmp_path / "experiment_result.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["dataset"] == "DEMO"
    assert row["method"] == "DP-edge-only"
    assert row["L_temporal"] == "NA"
    assert row["L_information"] == "NA"
    assert row["BUL"] == "NA"
    assert row["fitness"] == "NA"
    assert row["precision"] == "NA"
    assert row["generalization"] == "NA"

    metadata = json.loads(
        (tmp_path / "experiment_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["phase"] == "7A"
    assert metadata["stage_losses"]["total"]["BUL"] is None
    assert metadata["release"]["method"] == "DP-edge-only"
    assert metadata["temporal_generation"] is None
    assert metadata["synthetic_log_generation"] is None


def test_dp_full_requires_pre_fixed_temporal_scales(
    demo_traces: list[Trace],
) -> None:
    with pytest.raises(ValueError, match="fixed normalization_scales"):
        run_single_experiment(demo_traces, _full_config())

    with pytest.raises(ValueError, match="missing frozen temporal edges"):
        run_single_experiment(
            demo_traces,
            _full_config(),
            normalization_scales={("A", "B"): 3600.0},
        )


def test_process_quality_inputs_are_never_selected_implicitly(
    demo_traces: list[Trace],
) -> None:
    with pytest.raises(ValueError, match="evaluation_events"):
        run_single_experiment(
            demo_traces,
            _full_config(evaluate_process_quality=True),
            normalization_scales=FIXED_DEMO_SCALES,
        )

    events = pd.DataFrame(
        {
            "case_id": ["c"],
            "activity": ["A"],
            "timestamp": [pd.Timestamp("2026-01-01")],
        }
    )
    with pytest.raises(ValueError, match="unused evaluation log"):
        run_single_experiment(
            demo_traces,
            _full_config(),
            normalization_scales=FIXED_DEMO_SCALES,
            evaluation_events=events,
            evaluation_log_id="unused",
        )

    with pytest.raises(ValueError, match="unused temporal scales"):
        run_single_experiment(
            demo_traces,
            _edge_config(),
            normalization_scales=FIXED_DEMO_SCALES,
        )


@pytest.mark.parametrize(
    "config_factory",
    [
        lambda: SingleExperimentConfig(
            dataset="DEMO",
            method="DP-full",
            epsilon=3.0,
            seed=0,
            l_max=3,
            t_max=10.0,
            top_k_variants=3,
            min_edge_support=1,
            budget_strategy="uniform",
            dp_edge_mode=None,
            evaluate_process_quality=False,
            conformance_variant=None,
            process_model_noise_threshold=None,
        ),
        lambda: SingleExperimentConfig(
            dataset="DEMO",
            method="DP-full",
            epsilon=8.0,
            seed=0,
            l_max=3,
            t_max=10.0,
            top_k_variants=3,
            min_edge_support=1,
            budget_strategy=None,
            dp_edge_mode=None,
            evaluate_process_quality=False,
            conformance_variant=None,
            process_model_noise_threshold=None,
        ),
        lambda: SingleExperimentConfig(
            dataset="DEMO",
            method="DP-edge-only",
            epsilon=8.0,
            seed=0,
            l_max=3,
            t_max=10.0,
            top_k_variants=3,
            min_edge_support=1,
            budget_strategy="uniform",
            dp_edge_mode="component",
            evaluate_process_quality=False,
            conformance_variant=None,
            process_model_noise_threshold=None,
        ),
        lambda: SingleExperimentConfig(
            dataset="DEMO",
            method="DP-edge-only",
            epsilon=8.0,
            seed=0,
            l_max=3,
            t_max=10.0,
            top_k_variants=3,
            min_edge_support=1,
            budget_strategy=None,
            dp_edge_mode="component",
            evaluate_process_quality=True,
            conformance_variant="token_based",
            process_model_noise_threshold=0.0,
        ),
        lambda: SingleExperimentConfig(
            dataset="DEMO",
            method="DP-full",
            epsilon=8.0,
            seed=0,
            l_max=3,
            t_max=10.0,
            top_k_variants=3,
            min_edge_support=1,
            budget_strategy="uniform",
            dp_edge_mode=None,
            evaluate_process_quality=False,
            conformance_variant="token_based",
            process_model_noise_threshold=0.0,
        ),
    ],
)
def test_invalid_single_run_configs_fail_explicitly(
    config_factory: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        config_factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    "function",
    [
        SingleExperimentRunner.run,
        run_single_experiment,
        write_single_experiment_artifacts,
    ],
)
def test_public_phase7a_functions_document_contract_and_paper_step(
    function: object,
) -> None:
    documentation = inspect.getdoc(function) or ""
    assert "Input" in documentation
    assert "Output" in documentation
    assert "Paper experiment step" in documentation


def test_total_runtime_is_finite_and_non_negative(
    demo_traces: list[Trace],
) -> None:
    result = run_single_experiment(demo_traces, _edge_config())
    assert math.isfinite(result.runtime_seconds)
    assert result.runtime_seconds >= 0

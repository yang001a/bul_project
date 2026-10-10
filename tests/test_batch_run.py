from __future__ import annotations

import csv
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path

import pytest

from bul_project.config import BUDGET_STRATEGIES, DatasetSpec
from bul_project.experiments import (
    BatchExperimentConfig,
    BatchExperimentResult,
    BatchExperimentRunner,
    DatasetExperimentInput,
    ProcessQualitySelection,
    run_batch_experiments,
    write_batch_experiment_artifacts,
)
from bul_project.preprocessing import preprocess_dataset
from bul_project.preprocessing.models import Phase1Result


FIXED_DEMO_SCALES = {
    ("A", "B"): 3600.0,
    ("Review", "End"): 3600.0,
    ("Start", "Review"): 3600.0,
}


@pytest.fixture(scope="module")
def demo_phase1(demo_spec: DatasetSpec) -> Phase1Result:
    return preprocess_dataset(demo_spec)


def _dataset_input(
    phase1: Phase1Result,
    *,
    include_evaluation_log: bool,
) -> DatasetExperimentInput:
    return DatasetExperimentInput(
        dataset="DEMO",
        traces=phase1.traces,
        l_max=3,
        t_max=10.0,
        top_k_variants=3,
        min_edge_support=1,
        normalization_scales=FIXED_DEMO_SCALES,
        evaluation_events=(
            phase1.events.dropna(subset=["timestamp"]).reset_index(drop=True)
            if include_evaluation_log
            else None
        ),
        evaluation_log_id=(
            "DEMO_complete_timestamp_events"
            if include_evaluation_log
            else None
        ),
    )


def _small_batch_config(
    *,
    process_quality: bool,
) -> BatchExperimentConfig:
    selection = None
    if process_quality:
        selection = ProcessQualitySelection(
            datasets=("DEMO",),
            epsilons=(8.0,),
            seeds=(0,),
            budget_strategies=("uniform",),
            conformance_variant="token_based",
            noise_threshold=0.0,
        )
    return BatchExperimentConfig(
        epsilons=(8.0,),
        seeds=(0, 1),
        dp_full_strategies=("uniform",),
        dp_edge_modes=("component", "mechanism"),
        process_quality_selection=selection,
    )


@pytest.fixture(scope="module")
def demo_batch_result(
    demo_phase1: Phase1Result,
) -> BatchExperimentResult:
    return run_batch_experiments(
        (
            _dataset_input(
                demo_phase1,
                include_evaluation_log=True,
            ),
        ),
        _small_batch_config(process_quality=True),
    )


def test_default_main_batch_matrix_has_all_required_repetitions() -> None:
    config = BatchExperimentConfig()

    assert config.epsilons == (0.5, 1.0, 2.0, 4.0, 8.0)
    assert config.seeds == tuple(range(30))
    assert config.dp_full_strategies == BUDGET_STRATEGIES
    assert config.dp_edge_modes == ("component",)
    assert config.expected_dp_conditions(1) == 900
    assert config.expected_dp_conditions(3) == 2700

    appendix = BatchExperimentConfig(
        dp_edge_modes=("component", "mechanism")
    )
    assert appendix.expected_dp_conditions(3) == 3150
    assert config.to_dict()["failure_policy"] == "fail_fast"
    assert config.to_dict()["execution_order"] == (
        "sequential_deterministic"
    )


def test_batch_plan_is_deterministic_complete_and_unique(
    demo_phase1: Phase1Result,
) -> None:
    dataset_input = _dataset_input(
        demo_phase1,
        include_evaluation_log=False,
    )
    config = BatchExperimentConfig(
        epsilons=(0.5, 8.0),
        seeds=(0, 1),
        dp_full_strategies=("uniform", "time_focus"),
        dp_edge_modes=("component", "mechanism"),
    )
    runner = BatchExperimentRunner(config)

    first = runner.build_plan((dataset_input,))
    second = runner.build_plan((dataset_input,))

    assert first == second
    assert len(first) == 16
    assert len({condition.condition_id for condition in first}) == 16
    assert [
        (
            condition.method,
            condition.budget_strategy,
            condition.dp_edge_mode,
        )
        for condition in first[:4]
    ] == [
        ("DP-full", "uniform", None),
        ("DP-full", "time_focus", None),
        ("DP-edge-only", None, "component"),
        ("DP-edge-only", None, "mechanism"),
    ]
    assert all(
        condition.evaluate_process_quality is False for condition in first
    )


def test_batch_execution_includes_baselines_all_dp_conditions_and_stages(
    demo_batch_result: BatchExperimentResult,
) -> None:
    result = demo_batch_result

    assert len(result.conditions) == 6
    assert len(result.single_results) == 6
    records = result.experiment_records()
    assert len(records) == 8
    assert [record["method"] for record in records[:2]] == [
        "Raw",
        "ClippedRaw",
    ]
    assert Counter(record["method"] for record in records) == {
        "Raw": 1,
        "ClippedRaw": 1,
        "DP-full": 2,
        "DP-edge-only": 4,
    }
    assert records[0]["L_structure"] == 0.0
    assert records[0]["L_temporal"] == 0.0
    assert records[0]["L_information"] == 0.0
    assert records[0]["BUL"] == 0.0

    first_clipping = result.single_results[0].stage_losses["clipping"]
    assert records[1]["L_structure"] == first_clipping.l_structure
    assert records[1]["L_temporal"] == first_clipping.l_temporal
    assert records[1]["L_information"] == first_clipping.l_information
    assert records[1]["BUL"] == first_clipping.bul

    edge_rows = [
        record for record in records if record["method"] == "DP-edge-only"
    ]
    assert all(record["L_temporal"] is None for record in edge_rows)
    assert all(record["L_information"] is None for record in edge_rows)
    assert all(record["BUL"] is None for record in edge_rows)
    assert all(record["fitness"] is None for record in edge_rows)
    assert all(record["precision"] is None for record in edge_rows)
    assert all(record["generalization"] is None for record in edge_rows)

    quality_rows = [
        record for record in records
        if record["process_quality_evaluated"]
    ]
    assert len(quality_rows) == 1
    assert quality_rows[0]["method"] == "DP-full"
    assert quality_rows[0]["seed"] == 0
    assert quality_rows[0]["fitness"] is not None

    stages = result.stage_records()
    assert len(stages) == 18
    assert Counter(row["stage"] for row in stages) == {
        "clipping": 6,
        "dp_noise": 6,
        "total": 6,
    }
    counts = result.to_dict()["counts"]
    assert counts == {
        "datasets": 1,
        "dp_conditions": 6,
        "baseline_rows": 2,
        "experiment_rows": 8,
        "stage_rows": 18,
        "process_quality_conditions": 1,
        "failed_conditions": 0,
    }


def test_batch_artifacts_are_complete_hashed_and_repeatable(
    demo_batch_result: BatchExperimentResult,
    tmp_path: Path,
) -> None:
    artifacts = write_batch_experiment_artifacts(
        demo_batch_result,
        tmp_path,
    )
    first_bytes = {
        filename: (tmp_path / filename).read_bytes()
        for filename in artifacts.values()
    }
    repeated = write_batch_experiment_artifacts(
        demo_batch_result,
        tmp_path,
    )

    assert artifacts == repeated == {
        "experiment_result": "experiment_result.csv",
        "experiment_stage_result": "experiment_stage_result.csv",
        "experiment_run_metadata": "experiment_run_metadata.jsonl",
        "batch_manifest": "batch_manifest.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == set(
        artifacts.values()
    )
    assert {
        filename: (tmp_path / filename).read_bytes()
        for filename in artifacts.values()
    } == first_bytes

    with (tmp_path / "experiment_result.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        records = list(csv.DictReader(handle))
    edge_row = next(
        record for record in records
        if record["method"] == "DP-edge-only"
    )
    assert edge_row["L_temporal"] == "NA"
    assert edge_row["L_information"] == "NA"
    assert edge_row["BUL"] == "NA"
    assert edge_row["fitness"] == "NA"

    metadata_lines = (
        tmp_path / "experiment_run_metadata.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(metadata_lines) == len(demo_batch_result.conditions)
    first_metadata = json.loads(metadata_lines[0])
    assert first_metadata["condition"]["method"] == "DP-full"
    assert first_metadata["result"]["phase"] == "7A"

    manifest = json.loads(
        (tmp_path / "batch_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["phase"] == "7B"
    assert manifest["counts"]["dp_conditions"] == 6
    for key in (
        "experiment_result",
        "experiment_stage_result",
        "experiment_run_metadata",
    ):
        path = tmp_path / artifacts[key]
        assert manifest["artifact_sha256"][key] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()


def test_dataset_metadata_never_serializes_raw_rows(
    demo_phase1: Phase1Result,
) -> None:
    dataset_input = _dataset_input(
        demo_phase1,
        include_evaluation_log=True,
    )
    metadata = dataset_input.to_dict()

    assert metadata["trace_count"] == 3
    assert metadata["evaluation_event_count"] == 6
    assert metadata["raw_trace_rows_serialized"] is False
    assert metadata["raw_event_rows_serialized"] is False
    assert "traces" not in metadata
    assert "evaluation_events" not in metadata


def test_process_quality_selection_requires_explicit_available_log(
    demo_phase1: Phase1Result,
) -> None:
    dataset_input = _dataset_input(
        demo_phase1,
        include_evaluation_log=False,
    )
    runner = BatchExperimentRunner(
        _small_batch_config(process_quality=True)
    )

    with pytest.raises(ValueError, match="missing its explicit evaluation log"):
        runner.build_plan((dataset_input,))


def test_process_quality_selection_must_be_subset_of_batch(
    demo_phase1: Phase1Result,
) -> None:
    dataset_input = _dataset_input(
        demo_phase1,
        include_evaluation_log=True,
    )
    selection = ProcessQualitySelection(
        datasets=("DEMO",),
        epsilons=(8.0,),
        seeds=(2,),
        budget_strategies=("uniform",),
        conformance_variant="token_based",
        noise_threshold=0.0,
    )
    runner = BatchExperimentRunner(
        BatchExperimentConfig(
            epsilons=(8.0,),
            seeds=(0,),
            dp_full_strategies=("uniform",),
            dp_edge_modes=("component",),
            process_quality_selection=selection,
        )
    )

    with pytest.raises(ValueError, match="seeds must be a subset"):
        runner.build_plan((dataset_input,))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BatchExperimentConfig(epsilons=(3.0,)),
        lambda: BatchExperimentConfig(seeds=(0, 0)),
        lambda: BatchExperimentConfig(
            dp_full_strategies=("unknown",)
        ),
        lambda: BatchExperimentConfig(dp_edge_modes=("unknown",)),
        lambda: ProcessQualitySelection(
            datasets=("DEMO",),
            epsilons=(8.0,),
            seeds=(30,),
            budget_strategies=("uniform",),
            conformance_variant="token_based",
            noise_threshold=0.0,
        ),
        lambda: ProcessQualitySelection(
            datasets=("DEMO",),
            epsilons=(8.0,),
            seeds=(0,),
            budget_strategies=("uniform",),
            conformance_variant="invalid",  # type: ignore[arg-type]
            noise_threshold=0.0,
        ),
    ],
)
def test_invalid_batch_parameters_fail_explicitly(
    factory: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]


def test_duplicate_dataset_names_fail_before_execution(
    demo_phase1: Phase1Result,
) -> None:
    dataset_input = _dataset_input(
        demo_phase1,
        include_evaluation_log=False,
    )
    runner = BatchExperimentRunner(
        BatchExperimentConfig(
            epsilons=(8.0,),
            seeds=(0,),
            dp_full_strategies=("uniform",),
            dp_edge_modes=("component",),
        )
    )

    with pytest.raises(ValueError, match="duplicate dataset names"):
        runner.build_plan((dataset_input, dataset_input))


@pytest.mark.parametrize(
    "function",
    [
        BatchExperimentRunner.build_plan,
        BatchExperimentRunner.run,
        run_batch_experiments,
        write_batch_experiment_artifacts,
    ],
)
def test_public_phase7b_functions_document_contract_and_paper_step(
    function: object,
) -> None:
    documentation = inspect.getdoc(function) or ""
    assert "Input" in documentation
    assert "Output" in documentation
    assert "Paper experiment step" in documentation

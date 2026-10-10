from __future__ import annotations

import inspect
import json

import pandas as pd
import pytest
from pm4py.util import constants as pm4py_constants

from bul_project.privacy.temporal_generator import TemporalDistributionGenerator
from bul_project.validation.process_model import ProcessModelEvaluator
from bul_project.validation.synthetic_log import SyntheticLogGenerator


EDGES = (("A", "B"), ("B", "D"), ("A", "C"), ("C", "D"))


@pytest.fixture(scope="module")
def synthetic_log():
    temporal = TemporalDistributionGenerator(t_max=5.0, seed=6).generate(
        edge_counts={edge: 20 for edge in EDGES},
        time_sums={edge: 40.0 for edge in EDGES},
        time_squared_sums={edge: 100.0 for edge in EDGES},
    )
    return SyntheticLogGenerator(seed=6).generate(
        {("A", "B", "D"): 10, ("A", "C", "D"): 10},
        temporal,
        other_count=5,
    )


def test_token_based_external_quality_is_computed_from_discovered_petri_net(
    synthetic_log,
) -> None:
    evaluator = ProcessModelEvaluator(conformance_variant="token_based")
    result = evaluator.evaluate(
        synthetic_log,
        synthetic_log.events,
        evaluation_log_id="shared-demo-evaluation",
    )

    assert result.fitness == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)
    assert 0.0 <= result.generalization <= 1.0
    assert len(result.model.net.places) > 0
    assert len(result.model.net.transitions) > 0
    assert result.metadata["discovery_algorithm"] == "PM4Py Inductive Miner"
    assert result.metadata["quality_source"] == "discovered_petri_net_not_DP_statistics"
    assert result.metadata["fitness_function"] == "fitness_token_based_replay"
    assert result.metadata["precision_function"] == "precision_token_based_replay"
    assert result.metadata["generalization_function"] == "generalization_tbr"
    assert result.metadata["evaluation_log_id"] == "shared-demo-evaluation"


def test_alignment_variant_is_explicit_and_recorded(synthetic_log) -> None:
    evaluator = ProcessModelEvaluator(conformance_variant="alignments")
    result = evaluator.evaluate(
        synthetic_log,
        synthetic_log.events,
        evaluation_log_id="shared-demo-evaluation",
    )

    assert result.fitness == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)
    assert 0.0 <= result.generalization <= 1.0
    assert result.metadata["conformance_variant"] == "alignments"
    assert result.metadata["fitness_function"] == "fitness_alignments"
    assert result.metadata["precision_function"] == "precision_alignments"


def test_quality_evaluation_is_reproducible_and_log_fingerprint_is_stable(
    synthetic_log,
) -> None:
    evaluator = ProcessModelEvaluator(conformance_variant="token_based")
    first = evaluator.evaluate(
        synthetic_log,
        synthetic_log.events,
        evaluation_log_id="fixed-evaluation",
    )
    second = evaluator.evaluate(
        synthetic_log,
        synthetic_log.events.sample(frac=1.0, random_state=99),
        evaluation_log_id="fixed-evaluation",
    )

    assert (first.fitness, first.precision, first.generalization) == pytest.approx(
        (second.fitness, second.precision, second.generalization)
    )
    assert first.metadata["evaluation_log_sha256"] == second.metadata[
        "evaluation_log_sha256"
    ]
    assert first.metadata["discovery_log_sha256"] == second.metadata[
        "discovery_log_sha256"
    ]


def test_process_quality_metadata_is_json_serializable(synthetic_log) -> None:
    result = ProcessModelEvaluator(conformance_variant="token_based").evaluate(
        synthetic_log,
        synthetic_log.events,
        evaluation_log_id="json-evaluation",
    )
    payload = json.loads(json.dumps(result.to_dict()))
    metadata = json.loads(result.metadata_json())

    assert set(payload) == {"fitness", "precision", "generalization", "metadata"}
    assert metadata["synthetic_seed"] == 6
    assert metadata["discovery_event_count"] == len(synthetic_log.events)
    assert metadata["evaluation_event_count"] == len(synthetic_log.events)
    assert metadata["places"] == len(result.model.net.places)


def test_discovery_requires_synthetic_result_not_dp_statistics_or_dataframe(
    synthetic_log,
) -> None:
    evaluator = ProcessModelEvaluator(conformance_variant="token_based")

    with pytest.raises(TypeError, match="SyntheticLogResult"):
        evaluator.discover(synthetic_log.events)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SyntheticLogResult"):
        evaluator.discover({"edge_count": {("A", "B"): 1}})  # type: ignore[arg-type]


def test_progress_setting_is_restored_after_pm4py_calls(synthetic_log) -> None:
    original = pm4py_constants.SHOW_PROGRESS_BAR
    ProcessModelEvaluator(
        conformance_variant="token_based", show_progress=not original
    ).evaluate(
        synthetic_log,
        synthetic_log.events,
        evaluation_log_id="progress-evaluation",
    )
    assert pm4py_constants.SHOW_PROGRESS_BAR is original


@pytest.mark.parametrize(
    "call",
    [
        lambda: ProcessModelEvaluator(conformance_variant="unknown"),
        lambda: ProcessModelEvaluator(
            conformance_variant="token_based", noise_threshold=-0.1
        ),
        lambda: ProcessModelEvaluator(
            conformance_variant="token_based", noise_threshold=1.1
        ),
    ],
)
def test_invalid_evaluator_configuration_fails_explicitly(call: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()  # type: ignore[operator]


def test_invalid_evaluation_logs_and_identifier_fail_explicitly(
    synthetic_log,
) -> None:
    evaluator = ProcessModelEvaluator(conformance_variant="token_based")
    missing = synthetic_log.events.drop(columns="timestamp")
    invalid_timestamp = synthetic_log.events.copy()
    invalid_timestamp.loc[0, "timestamp"] = None

    with pytest.raises(ValueError, match="missing canonical columns"):
        evaluator.evaluate(
            synthetic_log, missing, evaluation_log_id="missing-column"
        )
    with pytest.raises(ValueError, match="at least one event"):
        evaluator.evaluate(
            synthetic_log,
            synthetic_log.events.iloc[0:0],
            evaluation_log_id="empty",
        )
    with pytest.raises(ValueError, match="missing canonical values"):
        evaluator.evaluate(
            synthetic_log,
            invalid_timestamp,
            evaluation_log_id="bad-timestamp",
        )
    with pytest.raises(ValueError, match="evaluation_log_id"):
        evaluator.evaluate(
            synthetic_log, synthetic_log.events, evaluation_log_id=" "
        )


def test_conformance_variant_and_evaluation_log_are_mandatory_choices() -> None:
    constructor = inspect.signature(ProcessModelEvaluator).parameters
    evaluate = inspect.signature(ProcessModelEvaluator.evaluate).parameters

    assert constructor["conformance_variant"].default is inspect.Parameter.empty
    assert evaluate["evaluation_events"].default is inspect.Parameter.empty
    assert evaluate["evaluation_log_id"].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    "function",
    [ProcessModelEvaluator.discover, ProcessModelEvaluator.evaluate],
)
def test_process_model_interfaces_document_input_output_and_paper_step(
    function: object,
) -> None:
    documentation = inspect.getdoc(function) or ""
    assert "Input" in documentation
    assert "Output" in documentation
    assert "Paper experiment step" in documentation

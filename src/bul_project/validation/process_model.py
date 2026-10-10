"""PM4Py Petri-net discovery and external process-quality evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

import pandas as pd
import pm4py
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.util import constants as pm4py_constants

from .synthetic_log import SyntheticLogResult


CORE_COLUMNS = ("case_id", "activity", "timestamp")
ConformanceVariant = Literal["token_based", "alignments"]


def _validate_probability(value: float, name: str) -> float:
    """Validate probability.

    Inputs:
        value: Candidate value to validate or transform.
        name: Parameter or field name used in validation messages.
    Outputs:
        A value of type `float` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeError(f"PM4Py {name} must be a finite value in [0,1].")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < -1e-12 or numeric > 1 + 1e-12:
        raise RuntimeError(f"PM4Py {name} must be a finite value in [0,1].")
    return min(max(numeric, 0.0), 1.0)


def _canonical_events(events: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return events in the canonical case, activity, and timestamp form.

    Inputs:
        events: Input value used by this operation.
        name: Parameter or field name used in validation messages.
    Outputs:
        A value of type `pd.DataFrame` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if not isinstance(events, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    missing_columns = set(CORE_COLUMNS) - set(events.columns)
    if missing_columns:
        raise ValueError(f"{name} is missing canonical columns: {missing_columns!r}")
    if events.empty:
        raise ValueError(f"{name} must contain at least one event.")
    canonical = events.loc[:, CORE_COLUMNS].copy()
    if canonical.isna().any().any():
        raise ValueError(f"{name} contains missing canonical values.")
    if any(
        not isinstance(value, str) or not value
        for value in canonical["case_id"]
    ):
        raise ValueError(f"{name} contains an invalid case_id.")
    if any(
        not isinstance(value, str) or not value
        for value in canonical["activity"]
    ):
        raise ValueError(f"{name} contains an invalid activity.")
    timestamps = pd.to_datetime(canonical["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{name} contains an invalid timestamp.")
    canonical["timestamp"] = timestamps
    canonical["_event_order"] = range(len(canonical))
    canonical = canonical.sort_values(
        ["case_id", "timestamp", "_event_order"], kind="mergesort"
    ).drop(columns="_event_order")
    return canonical.reset_index(drop=True)


def _event_fingerprint(events: pd.DataFrame) -> str:
    """Compute a deterministic fingerprint for canonical event data.

    Inputs:
        events: Input value used by this operation.
    Outputs:
        A value of type `str` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    rows = (
        f"{case_id}\x1f{activity}\x1f{timestamp.isoformat()}"
        for case_id, activity, timestamp in events.itertuples(index=False, name=None)
    )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PetriNetArtifact:
    """Inductive Miner Petri net and markings with discovery metadata."""

    net: PetriNet
    initial_marking: Marking
    final_marking: Marking
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ProcessQualityResult:
    """External fitness, precision and generalization for one Petri net."""

    fitness: float
    precision: float
    generalization: float
    model: PetriNetArtifact
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return quality values and JSON-compatible audit metadata."""

        return {
            "fitness": self.fitness,
            "precision": self.precision,
            "generalization": self.generalization,
            "metadata": self.metadata,
        }

    def metadata_json(self) -> str:
        """Serialize model-discovery and evaluation metadata."""

        return json.dumps(
            self.metadata,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


class ProcessModelEvaluator:
    """Discover a Petri net from a DP synthetic log and evaluate it in PM4Py.

    The caller must explicitly choose token-based or alignment-based fitness and
    precision and must explicitly provide the unified evaluation log. The
    generalization implementation is PM4Py token-based replay in both cases,
    which is recorded rather than hidden.
    """

    def __init__(
        self,
        *,
        conformance_variant: ConformanceVariant,
        noise_threshold: float = 0.0,
        show_progress: bool = False,
    ) -> None:
        """Initialize the instance and validate its construction inputs.

        Inputs:
            conformance_variant: Input value used by this operation.
            noise_threshold: Input value used by this operation.
            show_progress: Input value used by this operation.
        Outputs:
            None.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        if conformance_variant not in ("token_based", "alignments"):
            raise ValueError(
                "conformance_variant must be 'token_based' or 'alignments'."
            )
        if isinstance(noise_threshold, bool) or not isinstance(
            noise_threshold, Real
        ):
            raise ValueError("noise_threshold must be finite and in [0,1].")
        threshold = float(noise_threshold)
        if not math.isfinite(threshold) or threshold < 0 or threshold > 1:
            raise ValueError("noise_threshold must be finite and in [0,1].")
        self.conformance_variant = conformance_variant
        self.noise_threshold = threshold
        self.show_progress = bool(show_progress)

    def discover(
        self, synthetic_log: SyntheticLogResult
    ) -> PetriNetArtifact:
        """Discover one Petri net from a DP-generated synthetic event log.

        Input
        -----
        synthetic_log:
            A ``SyntheticLogResult`` produced only from DP Variant and temporal
            outputs. A bare DataFrame or direct DP statistics are rejected.

        Output
        ------
        PetriNetArtifact
            PM4Py Petri net, initial/final markings and discovery metadata.

        Paper experiment step
        ---------------------
        Phase 6 external validation: apply PM4Py Inductive Miner to the DP
        synthetic event log before calculating any process-quality metric.
        """

        if not isinstance(synthetic_log, SyntheticLogResult):
            raise TypeError(
                "synthetic_log must be a DP-generated SyntheticLogResult."
            )
        discovery_events = _canonical_events(
            synthetic_log.events, "synthetic_log.events"
        )
        previous_progress = pm4py_constants.SHOW_PROGRESS_BAR
        pm4py_constants.SHOW_PROGRESS_BAR = self.show_progress
        started = time.perf_counter()
        try:
            net, initial_marking, final_marking = (
                pm4py.discover_petri_net_inductive(
                    discovery_events,
                    noise_threshold=self.noise_threshold,
                    activity_key="activity",
                    timestamp_key="timestamp",
                    case_id_key="case_id",
                )
            )
        finally:
            pm4py_constants.SHOW_PROGRESS_BAR = previous_progress
        runtime = time.perf_counter() - started
        metadata: dict[str, Any] = {
            "discovery_algorithm": "PM4Py Inductive Miner",
            "pm4py_version": getattr(pm4py, "__version__", "unknown"),
            "noise_threshold": self.noise_threshold,
            "synthetic_seed": synthetic_log.seed,
            "discovery_event_count": len(discovery_events),
            "discovery_case_count": int(discovery_events["case_id"].nunique()),
            "discovery_log_sha256": _event_fingerprint(discovery_events),
            "places": len(net.places),
            "transitions": len(net.transitions),
            "arcs": len(net.arcs),
            "discovery_runtime_seconds": runtime,
            "quality_source": "discovered_petri_net_not_DP_statistics",
        }
        return PetriNetArtifact(
            net=net,
            initial_marking=initial_marking,
            final_marking=final_marking,
            metadata=metadata,
        )

    def evaluate(
        self,
        synthetic_log: SyntheticLogResult,
        evaluation_events: pd.DataFrame,
        *,
        evaluation_log_id: str,
    ) -> ProcessQualityResult:
        """Discover from synthetic data and score on one explicit evaluation log.

        Input
        -----
        synthetic_log:
            DP-generated discovery log accepted by ``discover``.
        evaluation_events:
            Canonical ``case_id/activity/timestamp`` event table used uniformly
            for every model being compared. The interface never selects or
            substitutes this log implicitly.
        evaluation_log_id:
            Non-empty caller-defined identifier recorded with its SHA-256
            fingerprint so later batch code can enforce one evaluation log.

        Output
        ------
        ProcessQualityResult
            PM4Py log fitness, precision and generalization in ``[0,1]``, plus
            the discovered Petri net and complete method metadata.

        Paper experiment step
        ---------------------
        Phase 6 external validation Scheme A: discover a Petri net from the DP
        synthetic log and calculate established process-quality metrics on a
        unified evaluation log. Metrics are not inferred from DP counts.
        """

        if not isinstance(evaluation_log_id, str) or not evaluation_log_id.strip():
            raise ValueError("evaluation_log_id must be a non-empty string.")
        model = self.discover(synthetic_log)
        evaluation = _canonical_events(evaluation_events, "evaluation_events")

        previous_progress = pm4py_constants.SHOW_PROGRESS_BAR
        pm4py_constants.SHOW_PROGRESS_BAR = self.show_progress
        started = time.perf_counter()
        try:
            common = {
                "activity_key": "activity",
                "timestamp_key": "timestamp",
                "case_id_key": "case_id",
            }
            if self.conformance_variant == "token_based":
                fitness_details = pm4py.fitness_token_based_replay(
                    evaluation,
                    model.net,
                    model.initial_marking,
                    model.final_marking,
                    **common,
                )
                precision_value = pm4py.precision_token_based_replay(
                    evaluation,
                    model.net,
                    model.initial_marking,
                    model.final_marking,
                    **common,
                )
            else:
                fitness_details = pm4py.fitness_alignments(
                    evaluation,
                    model.net,
                    model.initial_marking,
                    model.final_marking,
                    **common,
                )
                precision_value = pm4py.precision_alignments(
                    evaluation,
                    model.net,
                    model.initial_marking,
                    model.final_marking,
                    **common,
                )
            generalization_value = pm4py.generalization_tbr(
                evaluation,
                model.net,
                model.initial_marking,
                model.final_marking,
                **common,
            )
        finally:
            pm4py_constants.SHOW_PROGRESS_BAR = previous_progress
        evaluation_runtime = time.perf_counter() - started

        if "log_fitness" not in fitness_details:
            raise RuntimeError("PM4Py fitness output does not contain log_fitness.")
        fitness = _validate_probability(
            fitness_details["log_fitness"], "log_fitness"
        )
        precision = _validate_probability(precision_value, "precision")
        generalization = _validate_probability(
            generalization_value, "generalization"
        )
        metadata = {
            **model.metadata,
            "status": "success",
            "conformance_variant": self.conformance_variant,
            "fitness_function": (
                "fitness_token_based_replay"
                if self.conformance_variant == "token_based"
                else "fitness_alignments"
            ),
            "precision_function": (
                "precision_token_based_replay"
                if self.conformance_variant == "token_based"
                else "precision_alignments"
            ),
            "generalization_function": "generalization_tbr",
            "evaluation_log_id": evaluation_log_id.strip(),
            "evaluation_log_sha256": _event_fingerprint(evaluation),
            "evaluation_event_count": len(evaluation),
            "evaluation_case_count": int(evaluation["case_id"].nunique()),
            "fitness_details": {
                key: float(value) for key, value in fitness_details.items()
            },
            "evaluation_runtime_seconds": evaluation_runtime,
            "total_runtime_seconds": (
                model.metadata["discovery_runtime_seconds"] + evaluation_runtime
            ),
        }
        return ProcessQualityResult(
            fitness=fitness,
            precision=precision,
            generalization=generalization,
            model=model,
            metadata=metadata,
        )

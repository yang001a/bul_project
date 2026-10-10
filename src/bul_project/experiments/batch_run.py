"""Phase 7B deterministic batch orchestration built on Phase 7A."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

from bul_project.behavior import Edge
from bul_project.config import (
    BUDGET_STRATEGIES,
    CORE_COLUMNS,
    DEFAULT_DP_EDGE_MODE,
    DP_EDGE_MODES,
    EPS_LIST,
    EXPERIMENT_SEEDS,
)
from bul_project.preprocessing.models import Trace
from bul_project.privacy._validation import (
    validate_epsilon,
    validate_experiment_seed,
    validate_l_max,
    validate_t_max,
)

from bul_project.validation.process_model import ConformanceVariant
from .single_run import (
    LossDimensions,
    SingleExperimentConfig,
    SingleExperimentResult,
    run_single_experiment,
)


def _validate_positive_integer(value: int, name: str) -> int:
    """Validate positive integer.

    Inputs:
        value: Candidate value to validate or transform.
        name: Parameter or field name used in validation messages.
    Outputs:
        A value of type `int` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a strictly positive integer.")
    return value


def _validate_unique_nonempty(
    values: Sequence[Any],
    name: str,
) -> tuple[Any, ...]:
    """Validate unique nonempty.

    Inputs:
        values: Values processed by the operation.
        name: Parameter or field name used in validation messages.
    Outputs:
        A value of type `tuple[Any, ...]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    materialized = tuple(values)
    if not materialized:
        raise ValueError(f"{name} must contain at least one value.")
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{name} must not contain duplicate values.")
    return materialized


def _validate_noise_threshold(value: float) -> float:
    """Validate noise threshold.

    Inputs:
        value: Candidate value to validate or transform.
    Outputs:
        A value of type `float` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("noise_threshold must be finite and in [0,1].")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or numeric > 1:
        raise ValueError("noise_threshold must be finite and in [0,1].")
    return numeric


def _edge_rows(values: Mapping[Edge, float]) -> list[dict[str, Any]]:
    """Return edge rows.

    Inputs:
        values: Values processed by the operation.
    Outputs:
        A value of type `list[dict[str, Any]]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    return [
        {
            "source": edge[0],
            "target": edge[1],
            "value": values[edge],
        }
        for edge in sorted(values)
    ]


@dataclass(frozen=True)
class DatasetExperimentInput:
    """In-memory data and fixed parameters for one batch dataset."""

    dataset: str
    traces: Sequence[Trace]
    l_max: int
    t_max: float
    top_k_variants: int
    min_edge_support: int
    normalization_scales: Mapping[Edge, float]
    evaluation_events: pd.DataFrame | None = None
    evaluation_log_id: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the dataclass state after initialization.

        Inputs:
            None.
        Outputs:
            None.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        if not isinstance(self.dataset, str) or not self.dataset.strip():
            raise ValueError("dataset must be a non-empty string.")
        checked_traces = tuple(self.traces)
        if not checked_traces:
            raise ValueError("traces must contain at least one Phase 1 trace.")
        if any(not isinstance(trace, Trace) for trace in checked_traces):
            raise TypeError("traces must contain only Phase 1 Trace objects.")
        case_ids = [trace.case_id for trace in checked_traces]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("traces must contain unique case_id values.")

        checked_scales: dict[Edge, float] = {}
        for edge, value in self.normalization_scales.items():
            if (
                not isinstance(edge, tuple)
                or len(edge) != 2
                or any(not isinstance(label, str) or not label for label in edge)
            ):
                raise ValueError(
                    "normalization_scales keys must be valid transition edges."
                )
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(
                    f"normalization_scales[{edge!r}] must be finite and non-negative."
                )
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError(
                    f"normalization_scales[{edge!r}] must be finite and non-negative."
                )
            checked_scales[edge] = numeric
        if not checked_scales:
            raise ValueError(
                "normalization_scales must contain pre-fixed temporal scales."
            )

        if (self.evaluation_events is None) != (
            self.evaluation_log_id is None
        ):
            raise ValueError(
                "evaluation_events and evaluation_log_id must be supplied together."
            )
        checked_events = self.evaluation_events
        checked_log_id = self.evaluation_log_id
        if checked_events is not None:
            if not isinstance(checked_events, pd.DataFrame):
                raise TypeError("evaluation_events must be a pandas DataFrame.")
            if checked_events.empty:
                raise ValueError("evaluation_events must not be empty.")
            missing_columns = set(CORE_COLUMNS) - set(checked_events.columns)
            if missing_columns:
                raise ValueError(
                    "evaluation_events is missing canonical columns: "
                    f"{missing_columns!r}"
                )
            if not isinstance(checked_log_id, str) or not checked_log_id.strip():
                raise ValueError("evaluation_log_id must be a non-empty string.")
            checked_events = checked_events.loc[:, CORE_COLUMNS].copy(deep=True)
            checked_log_id = checked_log_id.strip()

        object.__setattr__(self, "dataset", self.dataset.strip())
        object.__setattr__(self, "traces", checked_traces)
        object.__setattr__(self, "l_max", validate_l_max(self.l_max))
        object.__setattr__(self, "t_max", validate_t_max(self.t_max))
        object.__setattr__(
            self,
            "top_k_variants",
            _validate_positive_integer(
                self.top_k_variants,
                "top_k_variants",
            ),
        )
        object.__setattr__(
            self,
            "min_edge_support",
            _validate_positive_integer(
                self.min_edge_support,
                "min_edge_support",
            ),
        )
        object.__setattr__(self, "normalization_scales", checked_scales)
        object.__setattr__(self, "evaluation_events", checked_events)
        object.__setattr__(self, "evaluation_log_id", checked_log_id)

    def to_dict(self) -> dict[str, Any]:
        """Return dataset parameters without serializing raw traces or events."""

        events = self.evaluation_events
        return {
            "dataset": self.dataset,
            "trace_count": len(self.traces),
            "l_max": self.l_max,
            "t_max": self.t_max,
            "top_k_variants": self.top_k_variants,
            "min_edge_support": self.min_edge_support,
            "normalization_scales": _edge_rows(
                self.normalization_scales
            ),
            "normalization_scale_policy": (
                "caller_supplied_fixed_before_batch"
            ),
            "evaluation_log_id": self.evaluation_log_id,
            "evaluation_event_count": (
                None if events is None else len(events)
            ),
            "evaluation_case_count": (
                None
                if events is None
                else int(events["case_id"].nunique())
            ),
            "raw_trace_rows_serialized": False,
            "raw_event_rows_serialized": False,
        }


@dataclass(frozen=True)
class ProcessQualitySelection:
    """Explicit DP-full conditions that receive PM4Py external validation."""

    datasets: tuple[str, ...]
    epsilons: tuple[float, ...]
    seeds: tuple[int, ...]
    budget_strategies: tuple[str, ...]
    conformance_variant: ConformanceVariant
    noise_threshold: float

    def __post_init__(self) -> None:
        """Validate and normalize the dataclass state after initialization.

        Inputs:
            None.
        Outputs:
            None.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        datasets = _validate_unique_nonempty(self.datasets, "datasets")
        if any(not isinstance(name, str) or not name.strip() for name in datasets):
            raise ValueError("datasets must contain non-empty strings.")
        datasets = tuple(name.strip() for name in datasets)

        epsilons = _validate_unique_nonempty(self.epsilons, "epsilons")
        checked_epsilons = tuple(
            validate_epsilon(value, "epsilon") for value in epsilons
        )
        if any(value not in EPS_LIST for value in checked_epsilons):
            raise ValueError(
                f"Process-quality epsilons must be selected from {EPS_LIST}."
            )

        seeds = _validate_unique_nonempty(self.seeds, "seeds")
        checked_seeds = tuple(validate_experiment_seed(seed) for seed in seeds)
        strategies = _validate_unique_nonempty(
            self.budget_strategies,
            "budget_strategies",
        )
        if any(strategy not in BUDGET_STRATEGIES for strategy in strategies):
            raise ValueError(
                "Process-quality budget_strategies must be configured strategies."
            )
        if self.conformance_variant not in ("token_based", "alignments"):
            raise ValueError(
                "conformance_variant must be token_based or alignments."
            )

        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "epsilons", checked_epsilons)
        object.__setattr__(self, "seeds", checked_seeds)
        object.__setattr__(self, "budget_strategies", strategies)
        object.__setattr__(
            self,
            "noise_threshold",
            _validate_noise_threshold(self.noise_threshold),
        )

    def matches(
        self,
        dataset: str,
        epsilon: float,
        seed: int,
        budget_strategy: str,
    ) -> bool:
        """Return whether one DP-full condition is externally evaluated."""

        return (
            dataset in self.datasets
            and epsilon in self.epsilons
            and seed in self.seeds
            and budget_strategy in self.budget_strategies
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible external-validation selection metadata."""

        return {
            "datasets": list(self.datasets),
            "epsilons": list(self.epsilons),
            "seeds": list(self.seeds),
            "budget_strategies": list(self.budget_strategies),
            "conformance_variant": self.conformance_variant,
            "noise_threshold": self.noise_threshold,
        }


@dataclass(frozen=True)
class BatchExperimentConfig:
    """Deterministic epsilon, seed, mechanism, and budget-strategy matrix."""

    epsilons: tuple[float, ...] = EPS_LIST
    seeds: tuple[int, ...] = EXPERIMENT_SEEDS
    dp_full_strategies: tuple[str, ...] = BUDGET_STRATEGIES
    dp_edge_modes: tuple[str, ...] = (DEFAULT_DP_EDGE_MODE,)
    process_quality_selection: ProcessQualitySelection | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the dataclass state after initialization.

        Inputs:
            None.
        Outputs:
            None.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        epsilons = _validate_unique_nonempty(self.epsilons, "epsilons")
        checked_epsilons = tuple(
            validate_epsilon(value, "epsilon") for value in epsilons
        )
        if any(value not in EPS_LIST for value in checked_epsilons):
            raise ValueError(
                f"Batch epsilons must be selected from {EPS_LIST}."
            )
        seeds = _validate_unique_nonempty(self.seeds, "seeds")
        checked_seeds = tuple(validate_experiment_seed(seed) for seed in seeds)
        strategies = _validate_unique_nonempty(
            self.dp_full_strategies,
            "dp_full_strategies",
        )
        if any(strategy not in BUDGET_STRATEGIES for strategy in strategies):
            raise ValueError(
                "dp_full_strategies contains an unknown budget strategy."
            )
        edge_modes = _validate_unique_nonempty(
            self.dp_edge_modes,
            "dp_edge_modes",
        )
        if any(mode not in DP_EDGE_MODES for mode in edge_modes):
            raise ValueError("dp_edge_modes contains an unknown fairness mode.")
        if (
            self.process_quality_selection is not None
            and not isinstance(
                self.process_quality_selection,
                ProcessQualitySelection,
            )
        ):
            raise TypeError(
                "process_quality_selection must be ProcessQualitySelection or None."
            )

        object.__setattr__(self, "epsilons", checked_epsilons)
        object.__setattr__(self, "seeds", checked_seeds)
        object.__setattr__(self, "dp_full_strategies", strategies)
        object.__setattr__(self, "dp_edge_modes", edge_modes)

    def expected_dp_conditions(self, dataset_count: int) -> int:
        """Return the DP condition count before adding two baseline rows."""

        count = _validate_positive_integer(dataset_count, "dataset_count")
        per_epsilon_seed = len(self.dp_full_strategies) + len(
            self.dp_edge_modes
        )
        return (
            count
            * len(self.epsilons)
            * len(self.seeds)
            * per_epsilon_seed
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete batch grid and execution policy."""

        return {
            "epsilons": list(self.epsilons),
            "seeds": list(self.seeds),
            "dp_full_strategies": list(self.dp_full_strategies),
            "dp_edge_modes": list(self.dp_edge_modes),
            "process_quality_selection": (
                None
                if self.process_quality_selection is None
                else self.process_quality_selection.to_dict()
            ),
            "execution_order": "sequential_deterministic",
            "failure_policy": "fail_fast",
        }


@dataclass(frozen=True)
class BatchCondition:
    """One deterministic Phase 7A invocation inside a Phase 7B batch."""

    dataset: str
    epsilon: float
    seed: int
    method: str
    budget_strategy: str | None
    dp_edge_mode: str | None
    evaluate_process_quality: bool
    conformance_variant: ConformanceVariant | None
    process_model_noise_threshold: float | None

    @property
    def condition_id(self) -> str:
        """Return a stable human-readable identifier for metadata joins."""

        mechanism = (
            self.budget_strategy
            if self.method == "DP-full"
            else self.dp_edge_mode
        )
        return (
            f"{self.dataset}|{self.method}|epsilon={self.epsilon:g}|"
            f"seed={self.seed}|mechanism={mechanism}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible condition metadata."""

        return {
            "condition_id": self.condition_id,
            "dataset": self.dataset,
            "epsilon": self.epsilon,
            "seed": self.seed,
            "method": self.method,
            "budget_strategy": self.budget_strategy,
            "dp_edge_mode": self.dp_edge_mode,
            "evaluate_process_quality": self.evaluate_process_quality,
            "conformance_variant": self.conformance_variant,
            "process_model_noise_threshold": (
                self.process_model_noise_threshold
            ),
        }

    def single_config(
        self,
        dataset_input: DatasetExperimentInput,
    ) -> SingleExperimentConfig:
        """Build the validated Phase 7A configuration for this condition."""

        return SingleExperimentConfig(
            dataset=self.dataset,
            method=self.method,  # type: ignore[arg-type]
            epsilon=self.epsilon,
            seed=self.seed,
            l_max=dataset_input.l_max,
            t_max=dataset_input.t_max,
            top_k_variants=dataset_input.top_k_variants,
            min_edge_support=dataset_input.min_edge_support,
            budget_strategy=self.budget_strategy,
            dp_edge_mode=self.dp_edge_mode,
            evaluate_process_quality=self.evaluate_process_quality,
            conformance_variant=self.conformance_variant,
            process_model_noise_threshold=(
                self.process_model_noise_threshold
            ),
        )


@dataclass(frozen=True)
class BatchExperimentResult:
    """Successful results and metadata for one complete Phase 7B batch."""

    config: BatchExperimentConfig
    dataset_metadata: tuple[dict[str, Any], ...]
    conditions: tuple[BatchCondition, ...]
    single_results: tuple[SingleExperimentResult, ...]
    runtime_seconds: float

    def _clipping_losses(self) -> dict[str, LossDimensions]:
        """Return clipping losses.

        Inputs:
            None.
        Outputs:
            A value of type `dict[str, LossDimensions]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        clipping: dict[str, LossDimensions] = {}
        for condition, result in zip(
            self.conditions,
            self.single_results,
            strict=True,
        ):
            if condition.method != "DP-full":
                continue
            current = result.stage_losses["clipping"]
            previous = clipping.setdefault(condition.dataset, current)
            if current != previous:
                raise RuntimeError(
                    "Clipping baseline changed across DP conditions for "
                    f"dataset {condition.dataset!r}."
                )
        expected_datasets = {
            metadata["dataset"] for metadata in self.dataset_metadata
        }
        if set(clipping) != expected_datasets:
            raise RuntimeError(
                "Every dataset requires at least one DP-full condition to "
                "derive the ClippedRaw baseline."
            )
        return clipping

    @staticmethod
    def _baseline_record(
        metadata: Mapping[str, Any],
        method: str,
        losses: LossDimensions,
    ) -> dict[str, Any]:
        """Return baseline record.

        Inputs:
            metadata: Metadata associated with the artifact or experiment condition.
            method: Input value used by this operation.
            losses: Input value used by this operation.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return {
            "dataset": metadata["dataset"],
            "epsilon": None,
            "method": method,
            "seed": None,
            "L_structure": losses.l_structure,
            "L_temporal": losses.l_temporal,
            "L_information": losses.l_information,
            "BUL": losses.bul,
            "fitness": None,
            "precision": None,
            "generalization": None,
            "runtime": None,
            "l_max": metadata["l_max"],
            "t_max": metadata["t_max"],
            "top_k_variants": metadata["top_k_variants"],
            "min_edge_support": metadata["min_edge_support"],
            "budget_strategy": None,
            "dp_edge_mode": None,
            "process_quality_evaluated": False,
            "conformance_variant": None,
            "condition_id": f"{metadata['dataset']}|{method}",
            "status": "success",
            "comparison": (
                "G_raw versus G_raw"
                if method == "Raw"
                else "G_raw versus G_clip"
            ),
        }

    def experiment_records(self) -> list[dict[str, Any]]:
        """Return Raw, ClippedRaw, and every DP result in stable order."""

        clipping = self._clipping_losses()
        records: list[dict[str, Any]] = []
        for metadata in self.dataset_metadata:
            dataset = metadata["dataset"]
            records.append(
                self._baseline_record(
                    metadata,
                    "Raw",
                    LossDimensions(
                        l_structure=0.0,
                        l_temporal=0.0,
                        l_information=0.0,
                        bul=0.0,
                    ),
                )
            )
            records.append(
                self._baseline_record(
                    metadata,
                    "ClippedRaw",
                    clipping[dataset],
                )
            )

        for condition, result in zip(
            self.conditions,
            self.single_results,
            strict=True,
        ):
            record = result.to_record()
            record.update(
                {
                    "condition_id": condition.condition_id,
                    "status": "success",
                    "comparison": "G_raw versus G_DP",
                }
            )
            records.append(record)
        return records

    def stage_records(self) -> list[dict[str, Any]]:
        """Return three independent loss-comparison rows per DP condition."""

        records: list[dict[str, Any]] = []
        for condition, result in zip(
            self.conditions,
            self.single_results,
            strict=True,
        ):
            for stage in ("clipping", "dp_noise", "total"):
                losses = result.stage_losses[stage]
                records.append(
                    {
                        "condition_id": condition.condition_id,
                        "dataset": condition.dataset,
                        "epsilon": condition.epsilon,
                        "method": condition.method,
                        "seed": condition.seed,
                        "budget_strategy": condition.budget_strategy,
                        "dp_edge_mode": condition.dp_edge_mode,
                        "stage": stage,
                        **losses.to_dict(),
                    }
                )
        return records

    def to_dict(self) -> dict[str, Any]:
        """Return a compact batch manifest without raw data rows."""

        process_quality_count = sum(
            result.process_quality is not None
            for result in self.single_results
        )
        return {
            "name": "BUL_batch_experiment",
            "phase": "7B",
            "config": self.config.to_dict(),
            "datasets": list(self.dataset_metadata),
            "counts": {
                "datasets": len(self.dataset_metadata),
                "dp_conditions": len(self.conditions),
                "baseline_rows": 2 * len(self.dataset_metadata),
                "experiment_rows": len(self.experiment_records()),
                "stage_rows": len(self.stage_records()),
                "process_quality_conditions": process_quality_count,
                "failed_conditions": 0,
            },
            "runtime_seconds": self.runtime_seconds,
            "condition_plan": [
                condition.to_dict() for condition in self.conditions
            ],
            "raw_data_rows_serialized": False,
        }


class BatchExperimentRunner:
    """Expand and execute a deterministic matrix of Phase 7A conditions."""

    def __init__(self, config: BatchExperimentConfig) -> None:
        """Initialize the instance and validate its construction inputs.

        Inputs:
            config: Configuration controlling the operation.
        Outputs:
            None.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        if not isinstance(config, BatchExperimentConfig):
            raise TypeError("config must be a BatchExperimentConfig.")
        self.config = config

    @staticmethod
    def _validate_dataset_inputs(
        dataset_inputs: Sequence[DatasetExperimentInput],
    ) -> tuple[DatasetExperimentInput, ...]:
        """Validate dataset inputs.

        Inputs:
            dataset_inputs: Input value used by this operation.
        Outputs:
            A value of type `tuple[DatasetExperimentInput, ...]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        materialized = tuple(dataset_inputs)
        if not materialized:
            raise ValueError(
                "dataset_inputs must contain at least one dataset."
            )
        if any(
            not isinstance(item, DatasetExperimentInput)
            for item in materialized
        ):
            raise TypeError(
                "dataset_inputs must contain DatasetExperimentInput objects."
            )
        names = [item.dataset for item in materialized]
        if len(set(names)) != len(names):
            raise ValueError("dataset_inputs contains duplicate dataset names.")
        return materialized

    def build_plan(
        self,
        dataset_inputs: Sequence[DatasetExperimentInput],
    ) -> tuple[BatchCondition, ...]:
        """Expand the configured datasets into a deterministic DP condition plan.

        Input
        -----
        dataset_inputs:
            One validated in-memory input per dataset.

        Output
        ------
        tuple[BatchCondition, ...]
            Dataset-major, epsilon-major, seed-major conditions, with DP-full
            strategies followed by the configured DP-edge-only modes.

        Paper experiment step
        ---------------------
        Phase 7B expands the approved epsilon, seed, budget allocation and
        edge-only fairness matrix without silently dropping repetitions.
        """

        inputs = self._validate_dataset_inputs(dataset_inputs)
        selection = self.config.process_quality_selection
        input_names = {item.dataset for item in inputs}
        if selection is not None:
            unknown_datasets = set(selection.datasets) - input_names
            if unknown_datasets:
                raise ValueError(
                    "Process-quality selection contains datasets absent from "
                    f"the batch: {unknown_datasets!r}"
                )
            if not set(selection.epsilons).issubset(self.config.epsilons):
                raise ValueError(
                    "Process-quality epsilons must be a subset of the batch."
                )
            if not set(selection.seeds).issubset(self.config.seeds):
                raise ValueError(
                    "Process-quality seeds must be a subset of the batch."
                )
            if not set(selection.budget_strategies).issubset(
                self.config.dp_full_strategies
            ):
                raise ValueError(
                    "Process-quality strategies must be a subset of the batch."
                )
            for item in inputs:
                if (
                    item.dataset in selection.datasets
                    and (
                        item.evaluation_events is None
                        or item.evaluation_log_id is None
                    )
                ):
                    raise ValueError(
                        "Selected process-quality dataset is missing its "
                        f"explicit evaluation log: {item.dataset!r}."
                    )

        plan: list[BatchCondition] = []
        for item in inputs:
            for epsilon in self.config.epsilons:
                for seed in self.config.seeds:
                    for strategy in self.config.dp_full_strategies:
                        evaluate = (
                            selection is not None
                            and selection.matches(
                                item.dataset,
                                epsilon,
                                seed,
                                strategy,
                            )
                        )
                        plan.append(
                            BatchCondition(
                                dataset=item.dataset,
                                epsilon=epsilon,
                                seed=seed,
                                method="DP-full",
                                budget_strategy=strategy,
                                dp_edge_mode=None,
                                evaluate_process_quality=evaluate,
                                conformance_variant=(
                                    selection.conformance_variant
                                    if evaluate and selection is not None
                                    else None
                                ),
                                process_model_noise_threshold=(
                                    selection.noise_threshold
                                    if evaluate and selection is not None
                                    else None
                                ),
                            )
                        )
                    for mode in self.config.dp_edge_modes:
                        plan.append(
                            BatchCondition(
                                dataset=item.dataset,
                                epsilon=epsilon,
                                seed=seed,
                                method="DP-edge-only",
                                budget_strategy=None,
                                dp_edge_mode=mode,
                                evaluate_process_quality=False,
                                conformance_variant=None,
                                process_model_noise_threshold=None,
                            )
                        )

        expected = self.config.expected_dp_conditions(len(inputs))
        if len(plan) != expected:
            raise RuntimeError(
                "Batch plan size does not match the configured matrix."
            )
        condition_ids = [condition.condition_id for condition in plan]
        if len(set(condition_ids)) != len(condition_ids):
            raise RuntimeError("Batch plan produced duplicate condition IDs.")
        return tuple(plan)

    def run(
        self,
        dataset_inputs: Sequence[DatasetExperimentInput],
    ) -> BatchExperimentResult:
        """Execute every planned condition through the Phase 7A runner.

        Input
        -----
        dataset_inputs:
            Dataset traces, fixed clipping/statistical-space parameters,
            pre-fixed nEMD scales, and optional explicit evaluation logs.

        Output
        ------
        BatchExperimentResult
            Ordered single-run results, Raw/ClippedRaw baseline support,
            stage-level comparisons, condition plan and runtime metadata.

        Paper experiment step
        ---------------------
        Phase 7B repeats each configured DP method for every epsilon and seed,
        preserving all random repetitions. It delegates every condition to
        Phase 7A and performs no statistical significance test or plotting.
        """

        inputs = self._validate_dataset_inputs(dataset_inputs)
        input_by_name = {item.dataset: item for item in inputs}
        plan = self.build_plan(inputs)
        started = time.perf_counter()
        results: list[SingleExperimentResult] = []

        for condition in plan:
            item = input_by_name[condition.dataset]
            result = run_single_experiment(
                item.traces,
                condition.single_config(item),
                normalization_scales=(
                    item.normalization_scales
                    if condition.method == "DP-full"
                    else None
                ),
                evaluation_events=(
                    item.evaluation_events
                    if condition.evaluate_process_quality
                    else None
                ),
                evaluation_log_id=(
                    item.evaluation_log_id
                    if condition.evaluate_process_quality
                    else None
                ),
            )
            results.append(result)

        batch = BatchExperimentResult(
            config=self.config,
            dataset_metadata=tuple(item.to_dict() for item in inputs),
            conditions=plan,
            single_results=tuple(results),
            runtime_seconds=time.perf_counter() - started,
        )
        batch.experiment_records()
        return batch


def run_batch_experiments(
    dataset_inputs: Sequence[DatasetExperimentInput],
    config: BatchExperimentConfig,
) -> BatchExperimentResult:
    """Run one complete Phase 7B batch matrix.

    Input
    -----
    dataset_inputs, config:
        Validated dataset inputs and one explicit batch grid.

    Output
    ------
    BatchExperimentResult
        Deterministically ordered results from BatchExperimentRunner.

    Paper experiment step
    ---------------------
    Phase 7B batch execution only. This function does not load XES files,
    choose clipping quantiles, calculate statistical tests, or create figures.
    """

    return BatchExperimentRunner(config).run(dataset_inputs)


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest of a file.

    Inputs:
        path: Path to the input or output artifact.
    Outputs:
        A value of type `str` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write json.

    Inputs:
        path: Path to the input or output artifact.
        payload: Structured content to validate, serialize, or write.
    Outputs:
        None.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_batch_experiment_artifacts(
    result: BatchExperimentResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write batch result, stage, run-metadata, and manifest artifacts.

    Input
    -----
    result:
        One successfully completed BatchExperimentResult.
    output_dir:
        Destination directory. Existing Phase 7B files are atomically
        replaced; no statistics or figure directories are populated.

    Output
    ------
    dict[str, str]
        Names of experiment_result.csv, experiment_stage_result.csv,
        experiment_run_metadata.jsonl and batch_manifest.json.

    Paper experiment step
    ---------------------
    Phase 7B persists every epsilon/method/seed row, three independent loss
    stages, complete per-run metadata and a hashed batch manifest. Missing
    dimensions are written as NA and never imputed.
    """

    if not isinstance(result, BatchExperimentResult):
        raise TypeError("result must be a BatchExperimentResult.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    experiment_path = destination / "experiment_result.csv"
    temporary_experiment = experiment_path.with_name(
        f".{experiment_path.name}.tmp"
    )
    pd.DataFrame(result.experiment_records()).to_csv(
        temporary_experiment,
        index=False,
        encoding="utf-8",
        na_rep="NA",
    )
    temporary_experiment.replace(experiment_path)

    stage_path = destination / "experiment_stage_result.csv"
    temporary_stage = stage_path.with_name(f".{stage_path.name}.tmp")
    pd.DataFrame(result.stage_records()).to_csv(
        temporary_stage,
        index=False,
        encoding="utf-8",
        na_rep="NA",
    )
    temporary_stage.replace(stage_path)

    metadata_path = destination / "experiment_run_metadata.jsonl"
    temporary_metadata = metadata_path.with_name(
        f".{metadata_path.name}.tmp"
    )
    with temporary_metadata.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for condition, single_result in zip(
            result.conditions,
            result.single_results,
            strict=True,
        ):
            handle.write(
                json.dumps(
                    {
                        "condition": condition.to_dict(),
                        "result": single_result.to_dict(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    temporary_metadata.replace(metadata_path)

    artifacts = {
        "experiment_result": experiment_path.name,
        "experiment_stage_result": stage_path.name,
        "experiment_run_metadata": metadata_path.name,
        "batch_manifest": "batch_manifest.json",
    }
    manifest_path = destination / artifacts["batch_manifest"]
    manifest = {
        **result.to_dict(),
        "artifacts": artifacts,
        "artifact_sha256": {
            "experiment_result": _sha256_file(experiment_path),
            "experiment_stage_result": _sha256_file(stage_path),
            "experiment_run_metadata": _sha256_file(metadata_path),
        },
    }
    _write_json(manifest_path, manifest)
    return artifacts

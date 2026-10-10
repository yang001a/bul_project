"""Phase 7C real-data preflight and immutable experiment-parameter freeze.

This module is deliberately upstream of Phase 7A/7B orchestration.  It reads
only in-memory Phase 1 contracts supplied by the caller, derives clipping and
statistical-space diagnostics, and never chooses parameters from a DP release
or from a BUL outcome.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from bul_project.behavior import Edge, Variant, build_g_raw, temporal_statistics
from bul_project.config import (
    BUDGET_STRATEGIES,
    CORE_COLUMNS,
    DP_EDGE_MODES,
    EDGE_TAU_OPTIONS,
    EPS_LIST,
    EXPERIMENT_SEEDS,
    LMAX_QUANTILE,
    TEMPORAL_BUDGET_RATIOS,
    TMAX_QUANTILE,
    TOP_K_VARIANT_OPTIONS,
)
from bul_project.preprocessing.models import Phase1Result, Trace
from bul_project.privacy import clip_traces

from .batch_run import (
    BatchExperimentConfig,
    BatchExperimentResult,
    DatasetExperimentInput,
    ProcessQualitySelection,
    run_batch_experiments,
    write_batch_experiment_artifacts,
)


FreezeStatus = Literal["blocked", "ready_for_smoke", "frozen"]
SelectionScope = Literal["global", "per_dataset"]
ConformanceVariant = Literal["token_based", "alignments"]


class FrozenConfigError(RuntimeError):
    """Raised when an existing final freeze would be changed or invalidated."""


def _package_version(package: str) -> str | None:
    """Return the installed version of a Python package when available.

    Inputs:
        package: Input value used by this operation.
    Outputs:
        A value of type `str | None` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _runtime_metadata() -> dict[str, str | None]:
    """Return runtime metadata.

    Inputs:
        None.
    Outputs:
        A value of type `dict[str, str | None]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    return {
        "python": platform.python_version(),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "scipy": _package_version("scipy"),
        "pm4py": _package_version("pm4py"),
        "lxml": _package_version("lxml"),
    }


def _validate_ratio(value: float, name: str) -> float:
    """Validate ratio.

    Inputs:
        value: Candidate value to validate or transform.
        name: Parameter or field name used in validation messages.
    Outputs:
        A value of type `float` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and in (0,1].")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0 or numeric > 1:
        raise ValueError(f"{name} must be finite and in (0,1].")
    return numeric


def _validate_quantile(value: float, name: str) -> float:
    """Validate quantile.

    Inputs:
        value: Candidate value to validate or transform.
        name: Parameter or field name used in validation messages.
    Outputs:
        A value of type `float` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and in (0,1).")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0 or numeric >= 1:
        raise ValueError(f"{name} must be finite and in (0,1).")
    return numeric


def _validate_candidates(values: Sequence[int], name: str) -> tuple[int, ...]:
    """Validate candidates.

    Inputs:
        values: Values processed by the operation.
        name: Parameter or field name used in validation messages.
    Outputs:
        A value of type `tuple[int, ...]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    materialized = tuple(values)
    if not materialized:
        raise ValueError(f"{name} must contain at least one value.")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in materialized
    ):
        raise ValueError(f"{name} must contain only positive integers.")
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{name} must not contain duplicate values.")
    if materialized != tuple(sorted(materialized)):
        raise ValueError(f"{name} must be strictly increasing.")
    return materialized


def _edge_rows(edges: Sequence[Edge]) -> list[dict[str, str]]:
    """Return edge rows.

    Inputs:
        edges: Input value used by this operation.
    Outputs:
        A value of type `list[dict[str, str]]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    return [
        {"source": source, "target": target}
        for source, target in sorted(edges)
    ]


def _variant_edges(variants: Sequence[Variant]) -> tuple[Edge, ...]:
    """Return variant edges.

    Inputs:
        variants: Input value used by this operation.
    Outputs:
        A value of type `tuple[Edge, ...]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    edges = {
        (source, target)
        for variant in variants
        for source, target in zip(variant, variant[1:])
    }
    return tuple(sorted(edges))


def _sha256_bytes(payload: bytes) -> str:
    """Compute the SHA-256 digest of a byte sequence.

    Inputs:
        payload: Structured content to validate, serialize, or write.
    Outputs:
        A value of type `str` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: object) -> bytes:
    """Serialize JSON content to deterministic UTF-8 bytes.

    Inputs:
        payload: Structured content to validate, serialize, or write.
    Outputs:
        A value of type `bytes` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    """Serialize tabular content to deterministic UTF-8 CSV bytes.

    Inputs:
        rows: Input value used by this operation.
        columns: Input value used by this operation.
    Outputs:
        A value of type `bytes` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    buffer = io.StringIO(newline="")
    pd.DataFrame(rows, columns=columns).to_csv(
        buffer,
        index=False,
        lineterminator="\n",
        na_rep="NA",
    )
    return buffer.getvalue().encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write a byte sequence atomically.

    Inputs:
        path: Path to the input or output artifact.
        payload: Structured content to validate, serialize, or write.
    Outputs:
        None.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


@dataclass(frozen=True)
class Phase7CPolicy:
    """Predeclared rules for Phase 7C diagnostics and parameter selection."""

    top_k_candidates: tuple[int, ...] = TOP_K_VARIANT_OPTIONS
    edge_tau_candidates: tuple[int, ...] = EDGE_TAU_OPTIONS
    reference_top_k: int = 100
    reference_edge_tau: int = 10
    variant_coverage_threshold: float = 0.95
    edge_mass_threshold: float = 0.95
    time_sample_threshold: float = 0.95
    selection_scope: SelectionScope = "global"
    l_max_quantile: float = LMAX_QUANTILE
    t_max_quantile: float = TMAX_QUANTILE
    normalization_low_quantile: float = 0.05
    normalization_high_quantile: float = 0.95
    quantile_method: str = "linear"
    zero_scale_fallback: float = 1.0

    def __post_init__(self) -> None:
        """Validate and normalize the dataclass state after initialization.

        Inputs:
            None.
        Outputs:
            None.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        object.__setattr__(
            self,
            "top_k_candidates",
            _validate_candidates(self.top_k_candidates, "top_k_candidates"),
        )
        object.__setattr__(
            self,
            "edge_tau_candidates",
            _validate_candidates(self.edge_tau_candidates, "edge_tau_candidates"),
        )
        if self.reference_top_k not in self.top_k_candidates:
            raise ValueError(
                "reference_top_k must be one of top_k_candidates."
            )
        if self.reference_edge_tau not in self.edge_tau_candidates:
            raise ValueError(
                "reference_edge_tau must be one of edge_tau_candidates."
            )
        object.__setattr__(
            self,
            "variant_coverage_threshold",
            _validate_ratio(
                self.variant_coverage_threshold,
                "variant_coverage_threshold",
            ),
        )
        object.__setattr__(
            self,
            "edge_mass_threshold",
            _validate_ratio(self.edge_mass_threshold, "edge_mass_threshold"),
        )
        object.__setattr__(
            self,
            "time_sample_threshold",
            _validate_ratio(
                self.time_sample_threshold,
                "time_sample_threshold",
            ),
        )
        if self.selection_scope not in ("global", "per_dataset"):
            raise ValueError("selection_scope must be global or per_dataset.")
        object.__setattr__(
            self,
            "l_max_quantile",
            _validate_quantile(self.l_max_quantile, "l_max_quantile"),
        )
        object.__setattr__(
            self,
            "t_max_quantile",
            _validate_quantile(self.t_max_quantile, "t_max_quantile"),
        )
        low = _validate_quantile(
            self.normalization_low_quantile,
            "normalization_low_quantile",
        )
        high = _validate_quantile(
            self.normalization_high_quantile,
            "normalization_high_quantile",
        )
        if low >= high:
            raise ValueError(
                "normalization_low_quantile must be below the high quantile."
            )
        object.__setattr__(self, "normalization_low_quantile", low)
        object.__setattr__(self, "normalization_high_quantile", high)
        if self.quantile_method != "linear":
            raise ValueError(
                "Phase 7C freezes quantile_method='linear' for reproducibility."
            )
        if self.zero_scale_fallback != 1.0:
            raise ValueError(
                "Phase 7C requires the exact-zero normalization fallback 1.0."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Inputs:
            None.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        data = asdict(self)
        data["top_k_candidates"] = list(self.top_k_candidates)
        data["edge_tau_candidates"] = list(self.edge_tau_candidates)
        data["top_k_selection_rule"] = (
            "predeclared central reference; Variant coverage is diagnostic only"
        )
        data["edge_tau_selection_rule"] = (
            "predeclared central reference meeting occurrence-mass and "
            "valid-time requirements"
        )
        data["variant_coverage_role"] = "diagnostic_only"
        data["other_mass_policy"] = "retained_in_variant_distribution_and_BUL"
        data["synthetic_consistency_scope"] = "process_quality_datasets_only"
        data["sensitivity_analysis_required"] = True
        data["sensitivity_grid"] = {
            "top_k": list(self.top_k_candidates),
            "edge_tau": list(self.edge_tau_candidates),
            "full_cross_product": True,
        }
        return data


@dataclass(frozen=True)
class ProcessQualityFreezeSettings:
    """Explicit PM4Py settings and datasets whose evaluation log is frozen."""

    datasets: tuple[str, ...]
    conformance_variant: ConformanceVariant = "token_based"
    noise_threshold: float = 0.0

    def __post_init__(self) -> None:
        """Validate and normalize the dataclass state after initialization.

        Inputs:
            None.
        Outputs:
            None.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        datasets = tuple(name.strip() for name in self.datasets)
        if not datasets or any(not name for name in datasets):
            raise ValueError("process-quality datasets must be non-empty strings.")
        if len(set(datasets)) != len(datasets):
            raise ValueError("process-quality datasets must be unique.")
        if self.conformance_variant not in ("token_based", "alignments"):
            raise ValueError(
                "conformance_variant must be token_based or alignments."
            )
        if isinstance(self.noise_threshold, bool) or not isinstance(
            self.noise_threshold,
            Real,
        ):
            raise ValueError("noise_threshold must be finite and in [0,1].")
        threshold = float(self.noise_threshold)
        if not math.isfinite(threshold) or threshold < 0 or threshold > 1:
            raise ValueError("noise_threshold must be finite and in [0,1].")
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "noise_threshold", threshold)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Inputs:
            None.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return {
            "datasets": list(self.datasets),
            "conformance_variant": self.conformance_variant,
            "inductive_miner_noise_threshold": self.noise_threshold,
            "evaluation_log_policy": (
                "one explicit complete canonical log and SHA-256 per dataset; "
                "identical across epsilon, method, and seed"
            ),
        }


@dataclass(frozen=True)
class Phase7CDatasetInput:
    """In-memory Phase 1 data supplied explicitly to the Phase 7C analyzer."""

    dataset: str
    traces: Sequence[Trace]
    source_sha256: str
    source_path: str | None = None
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
        if not isinstance(self.source_sha256, str) or not self.source_sha256.strip():
            raise ValueError("source_sha256 must be a non-empty string.")
        if (self.evaluation_events is None) != (self.evaluation_log_id is None):
            raise ValueError(
                "evaluation_events and evaluation_log_id must be supplied together."
            )
        if self.evaluation_events is not None and not isinstance(
            self.evaluation_events,
            pd.DataFrame,
        ):
            raise TypeError("evaluation_events must be a pandas DataFrame.")
        if self.evaluation_log_id is not None and (
            not isinstance(self.evaluation_log_id, str)
            or not self.evaluation_log_id.strip()
        ):
            raise ValueError("evaluation_log_id must be a non-empty string.")
        object.__setattr__(self, "dataset", self.dataset.strip())
        object.__setattr__(self, "traces", checked_traces)
        object.__setattr__(self, "source_sha256", self.source_sha256.strip())
        object.__setattr__(
            self,
            "evaluation_log_id",
            None
            if self.evaluation_log_id is None
            else self.evaluation_log_id.strip(),
        )

    @classmethod
    def from_phase1_result(
        cls,
        result: Phase1Result,
        *,
        include_evaluation_log: bool,
        evaluation_log_id: str | None = None,
    ) -> "Phase7CDatasetInput":
        """Create a Phase 7C input without rereading or rewriting an XES file.

        Input
        -----
        result:
            One already completed Phase 1 in-memory result.
        include_evaluation_log:
            Whether the complete canonical Phase 1 event table is explicitly
            designated as the fixed PM4Py evaluation log.
        evaluation_log_id:
            Optional explicit identifier.  A deterministic dataset identifier
            is used when the log is included and this value is omitted.

        Output
        ------
        Phase7CDatasetInput
            An immutable input contract.  No XES reader is called here.

        Paper experiment step
        ---------------------
        Phase 7C Step 1 and Step 5: carry forward the audited Phase 1 traces,
        source hash, and optional unified external-evaluation log.
        """

        if not isinstance(result, Phase1Result):
            raise TypeError("result must be a Phase1Result.")
        if not isinstance(include_evaluation_log, bool):
            raise TypeError("include_evaluation_log must be bool.")
        log_id = evaluation_log_id
        if include_evaluation_log and log_id is None:
            log_id = f"{result.dataset}_phase1_complete_canonical_log"
        if not include_evaluation_log and evaluation_log_id is not None:
            raise ValueError(
                "evaluation_log_id cannot be set when the log is excluded."
            )
        return cls(
            dataset=result.dataset,
            traces=result.traces,
            source_sha256=result.source_sha256,
            source_path=result.source_path,
            evaluation_events=(result.events if include_evaluation_log else None),
            evaluation_log_id=log_id,
        )


@dataclass(frozen=True)
class EvaluationLogMetadata:
    """Canonical fingerprint of one caller-selected PM4Py evaluation log."""

    evaluation_log_id: str
    sha256: str
    event_count: int
    case_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Inputs:
            None.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return {
            "evaluation_log_id": self.evaluation_log_id,
            "evaluation_log_sha256": self.sha256,
            "event_count": self.event_count,
            "case_count": self.case_count,
            "core_columns": list(CORE_COLUMNS),
            "timestamp_policy": "complete canonical timestamps required",
            "condition_policy": (
                "same event rows and fingerprint for every epsilon, method, "
                "and seed in this dataset"
            ),
        }


def build_evaluation_log_metadata(
    events: pd.DataFrame,
    evaluation_log_id: str,
) -> EvaluationLogMetadata:
    """Validate, canonicalize, and fingerprint a fixed evaluation log.

    Input
    -----
    events:
        Canonical event table containing case_id, activity, and timestamp.
    evaluation_log_id:
        Caller-selected stable identifier used by all experiment conditions.

    Output
    ------
    EvaluationLogMetadata
        SHA-256 and row/case counts calculated with the same canonical ordering
        used by the Phase 6 PM4Py evaluator.

    Paper experiment step
    ---------------------
    Phase 7C Step 1/6: freeze one external evaluation log before inspecting any
    DP process-quality result.  Missing timestamps are rejected, not selected
    differently for different epsilon values.
    """

    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a pandas DataFrame.")
    if not isinstance(evaluation_log_id, str) or not evaluation_log_id.strip():
        raise ValueError("evaluation_log_id must be a non-empty string.")
    missing_columns = set(CORE_COLUMNS) - set(events.columns)
    if missing_columns:
        raise ValueError(
            f"evaluation log is missing canonical columns: {missing_columns!r}"
        )
    if events.empty:
        raise ValueError("evaluation log must contain at least one event.")
    canonical = events.loc[:, CORE_COLUMNS].copy()
    if canonical.isna().any().any():
        raise ValueError("evaluation log contains missing canonical values.")
    if any(
        not isinstance(value, str) or not value
        for value in canonical["case_id"]
    ):
        raise ValueError("evaluation log contains an invalid case_id.")
    if any(
        not isinstance(value, str) or not value
        for value in canonical["activity"]
    ):
        raise ValueError("evaluation log contains an invalid activity.")
    timestamps = pd.to_datetime(canonical["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("evaluation log contains an invalid timestamp.")
    canonical["timestamp"] = timestamps
    canonical["_event_order"] = range(len(canonical))
    canonical = canonical.sort_values(
        ["case_id", "timestamp", "_event_order"],
        kind="mergesort",
    ).drop(columns="_event_order").reset_index(drop=True)
    rows = (
        f"{case_id}\x1f{activity}\x1f{timestamp.isoformat()}"
        for case_id, activity, timestamp in canonical.itertuples(
            index=False,
            name=None,
        )
    )
    fingerprint = hashlib.sha256(
        "\n".join(rows).encode("utf-8")
    ).hexdigest()
    return EvaluationLogMetadata(
        evaluation_log_id=evaluation_log_id.strip(),
        sha256=fingerprint,
        event_count=len(canonical),
        case_count=int(canonical["case_id"].nunique()),
    )


@dataclass(frozen=True)
class DatasetProfile:
    dataset: str
    case_count: int
    event_count: int
    activity_count: int
    raw_variant_count: int
    clipped_variant_count: int
    minimum_k_for_variant_threshold: int
    mean_trace_length: float
    p95_trace_length: float
    l_max: int
    valid_transition_samples: int
    missing_transition_samples: int
    zero_transition_samples: int
    p95_transition_seconds: float
    t_max_seconds: float
    unavailable_timestamp_events: int

    def to_row(self) -> dict[str, Any]:
        """Return the record as one tabular output row.

        Inputs:
            None.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return asdict(self)


@dataclass(frozen=True)
class VariantCoverageRow:
    dataset: str
    k: int
    effective_k: int
    total_cases: int
    top_k_case_count: int
    coverage: float
    other_count: int
    other_mass: float
    meets_threshold: bool
    selected_variants: tuple[Variant, ...]

    def to_row(self) -> dict[str, Any]:
        """Return the record as one tabular output row.

        Inputs:
            None.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return {
            "dataset": self.dataset,
            "k": self.k,
            "effective_k": self.effective_k,
            "total_cases": self.total_cases,
            "top_k_case_count": self.top_k_case_count,
            "coverage": self.coverage,
            "other_count": self.other_count,
            "other_mass": self.other_mass,
            "meets_threshold": self.meets_threshold,
        }


@dataclass(frozen=True)
class EdgeThresholdRow:
    dataset: str
    tau: int
    total_edge_types: int
    retained_edge_types: int
    edge_type_retention: float
    total_edge_occurrences: int
    retained_edge_occurrences: int
    occurrence_mass_retention: float
    total_valid_time_samples: int
    retained_valid_time_samples: int
    valid_time_sample_retention: float
    retained_temporal_valid_edges: int
    temporal_valid_edge_ratio: float
    meets_mass_threshold: bool
    meets_time_threshold: bool
    retained_edges: tuple[Edge, ...]
    temporal_valid_edges: tuple[Edge, ...]

    def to_row(self) -> dict[str, Any]:
        """Return the record as one tabular output row.

        Inputs:
            None.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return {
            "dataset": self.dataset,
            "tau": self.tau,
            "total_edge_types": self.total_edge_types,
            "retained_edge_types": self.retained_edge_types,
            "edge_type_retention": self.edge_type_retention,
            "total_edge_occurrences": self.total_edge_occurrences,
            "retained_edge_occurrences": self.retained_edge_occurrences,
            "occurrence_mass_retention": self.occurrence_mass_retention,
            "total_valid_time_samples": self.total_valid_time_samples,
            "retained_valid_time_samples": self.retained_valid_time_samples,
            "valid_time_sample_retention": self.valid_time_sample_retention,
            "retained_temporal_valid_edges": (
                self.retained_temporal_valid_edges
            ),
            "temporal_valid_edge_ratio": self.temporal_valid_edge_ratio,
            "meets_mass_threshold": self.meets_mass_threshold,
            "meets_time_threshold": self.meets_time_threshold,
        }


@dataclass(frozen=True)
class VariantEdgeConsistencyRow:
    dataset: str
    k: int
    tau: int
    required_edge_count: int
    release_domain_edge_count: int
    temporal_valid_edge_count: int
    missing_release_edges: tuple[Edge, ...]
    missing_temporal_edges: tuple[Edge, ...]
    feasible: bool

    def to_row(self) -> dict[str, Any]:
        """Return the record as one tabular output row.

        Inputs:
            None.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return {
            "dataset": self.dataset,
            "k": self.k,
            "tau": self.tau,
            "required_edge_count": self.required_edge_count,
            "release_domain_edge_count": self.release_domain_edge_count,
            "temporal_valid_edge_count": self.temporal_valid_edge_count,
            "missing_release_edge_count": len(self.missing_release_edges),
            "missing_temporal_edge_count": len(self.missing_temporal_edges),
            "missing_release_edges": json.dumps(
                _edge_rows(self.missing_release_edges),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "missing_temporal_edges": json.dumps(
                _edge_rows(self.missing_temporal_edges),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "feasible": self.feasible,
        }


@dataclass(frozen=True)
class EdgeNormalizationFactor:
    dataset: str
    edge: Edge
    raw_sample_count: int
    q_low_seconds: float
    q_high_seconds: float
    raw_spread_seconds: float
    normalization_factor_seconds: float
    zero_fallback_applied: bool

    def to_dict(self, *, selected_for_experiment: bool) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Inputs:
            selected_for_experiment: Input value used by this operation.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return {
            "source": self.edge[0],
            "target": self.edge[1],
            "raw_sample_count": self.raw_sample_count,
            "q05_seconds": self.q_low_seconds,
            "q95_seconds": self.q_high_seconds,
            "raw_spread_seconds": self.raw_spread_seconds,
            "normalization_factor_seconds": (
                self.normalization_factor_seconds
            ),
            "zero_fallback_applied": self.zero_fallback_applied,
            "selected_for_experiment": selected_for_experiment,
        }


@dataclass(frozen=True)
class DatasetParameterAnalysis:
    dataset: str
    source_sha256: str
    source_path: str | None
    profile: DatasetProfile
    variant_coverage: tuple[VariantCoverageRow, ...]
    edge_thresholds: tuple[EdgeThresholdRow, ...]
    consistency: tuple[VariantEdgeConsistencyRow, ...]
    normalization_factors: tuple[EdgeNormalizationFactor, ...]
    evaluation_log: EvaluationLogMetadata | None

    def coverage_for(self, k: int) -> VariantCoverageRow:
        """Return coverage for.

        Inputs:
            k: Input value used by this operation.
        Outputs:
            A value of type `VariantCoverageRow` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return next(row for row in self.variant_coverage if row.k == k)

    def threshold_for(self, tau: int) -> EdgeThresholdRow:
        """Return threshold for.

        Inputs:
            tau: Input value used by this operation.
        Outputs:
            A value of type `EdgeThresholdRow` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return next(row for row in self.edge_thresholds if row.tau == tau)

    def consistency_for(self, k: int, tau: int) -> VariantEdgeConsistencyRow:
        """Return consistency for.

        Inputs:
            k: Input value used by this operation.
            tau: Input value used by this operation.
        Outputs:
            A value of type `VariantEdgeConsistencyRow` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return next(
            row
            for row in self.consistency
            if row.k == k and row.tau == tau
        )


def _normalization_factors(
    dataset: str,
    samples: Mapping[Edge, Sequence[float]],
    policy: Phase7CPolicy,
) -> tuple[EdgeNormalizationFactor, ...]:
    """Return normalization factors.

    Inputs:
        dataset: Dataset identifier or dataset-specific input.
        samples: Input value used by this operation.
        policy: Input value used by this operation.
    Outputs:
        A value of type `tuple[EdgeNormalizationFactor, ...]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    factors: list[EdgeNormalizationFactor] = []
    for edge in sorted(samples):
        values = np.asarray(samples[edge], dtype=float)
        if values.size == 0:
            continue
        low = float(
            np.quantile(
                values,
                policy.normalization_low_quantile,
                method=policy.quantile_method,
            )
        )
        high = float(
            np.quantile(
                values,
                policy.normalization_high_quantile,
                method=policy.quantile_method,
            )
        )
        spread = high - low
        if spread < 0:
            raise RuntimeError(f"Negative quantile spread for edge {edge!r}.")
        zero_fallback = spread == 0.0
        effective = policy.zero_scale_fallback if zero_fallback else spread
        factors.append(
            EdgeNormalizationFactor(
                dataset=dataset,
                edge=edge,
                raw_sample_count=int(values.size),
                q_low_seconds=low,
                q_high_seconds=high,
                raw_spread_seconds=spread,
                normalization_factor_seconds=effective,
                zero_fallback_applied=zero_fallback,
            )
        )
    return tuple(factors)


def analyze_phase7c_dataset(
    dataset_input: Phase7CDatasetInput,
    policy: Phase7CPolicy = Phase7CPolicy(),
) -> DatasetParameterAnalysis:
    """Calculate one dataset's clipping, coverage, and normalization diagnostics.

    Input
    -----
    dataset_input:
        Explicit in-memory Phase 1 traces, source hash, and optional fixed
        evaluation log.  This function never reads an XES path itself.
    policy:
        Predeclared candidate sets, coverage diagnostics, reference settings,
        and quantile definitions.

    Output
    ------
    DatasetParameterAnalysis
        Dataset profile, every Top-k and tau candidate result, the complete
        k-by-tau Variant-edge feasibility matrix, and raw-reference nEMD
        normalization factors.  Raw trace/time samples are not serialized.

    Paper experiment step
    ---------------------
    Phase 7C Steps 1-5: derive Lmax/Tmax from adjacent case transitions, clip
    once, analyze Top-k and edge thresholds on G_clip, and precompute fixed
    edge-specific EMD normalization factors from G_raw.  No epsilon, noise,
    metric, or BUL result is inspected.
    """

    if not isinstance(dataset_input, Phase7CDatasetInput):
        raise TypeError("dataset_input must be a Phase7CDatasetInput.")
    if not isinstance(policy, Phase7CPolicy):
        raise TypeError("policy must be a Phase7CPolicy.")
    traces = tuple(dataset_input.traces)
    lengths = np.asarray([trace.length for trace in traces], dtype=float)
    l_quantile = float(
        np.quantile(lengths, policy.l_max_quantile, method=policy.quantile_method)
    )
    l_max = int(math.ceil(l_quantile))

    raw_temporal = temporal_statistics(traces)
    if raw_temporal.valid_interval_pairs == 0:
        raise ValueError(
            f"Dataset {dataset_input.dataset!r} has no valid transition times."
        )
    pooled_times = np.fromiter(
        (
            value
            for values in raw_temporal.samples.values()
            for value in values
        ),
        dtype=float,
        count=raw_temporal.valid_interval_pairs,
    )
    if np.any(pooled_times < 0):
        raise ValueError(
            f"Dataset {dataset_input.dataset!r} has negative transition times; "
            "Phase 7C never applies abs()."
        )
    t_max = float(
        np.quantile(
            pooled_times,
            policy.t_max_quantile,
            method=policy.quantile_method,
        )
    )
    if not math.isfinite(t_max) or t_max <= 0:
        raise ValueError(
            f"Dataset {dataset_input.dataset!r} produced a non-positive Tmax."
        )

    factors = _normalization_factors(
        dataset_input.dataset,
        raw_temporal.samples,
        policy,
    )
    raw_temporal_valid = {
        edge for edge, count in raw_temporal.sample_counts.items() if count > 0
    }
    valid_transition_samples = raw_temporal.valid_interval_pairs
    missing_transition_samples = raw_temporal.missing_interval_pairs
    zero_transition_samples = int(np.count_nonzero(pooled_times == 0))

    clipped_traces = clip_traces(traces, l_max=l_max, t_max=t_max)
    g_clip = build_g_raw(clipped_traces)
    ranked_variants = sorted(
        g_clip.variants.full_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    cumulative_cases = 0
    minimum_k_for_threshold = len(ranked_variants)
    for rank, (_, count) in enumerate(ranked_variants, start=1):
        cumulative_cases += count
        if (
            cumulative_cases / g_clip.variants.total_cases
            >= policy.variant_coverage_threshold
        ):
            minimum_k_for_threshold = rank
            break
    coverage_rows: list[VariantCoverageRow] = []
    selected_by_k: dict[int, tuple[Variant, ...]] = {}
    for k in policy.top_k_candidates:
        selected = tuple(variant for variant, _ in ranked_variants[:k])
        selected_count = sum(count for _, count in ranked_variants[:k])
        coverage = selected_count / g_clip.variants.total_cases
        selected_by_k[k] = selected
        coverage_rows.append(
            VariantCoverageRow(
                dataset=dataset_input.dataset,
                k=k,
                effective_k=len(selected),
                total_cases=g_clip.variants.total_cases,
                top_k_case_count=selected_count,
                coverage=coverage,
                other_count=g_clip.variants.total_cases - selected_count,
                other_mass=1.0 - coverage,
                meets_threshold=(
                    coverage >= policy.variant_coverage_threshold
                ),
                selected_variants=selected,
            )
        )

    edge_counts = g_clip.dfg.counts
    total_edge_types = len(edge_counts)
    total_occurrences = g_clip.dfg.total_edge_occurrences
    total_valid_times = g_clip.temporal.valid_interval_pairs
    threshold_rows: list[EdgeThresholdRow] = []
    for tau in policy.edge_tau_candidates:
        retained = tuple(
            sorted(edge for edge, count in edge_counts.items() if count >= tau)
        )
        retained_occurrences = sum(edge_counts[edge] for edge in retained)
        retained_valid_times = sum(
            g_clip.temporal.sample_counts[edge] for edge in retained
        )
        temporal_valid = tuple(
            edge
            for edge in retained
            if edge in raw_temporal_valid
            and g_clip.temporal.sample_counts[edge] > 0
        )
        type_retention = (
            len(retained) / total_edge_types if total_edge_types else 0.0
        )
        mass_retention = (
            retained_occurrences / total_occurrences
            if total_occurrences
            else 0.0
        )
        time_retention = (
            retained_valid_times / total_valid_times
            if total_valid_times
            else 0.0
        )
        temporal_ratio = (
            len(temporal_valid) / len(retained) if retained else 0.0
        )
        threshold_rows.append(
            EdgeThresholdRow(
                dataset=dataset_input.dataset,
                tau=tau,
                total_edge_types=total_edge_types,
                retained_edge_types=len(retained),
                edge_type_retention=type_retention,
                total_edge_occurrences=total_occurrences,
                retained_edge_occurrences=retained_occurrences,
                occurrence_mass_retention=mass_retention,
                total_valid_time_samples=total_valid_times,
                retained_valid_time_samples=retained_valid_times,
                valid_time_sample_retention=time_retention,
                retained_temporal_valid_edges=len(temporal_valid),
                temporal_valid_edge_ratio=temporal_ratio,
                meets_mass_threshold=(
                    mass_retention >= policy.edge_mass_threshold
                ),
                meets_time_threshold=(
                    time_retention >= policy.time_sample_threshold
                ),
                retained_edges=retained,
                temporal_valid_edges=temporal_valid,
            )
        )

    threshold_by_tau = {row.tau: row for row in threshold_rows}
    consistency_rows: list[VariantEdgeConsistencyRow] = []
    for k in policy.top_k_candidates:
        required = set(_variant_edges(selected_by_k[k]))
        for tau in policy.edge_tau_candidates:
            threshold = threshold_by_tau[tau]
            release_domain = set(threshold.retained_edges)
            temporal_domain = set(threshold.temporal_valid_edges)
            missing_release = tuple(sorted(required - release_domain))
            missing_temporal = tuple(sorted(required - temporal_domain))
            consistency_rows.append(
                VariantEdgeConsistencyRow(
                    dataset=dataset_input.dataset,
                    k=k,
                    tau=tau,
                    required_edge_count=len(required),
                    release_domain_edge_count=len(release_domain),
                    temporal_valid_edge_count=len(temporal_domain),
                    missing_release_edges=missing_release,
                    missing_temporal_edges=missing_temporal,
                    feasible=(not missing_release and not missing_temporal),
                )
            )

    all_activities = {
        activity for trace in traces for activity in trace.activities
    }
    raw_variants = {trace.activities for trace in traces}
    unavailable_timestamps = sum(
        value is None or pd.isna(value)
        for trace in traces
        for value in trace.timestamps
    )
    profile = DatasetProfile(
        dataset=dataset_input.dataset,
        case_count=len(traces),
        event_count=sum(trace.length for trace in traces),
        activity_count=len(all_activities),
        raw_variant_count=len(raw_variants),
        clipped_variant_count=g_clip.variants.unique_variants,
        minimum_k_for_variant_threshold=minimum_k_for_threshold,
        mean_trace_length=float(np.mean(lengths)),
        p95_trace_length=l_quantile,
        l_max=l_max,
        valid_transition_samples=valid_transition_samples,
        missing_transition_samples=missing_transition_samples,
        zero_transition_samples=zero_transition_samples,
        p95_transition_seconds=t_max,
        t_max_seconds=t_max,
        unavailable_timestamp_events=int(unavailable_timestamps),
    )

    evaluation = None
    if dataset_input.evaluation_events is not None:
        if dataset_input.evaluation_log_id is None:
            raise RuntimeError("Validated evaluation_log_id became unavailable.")
        evaluation = build_evaluation_log_metadata(
            dataset_input.evaluation_events,
            dataset_input.evaluation_log_id,
        )
    return DatasetParameterAnalysis(
        dataset=dataset_input.dataset,
        source_sha256=dataset_input.source_sha256,
        source_path=dataset_input.source_path,
        profile=profile,
        variant_coverage=tuple(coverage_rows),
        edge_thresholds=tuple(threshold_rows),
        consistency=tuple(consistency_rows),
        normalization_factors=factors,
        evaluation_log=evaluation,
    )


@dataclass(frozen=True)
class FrozenDatasetParameters:
    dataset: str
    source_sha256: str
    source_path: str | None
    l_max: int
    t_max: float
    top_k_variants: int
    min_edge_support: int
    normalization_factors: tuple[EdgeNormalizationFactor, ...]
    evaluation_log: EvaluationLogMetadata | None

    @property
    def normalization_scales(self) -> dict[Edge, float]:
        """Return normalization scales.

        Inputs:
            None.
        Outputs:
            A value of type `dict[Edge, float]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return {
            row.edge: row.normalization_factor_seconds
            for row in self.normalization_factors
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Inputs:
            None.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return {
            "dataset": self.dataset,
            "source_sha256": self.source_sha256,
            "source_path": self.source_path,
            "l_max": self.l_max,
            "t_max_seconds": self.t_max,
            "top_k_variants": self.top_k_variants,
            "min_edge_support": self.min_edge_support,
            "normalization_edge_count": len(self.normalization_factors),
            "normalization_reference": "edge_normalization.json",
            "evaluation_log": (
                None
                if self.evaluation_log is None
                else self.evaluation_log.to_dict()
            ),
        }


@dataclass(frozen=True)
class Phase7CSmokeEvidence:
    passed: bool
    epsilons: tuple[float, ...]
    seed: int
    budget_strategy: str
    dp_edge_mode: str
    experiment_rows: int
    dp_conditions: int
    process_quality_conditions: int
    artifact_sha256: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Inputs:
            None.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        data = asdict(self)
        data["epsilons"] = list(self.epsilons)
        return data


@dataclass(frozen=True)
class Phase7CPreflightResult:
    status: FreezeStatus
    policy: Phase7CPolicy
    analyses: tuple[DatasetParameterAnalysis, ...]
    frozen_parameters: tuple[FrozenDatasetParameters, ...]
    blocking_reasons: tuple[str, ...]
    process_quality: ProcessQualityFreezeSettings | None
    smoke_evidence: Phase7CSmokeEvidence | None = None

    def parameter_for(self, dataset: str) -> FrozenDatasetParameters:
        """Return parameter for.

        Inputs:
            dataset: Dataset identifier or dataset-specific input.
        Outputs:
            A value of type `FrozenDatasetParameters` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return next(
            row for row in self.frozen_parameters if row.dataset == dataset
        )

    def to_preflight_dict(self) -> dict[str, Any]:
        """Return the preflight state as a serializable dictionary.

        Inputs:
            None.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return {
            "name": "BUL_Phase7C_parameter_preflight",
            "phase": "7C",
            "status": self.status,
            "policy": self.policy.to_dict(),
            "blocking_reasons": list(self.blocking_reasons),
            "selected_parameters": [
                row.to_dict() for row in self.frozen_parameters
            ],
            "process_quality": (
                None
                if self.process_quality is None
                else self.process_quality.to_dict()
            ),
            "smoke_evidence": (
                None
                if self.smoke_evidence is None
                else self.smoke_evidence.to_dict()
            ),
            "runtime": _runtime_metadata(),
            "privacy_boundary": {
                "benchmark_calibration": (
                    "data-dependent clipping bounds and statistical domains "
                    "are frozen public-benchmark experiment inputs"
                ),
                "conditional_dp_claim": True,
                "private_deployment_requirement": (
                    "use public/predeclared domains and bounds or account for "
                    "their private selection in the privacy budget"
                ),
                "normalization_factors": (
                    "raw-reference evaluation-only values; not a DP release "
                    "and not a synthetic-generation input"
                ),
            },
            "raw_trace_rows_serialized": False,
            "raw_event_rows_serialized": False,
            "raw_time_samples_serialized": False,
        }


def select_phase7c_parameters(
    analyses: Sequence[DatasetParameterAnalysis],
    policy: Phase7CPolicy = Phase7CPolicy(),
    *,
    process_quality: ProcessQualityFreezeSettings | None = None,
) -> Phase7CPreflightResult:
    """Select predeclared candidates without inspecting any DP/BUL result.

    Input
    -----
    analyses:
        One completed dataset analysis for every experiment dataset.
    policy:
        Predeclared central reference settings, diagnostic coverage threshold,
        and hard edge/time-retention thresholds.
    process_quality:
        Optional explicit PM4Py settings.  Every selected dataset must already
        carry one complete evaluation-log fingerprint.

    Output
    ------
    Phase7CPreflightResult
        ``ready_for_smoke`` with selected parameters, or ``blocked`` with
        diagnostic reasons.  It is never marked frozen before smoke evidence.

    Paper experiment step
    ---------------------
    Phase 7C Steps 3-5: freeze the predeclared central Top-k/tau reference,
    retain Variant coverage and OTHER mass as diagnostics, enforce edge/time
    retention for every dataset, and enforce complete Top-k Variant-edge
    temporal consistency only for datasets designated for Synthetic Log/PM4Py
    validation.  No DP, BUL, or process-quality outcome is used for selection.
    """

    if not isinstance(policy, Phase7CPolicy):
        raise TypeError("policy must be a Phase7CPolicy.")
    checked = tuple(analyses)
    if not checked:
        raise ValueError("analyses must contain at least one dataset.")
    if any(not isinstance(row, DatasetParameterAnalysis) for row in checked):
        raise TypeError("analyses must contain DatasetParameterAnalysis values.")
    names = [row.dataset for row in checked]
    if len(set(names)) != len(names):
        raise ValueError("analyses must contain unique dataset names.")
    checked = tuple(sorted(checked, key=lambda row: row.dataset))
    reasons: list[str] = []
    selected_k = {
        row.dataset: policy.reference_top_k for row in checked
    }
    selected_tau = {
        row.dataset: policy.reference_edge_tau for row in checked
    }
    quality_datasets = (
        set() if process_quality is None else set(process_quality.datasets)
    )

    if process_quality is not None:
        unknown = quality_datasets - set(names)
        if unknown:
            reasons.append(
                "Process-quality datasets are missing from Phase 7C analyses: "
                f"{sorted(unknown)!r}."
            )
        for row in checked:
            if (
                row.dataset in process_quality.datasets
                and row.evaluation_log is None
            ):
                reasons.append(
                    f"Dataset {row.dataset!r} has no explicit complete PM4Py "
                    "evaluation log."
                )

    for row in checked:
        threshold = row.threshold_for(policy.reference_edge_tau)
        if not threshold.meets_mass_threshold:
            reasons.append(
                f"Dataset {row.dataset!r} reference tau="
                f"{policy.reference_edge_tau} retains occurrence mass "
                f"{threshold.occurrence_mass_retention:.6f}, below the hard "
                f"threshold {policy.edge_mass_threshold:.6f}."
            )
        if not threshold.meets_time_threshold:
            reasons.append(
                f"Dataset {row.dataset!r} reference tau="
                f"{policy.reference_edge_tau} retains valid-time samples "
                f"{threshold.valid_time_sample_retention:.6f}, below the hard "
                f"threshold {policy.time_sample_threshold:.6f}."
            )
        if threshold.retained_temporal_valid_edges <= 0:
            reasons.append(
                f"Dataset {row.dataset!r} reference tau="
                f"{policy.reference_edge_tau} has no valid temporal edge."
            )
        if row.dataset in quality_datasets:
            consistency = row.consistency_for(
                policy.reference_top_k,
                policy.reference_edge_tau,
            )
            if not consistency.feasible:
                reasons.append(
                    f"Dataset {row.dataset!r} fails required Synthetic Log "
                    "Variant-edge consistency at reference "
                    f"k={policy.reference_top_k}, tau="
                    f"{policy.reference_edge_tau}: missing release edges="
                    f"{len(consistency.missing_release_edges)}, missing "
                    f"temporal edges={len(consistency.missing_temporal_edges)}."
                )

    frozen: list[FrozenDatasetParameters] = []
    if not reasons and len(selected_tau) == len(checked):
        for row in checked:
            tau = selected_tau[row.dataset]
            temporal_edges = set(row.threshold_for(tau).temporal_valid_edges)
            selected_factors = tuple(
                factor
                for factor in row.normalization_factors
                if factor.edge in temporal_edges
            )
            if not selected_factors:
                reasons.append(
                    f"Dataset {row.dataset!r} has no normalization factors "
                    "in the selected temporal domain."
                )
                continue
            frozen.append(
                FrozenDatasetParameters(
                    dataset=row.dataset,
                    source_sha256=row.source_sha256,
                    source_path=row.source_path,
                    l_max=row.profile.l_max,
                    t_max=row.profile.t_max_seconds,
                    top_k_variants=selected_k[row.dataset],
                    min_edge_support=tau,
                    normalization_factors=selected_factors,
                    evaluation_log=row.evaluation_log,
                )
            )

    if reasons:
        frozen = []
    return Phase7CPreflightResult(
        status="blocked" if reasons else "ready_for_smoke",
        policy=policy,
        analyses=checked,
        frozen_parameters=tuple(frozen),
        blocking_reasons=tuple(reasons),
        process_quality=process_quality,
    )


def build_frozen_dataset_inputs(
    preflight: Phase7CPreflightResult,
    dataset_inputs: Sequence[Phase7CDatasetInput],
) -> tuple[DatasetExperimentInput, ...]:
    """Bridge selected Phase 7C parameters into the unchanged Phase 7B API.

    Input
    -----
    preflight:
        A ``ready_for_smoke`` or already ``frozen`` Phase 7C result.
    dataset_inputs:
        Original in-memory traces and explicit evaluation logs for the same
        datasets.  Their evaluation fingerprints are rechecked before use.

    Output
    ------
    tuple[DatasetExperimentInput, ...]
        Existing Phase 7B contracts populated only with preselected values.

    Paper experiment step
    ---------------------
    Phase 7C Step 6: pass the frozen clipping bounds, statistical domains,
    normalization factors, and unified evaluation logs to the already tested
    batch runner without modifying Phase 7A/7B.
    """

    if not isinstance(preflight, Phase7CPreflightResult):
        raise TypeError("preflight must be a Phase7CPreflightResult.")
    if preflight.status not in ("ready_for_smoke", "frozen"):
        raise ValueError("Blocked Phase 7C parameters cannot enter Phase 7B.")
    inputs = tuple(dataset_inputs)
    input_map = {row.dataset: row for row in inputs}
    if len(input_map) != len(inputs):
        raise ValueError("dataset_inputs must contain unique dataset names.")
    expected = {row.dataset for row in preflight.frozen_parameters}
    if set(input_map) != expected:
        raise ValueError(
            "dataset_inputs must exactly match the selected Phase 7C datasets."
        )
    quality_datasets = (
        set()
        if preflight.process_quality is None
        else set(preflight.process_quality.datasets)
    )
    bridged: list[DatasetExperimentInput] = []
    for parameters in preflight.frozen_parameters:
        source = input_map[parameters.dataset]
        if source.source_sha256 != parameters.source_sha256:
            raise FrozenConfigError(
                f"Source dataset hash changed after freeze for "
                f"{parameters.dataset!r}."
            )
        events = source.evaluation_events if parameters.dataset in quality_datasets else None
        log_id = source.evaluation_log_id if parameters.dataset in quality_datasets else None
        if parameters.dataset in quality_datasets and events is None:
            raise ValueError(
                f"Dataset {parameters.dataset!r} is missing its frozen "
                "evaluation_events."
            )
        if events is not None:
            if log_id is None or parameters.evaluation_log is None:
                raise RuntimeError("Frozen evaluation-log metadata is unavailable.")
            current = build_evaluation_log_metadata(events, log_id)
            if current != parameters.evaluation_log:
                raise FrozenConfigError(
                    f"Evaluation log changed after freeze for {parameters.dataset!r}."
                )
        bridged.append(
            DatasetExperimentInput(
                dataset=parameters.dataset,
                traces=source.traces,
                l_max=parameters.l_max,
                t_max=parameters.t_max,
                top_k_variants=parameters.top_k_variants,
                min_edge_support=parameters.min_edge_support,
                normalization_scales=parameters.normalization_scales,
                evaluation_events=events,
                evaluation_log_id=log_id,
            )
        )
    return tuple(bridged)


def run_phase7c_smoke_test(
    preflight: Phase7CPreflightResult,
    dataset_inputs: Sequence[Phase7CDatasetInput],
    output_dir: str | Path,
) -> tuple[BatchExperimentResult, Phase7CSmokeEvidence]:
    """Run the fixed low/high-epsilon smoke matrix and save its audit artifacts.

    Input
    -----
    preflight:
        A ``ready_for_smoke`` Phase 7C selection result.
    dataset_inputs:
        Matching traces and explicit evaluation logs.
    output_dir:
        Dedicated destination for Phase 7B-format smoke artifacts.

    Output
    ------
    tuple[BatchExperimentResult, Phase7CSmokeEvidence]
        Existing batch-run output and compact evidence required for final
        parameter freezing.

    Paper experiment step
    ---------------------
    Phase 7C Step 6: run epsilon 0.5 and 8, seed 0, uniform DP-full and main
    component DP-edge-only.  PM4Py is run only for DP-full epsilon 8 on the
    explicitly selected process-quality datasets.  This is a preflight, not a
    full experiment or a parameter-tuning objective.
    """

    if preflight.status != "ready_for_smoke":
        raise ValueError("Smoke testing requires ready_for_smoke parameters.")
    bridged = build_frozen_dataset_inputs(preflight, dataset_inputs)
    quality_selection = None
    if preflight.process_quality is not None:
        quality_selection = ProcessQualitySelection(
            datasets=preflight.process_quality.datasets,
            epsilons=(8.0,),
            seeds=(0,),
            budget_strategies=("uniform",),
            conformance_variant=preflight.process_quality.conformance_variant,
            noise_threshold=preflight.process_quality.noise_threshold,
        )
    config = BatchExperimentConfig(
        epsilons=(0.5, 8.0),
        seeds=(0,),
        dp_full_strategies=("uniform",),
        dp_edge_modes=("component",),
        process_quality_selection=quality_selection,
    )
    result = run_batch_experiments(bridged, config)
    artifacts = write_batch_experiment_artifacts(result, output_dir)
    destination = Path(output_dir)
    hashes = {
        key: _sha256_bytes((destination / filename).read_bytes())
        for key, filename in artifacts.items()
    }
    counts = result.to_dict()["counts"]
    evidence = Phase7CSmokeEvidence(
        passed=(counts["failed_conditions"] == 0),
        epsilons=(0.5, 8.0),
        seed=0,
        budget_strategy="uniform",
        dp_edge_mode="component",
        experiment_rows=counts["experiment_rows"],
        dp_conditions=counts["dp_conditions"],
        process_quality_conditions=counts["process_quality_conditions"],
        artifact_sha256=hashes,
    )
    return result, evidence


def finalize_phase7c_parameters(
    preflight: Phase7CPreflightResult,
    smoke_evidence: Phase7CSmokeEvidence,
) -> Phase7CPreflightResult:
    """Promote smoke-tested parameters to the immutable frozen state.

    Input
    -----
    preflight:
        A ``ready_for_smoke`` Phase 7C result.
    smoke_evidence:
        Evidence returned by ``run_phase7c_smoke_test``.

    Output
    ------
    Phase7CPreflightResult
        A ``frozen`` result eligible to write experiment_config_final.json.

    Paper experiment step
    ---------------------
    Phase 7C Step 7: freeze parameters only after both epsilon smoke paths and
    the requested PM4Py path have completed without a failed condition.
    """

    if preflight.status != "ready_for_smoke":
        raise ValueError("Only ready_for_smoke parameters can be finalized.")
    if not isinstance(smoke_evidence, Phase7CSmokeEvidence):
        raise TypeError("smoke_evidence must be Phase7CSmokeEvidence.")
    if not smoke_evidence.passed:
        return replace(
            preflight,
            status="blocked",
            blocking_reasons=("Phase 7C smoke test did not pass.",),
            frozen_parameters=(),
            smoke_evidence=smoke_evidence,
        )
    expected_quality = (
        0
        if preflight.process_quality is None
        else len(preflight.process_quality.datasets)
    )
    if smoke_evidence.process_quality_conditions != expected_quality:
        raise ValueError(
            "Smoke evidence does not contain the requested PM4Py conditions."
        )
    return replace(
        preflight,
        status="frozen",
        smoke_evidence=smoke_evidence,
    )


def _normalization_payload(
    result: Phase7CPreflightResult,
) -> dict[str, Any]:
    """Return normalization payload.

    Inputs:
        result: Experiment result processed by the operation.
    Outputs:
        A value of type `dict[str, Any]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    selected: dict[str, set[Edge]] = {
        row.dataset: {factor.edge for factor in row.normalization_factors}
        for row in result.frozen_parameters
    }
    return {
        "name": "edge_specific_EMD_normalization_factors",
        "phase": "7C",
        "status": result.status,
        "policy": {
            "low_quantile": result.policy.normalization_low_quantile,
            "high_quantile": result.policy.normalization_high_quantile,
            "quantile_method": result.policy.quantile_method,
            "unit": "seconds",
            "zero_scale_policy": "replace exact zero with 1 only",
            "normalization_only": True,
            "dp_release": False,
            "synthetic_generation_input": False,
            "raw_reference_used": True,
            "epsilon_independent": True,
        },
        "datasets": {
            analysis.dataset: [
                factor.to_dict(
                    selected_for_experiment=(
                        factor.edge in selected.get(analysis.dataset, set())
                    )
                )
                for factor in analysis.normalization_factors
            ]
            for analysis in result.analyses
        },
        "raw_time_samples_serialized": False,
    }


def _final_config_payload(
    result: Phase7CPreflightResult,
    artifact_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Return final config payload.

    Inputs:
        result: Experiment result processed by the operation.
        artifact_sha256: Input value used by this operation.
    Outputs:
        A value of type `dict[str, Any]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    if result.status != "frozen":
        raise ValueError("Final experiment config requires frozen status.")
    return {
        "name": "BUL_experiment_config_final",
        "phase": "7C",
        "status": "frozen",
        "policy": result.policy.to_dict(),
        "datasets": [row.to_dict() for row in result.frozen_parameters],
        "experiment_matrix": {
            "epsilon_values": list(EPS_LIST),
            "seeds": list(EXPERIMENT_SEEDS),
            "budget_strategies": list(BUDGET_STRATEGIES),
            "dp_edge_modes": list(DP_EDGE_MODES),
            "main_dp_edge_mode": "component",
            "appendix_dp_edge_mode": "mechanism",
            "time_budget_internal_split": dict(TEMPORAL_BUDGET_RATIOS),
        },
        "parameter_robustness": {
            "reference_top_k": result.policy.reference_top_k,
            "reference_edge_tau": result.policy.reference_edge_tau,
            "top_k_sensitivity": list(result.policy.top_k_candidates),
            "edge_tau_sensitivity": list(result.policy.edge_tau_candidates),
            "full_cross_product_required": True,
            "variant_coverage_role": "diagnostic_only",
            "other_mass_report_required": True,
        },
        "dp_edge_only_missing_dimensions": [
            "L_temporal",
            "L_information",
            "BUL",
            "fitness",
            "precision",
            "generalization",
        ],
        "synthetic_log_policy": {
            "variant_driven": True,
            "other_excluded": True,
            "interpretation": "Top-k core-behavior projection",
            "complete_variant_reconstruction_claim": False,
            "top_k_coverage_must_be_reported": True,
            "timestamp_start": "2026-01-01T00:00:00",
            "same_generation_rule_across_conditions": True,
            "same_synthetic_cases_across_conditions": False,
        },
        "process_quality": (
            None
            if result.process_quality is None
            else result.process_quality.to_dict()
        ),
        "smoke_evidence": (
            None
            if result.smoke_evidence is None
            else result.smoke_evidence.to_dict()
        ),
        "artifact_sha256": dict(sorted(artifact_sha256.items())),
        "runtime": _runtime_metadata(),
        "immutability_policy": (
            "an existing final config may be rewritten only with identical bytes"
        ),
        "privacy_boundary": result.to_preflight_dict()["privacy_boundary"],
    }


def write_phase7c_artifacts(
    result: Phase7CPreflightResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write deterministic diagnostics and, only when frozen, the final config.

    Input
    -----
    result:
        Blocked, ready-for-smoke, or fully frozen Phase 7C result.
    output_dir:
        Destination dedicated to one Phase 7C preflight/freeze attempt.

    Output
    ------
    dict[str, str]
        Logical artifact names.  The final configuration is absent unless the
        smoke-tested result has ``status='frozen'``.

    Paper experiment step
    ---------------------
    Phase 7C Steps 1-7: preserve dataset profiles, every candidate diagnostic,
    fixed normalization metadata, selection status, hashes, and smoke evidence.
    A blocked or untested analysis can never masquerade as a final experiment
    configuration.
    """

    if not isinstance(result, Phase7CPreflightResult):
        raise TypeError("result must be a Phase7CPreflightResult.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    final_path = destination / "experiment_config_final.json"
    if result.status != "frozen" and final_path.exists():
        raise FrozenConfigError(
            "A final frozen config already exists in this output directory; "
            "use a new directory for blocked or pre-smoke diagnostics."
        )

    profile_rows = [row.profile.to_row() for row in result.analyses]
    coverage_rows = []
    for row in result.analyses:
        for candidate in row.variant_coverage:
            payload = candidate.to_row()
            payload["selection_role"] = (
                "reference"
                if candidate.k == result.policy.reference_top_k
                else "sensitivity"
            )
            payload["coverage_role"] = "diagnostic_only"
            coverage_rows.append(payload)
    threshold_rows = []
    for row in result.analyses:
        for candidate in row.edge_thresholds:
            payload = candidate.to_row()
            payload["selection_role"] = (
                "reference"
                if candidate.tau == result.policy.reference_edge_tau
                else "sensitivity"
            )
            threshold_rows.append(payload)
    quality_datasets = (
        set()
        if result.process_quality is None
        else set(result.process_quality.datasets)
    )
    consistency_rows = []
    for row in result.analyses:
        required = row.dataset in quality_datasets
        for candidate in row.consistency:
            payload = candidate.to_row()
            payload["required_for_process_quality"] = required
            payload["selection_status"] = (
                "required_pass"
                if required and candidate.feasible
                else "required_fail"
                if required
                else "not_applicable"
            )
            consistency_rows.append(payload)
    payloads: dict[str, bytes] = {
        "dataset_profile.csv": _csv_bytes(
            profile_rows,
            tuple(DatasetProfile.__dataclass_fields__),
        ),
        "variant_coverage.csv": _csv_bytes(
            coverage_rows,
            (
                "dataset",
                "k",
                "effective_k",
                "total_cases",
                "top_k_case_count",
                "coverage",
                "other_count",
                "other_mass",
                "meets_threshold",
                "selection_role",
                "coverage_role",
            ),
        ),
        "edge_threshold_coverage.csv": _csv_bytes(
            threshold_rows,
            (
                "dataset",
                "tau",
                "total_edge_types",
                "retained_edge_types",
                "edge_type_retention",
                "total_edge_occurrences",
                "retained_edge_occurrences",
                "occurrence_mass_retention",
                "total_valid_time_samples",
                "retained_valid_time_samples",
                "valid_time_sample_retention",
                "retained_temporal_valid_edges",
                "temporal_valid_edge_ratio",
                "meets_mass_threshold",
                "meets_time_threshold",
                "selection_role",
            ),
        ),
        "variant_edge_consistency.csv": _csv_bytes(
            consistency_rows,
            (
                "dataset",
                "k",
                "tau",
                "required_edge_count",
                "release_domain_edge_count",
                "temporal_valid_edge_count",
                "missing_release_edge_count",
                "missing_temporal_edge_count",
                "missing_release_edges",
                "missing_temporal_edges",
                "feasible",
                "required_for_process_quality",
                "selection_status",
            ),
        ),
        "edge_normalization.json": _json_bytes(
            _normalization_payload(result)
        ),
    }
    diagnostic_hashes = {
        name: _sha256_bytes(payload) for name, payload in payloads.items()
    }
    preflight_payload = result.to_preflight_dict()
    preflight_payload["diagnostic_artifact_sha256"] = dict(
        sorted(diagnostic_hashes.items())
    )
    payloads["phase7c_preflight.json"] = _json_bytes(preflight_payload)

    config_bytes = None
    if result.status == "frozen":
        config_hash_inputs = {
            name: _sha256_bytes(payload) for name, payload in payloads.items()
        }
        config_bytes = _json_bytes(
            _final_config_payload(result, config_hash_inputs)
        )
        if final_path.exists() and final_path.read_bytes() != config_bytes:
            raise FrozenConfigError(
                "experiment_config_final.json already exists with different bytes."
            )
        payloads[final_path.name] = config_bytes

    artifact_hashes = {
        name: _sha256_bytes(payload) for name, payload in payloads.items()
    }
    manifest = {
        "name": "BUL_Phase7C_manifest",
        "phase": "7C",
        "status": result.status,
        "blocking_reasons": list(result.blocking_reasons),
        "artifacts": sorted(payloads),
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
        "source_sha256": {
            row.dataset: row.source_sha256 for row in result.analyses
        },
        "runtime": _runtime_metadata(),
        "raw_rows_serialized": False,
    }
    payloads["phase7c_manifest.json"] = _json_bytes(manifest)
    for filename, payload in payloads.items():
        _atomic_write_bytes(destination / filename, payload)
    return {
        Path(filename).stem: filename for filename in sorted(payloads)
    }


# Natural public API names for the release artifact. Historical Phase 7C names
# remain available for frozen provenance and downstream compatibility.
DatasetParameterInput = Phase7CDatasetInput
ParameterFreezePolicy = Phase7CPolicy
ParameterFreezeResult = Phase7CPreflightResult
ParameterSmokeEvidence = Phase7CSmokeEvidence
analyze_dataset_parameters = analyze_phase7c_dataset
finalize_experiment_parameters = finalize_phase7c_parameters
run_parameter_smoke_test = run_phase7c_smoke_test
select_experiment_parameters = select_phase7c_parameters
write_parameter_artifacts = write_phase7c_artifacts


__all__ = [
    "DatasetParameterInput",
    "DatasetParameterAnalysis",
    "EdgeNormalizationFactor",
    "EdgeThresholdRow",
    "EvaluationLogMetadata",
    "FrozenConfigError",
    "FrozenDatasetParameters",
    "ParameterFreezePolicy",
    "ParameterFreezeResult",
    "ParameterSmokeEvidence",
    "Phase7CDatasetInput",
    "Phase7CPolicy",
    "Phase7CPreflightResult",
    "Phase7CSmokeEvidence",
    "ProcessQualityFreezeSettings",
    "VariantCoverageRow",
    "VariantEdgeConsistencyRow",
    "analyze_dataset_parameters",
    "analyze_phase7c_dataset",
    "build_evaluation_log_metadata",
    "build_frozen_dataset_inputs",
    "finalize_experiment_parameters",
    "finalize_phase7c_parameters",
    "run_parameter_smoke_test",
    "run_phase7c_smoke_test",
    "select_experiment_parameters",
    "select_phase7c_parameters",
    "write_parameter_artifacts",
    "write_phase7c_artifacts",
]

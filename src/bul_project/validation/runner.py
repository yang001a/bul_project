"""Resumable formal runner for three representative PM4Py conditions.

The runner is deliberately separate from the completed numerical reference
matrix.  It authenticates and reuses committed DP releases, performs only DP
post-processing during synthetic-log generation, and supplies the frozen raw
benchmark evaluation sample solely to the external PM4Py evaluator.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
from contextlib import contextmanager
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pm4py

from bul_project.config import PROJECT_ROOT
from bul_project.experiments.parameter_freeze import build_evaluation_log_metadata
from bul_project.validation.process_model import (
    ProcessModelEvaluator,
    ProcessQualityResult,
)
from bul_project.validation.artifacts import (
    FileDigest,
    VerifiedArtifact,
    petri_net_semantic_sha256,
    sha256_file,
    synthetic_event_semantic_sha256,
    write_petri_pnml_atomic,
    write_process_quality_metadata_json_atomic,
    write_process_quality_result_csv_atomic,
    write_synthetic_xes_atomic,
)
from bul_project.validation.release import (
    ProcessQualityReleaseResult,
    generate_process_quality_release,
)
from bul_project.validation.scope import (
    EXTERNAL_VALIDITY_CLAIM_STRENGTH,
    EXTERNAL_VALIDITY_ROLE,
    SCOPE_VERSION,
    FrozenProcessQualityScope,
    select_preregistered_seed,
    validate_process_quality_scope,
)


_SCOPE_FILENAME = "process_quality_scope_freeze.json"
_PLAN_FILENAME = "process_quality_execution_plan.json"
_PROGRESS_FILENAME = "process_quality_progress.json"
_FINAL_MANIFEST = "process_quality_manifest.json"
_CHECKPOINT_MANIFEST = "checkpoint_manifest.json"
_RESULT_FILENAME = "process_quality_result.csv"
_METADATA_FILENAME = "process_quality_run_metadata.jsonl"

_CODE_FILES = (
    "src/bul_project/config.py",
    "src/bul_project/behavior/models.py",
    "src/bul_project/privacy/_validation.py",
    "src/bul_project/privacy/temporal_generator.py",
    "src/bul_project/validation/synthetic_log.py",
    "src/bul_project/validation/process_model.py",
    "src/bul_project/experiments/parameter_freeze.py",
    "src/bul_project/preprocessing/__init__.py",
    "src/bul_project/preprocessing/cleaning.py",
    "src/bul_project/preprocessing/dataset_report.py",
    "src/bul_project/preprocessing/models.py",
    "src/bul_project/preprocessing/pipeline.py",
    "src/bul_project/preprocessing/traces.py",
    "src/bul_project/preprocessing/xes_reader.py",
    "src/bul_project/validation/__init__.py",
    "src/bul_project/validation/evaluation_sample.py",
    "src/bul_project/validation/scope.py",
    "src/bul_project/validation/release.py",
    "src/bul_project/validation/artifacts.py",
    "src/bul_project/validation/runner.py",
    "scripts/run_external_validation.py",
)


class ProcessQualityFormalIntegrityError(RuntimeError):
    """Raised when an immutable source, checkpoint, or output changed."""


class ProcessQualityFormalConditionError(RuntimeError):
    """Raised after one condition fails without writing a completion marker."""


def _json_bytes(payload: object) -> bytes:
    """Serialize JSON content to deterministic UTF-8 bytes.

    Inputs:
        payload: Structured content to validate, serialize, or write.
    Outputs:
        A value of type `bytes` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    """Compute the SHA-256 digest of a byte sequence.

    Inputs:
        payload: Structured content to validate, serialize, or write.
    Outputs:
        A value of type `str` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    """Read and parse a JSON document.

    Inputs:
        path: Path to the input or output artifact.
        label: Input value used by this operation.
    Outputs:
        A value of type `dict[str, Any]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if not path.is_file():
        raise ProcessQualityFormalIntegrityError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProcessQualityFormalIntegrityError(
            f"{label} is not valid UTF-8 JSON: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise ProcessQualityFormalIntegrityError(f"{label} must be a JSON object.")
    return value


def _read_authenticated_json(
    path: Path,
    expected_sha256: Any,
    label: str,
) -> dict[str, Any]:
    """Hash and parse the same immutable byte snapshot."""

    expected = _require_sha256(expected_sha256, f"{label} SHA-256")
    if not path.is_file():
        raise ProcessQualityFormalIntegrityError(f"{label} is missing: {path}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProcessQualityFormalIntegrityError(
            f"Cannot read {label}: {path}"
        ) from exc
    if _sha256_bytes(payload) != expected:
        raise ProcessQualityFormalIntegrityError(f"{label} changed: {path}")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProcessQualityFormalIntegrityError(
            f"{label} is not valid UTF-8 JSON: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise ProcessQualityFormalIntegrityError(f"{label} must be a JSON object.")
    return value


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Write a byte sequence atomically.

    Inputs:
        path: Path to the input or output artifact.
        payload: Structured content to validate, serialize, or write.
    Outputs:
        None.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, payload: object) -> None:
    """Write JSON content atomically.

    Inputs:
        path: Path to the input or output artifact.
        payload: Structured content to validate, serialize, or write.
    Outputs:
        None.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    _write_bytes_atomic(path, _json_bytes(payload))


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    """Validate and return a mapping value.

    Inputs:
        value: Candidate value to validate or transform.
        label: Input value used by this operation.
    Outputs:
        A value of type `dict[str, Any]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if not isinstance(value, Mapping):
        raise ProcessQualityFormalIntegrityError(f"{label} must be a mapping.")
    return dict(value)


def _require_sequence(value: Any, label: str) -> list[Any]:
    """Validate and return a sequence value.

    Inputs:
        value: Candidate value to validate or transform.
        label: Input value used by this operation.
    Outputs:
        A value of type `list[Any]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProcessQualityFormalIntegrityError(f"{label} must be a sequence.")
    return list(value)


def _require_sha256(value: Any, label: str) -> str:
    """Validate and return a SHA-256 digest.

    Inputs:
        value: Candidate value to validate or transform.
        label: Input value used by this operation.
    Outputs:
        A value of type `str` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProcessQualityFormalIntegrityError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return value


def _unit_interval(value: Any, label: str) -> float:
    """Validate and return a finite value in the unit interval.

    Inputs:
        value: Candidate value to validate or transform.
        label: Input value used by this operation.
    Outputs:
        A value of type `float` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ProcessQualityFormalIntegrityError(
            f"{label} must be a finite value in [0,1]."
        )
    return float(value)


def _runtime_metadata() -> dict[str, str | None]:
    """Return runtime metadata.

    Inputs:
        None.
    Outputs:
        A value of type `dict[str, str | None]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    values: dict[str, str | None] = {
        "python": platform.python_version(),
    }
    for package in ("numpy", "pandas", "scipy", "pm4py", "lxml"):
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package] = None
    return values


def _code_manifest(project_root: Path) -> dict[str, str]:
    """Return code manifest.

    Inputs:
        project_root: Input value used by this operation.
    Outputs:
        A value of type `dict[str, str]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    hashes: dict[str, str] = {}
    for relative in _CODE_FILES:
        path = project_root / relative
        if not path.is_file():
            raise ProcessQualityFormalIntegrityError(
                f"Required process-quality code file is unavailable: {path}"
            )
        hashes[relative] = sha256_file(path)
    return hashes


def _validated_scope_payload(scope: FrozenProcessQualityScope) -> dict[str, Any]:
    """Return validated scope payload.

    Inputs:
        scope: Input value used by this operation.
    Outputs:
        A value of type `dict[str, Any]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if not isinstance(scope, FrozenProcessQualityScope):
        raise TypeError("scope must be a FrozenProcessQualityScope.")
    payload = scope.to_dict()
    declared = _require_sha256(payload.get("scope_sha256"), "scope_sha256")
    core = dict(payload)
    core.pop("scope_sha256", None)
    if _sha256_bytes(_json_bytes(core)) != declared:
        raise ProcessQualityFormalIntegrityError(
            "The process-quality scope differs from its declared SHA-256."
        )
    if (
        payload.get("name") != "BUL_formal_process_quality_scope"
        or payload.get("original_phase") != 6
        or payload.get("scope_version") != SCOPE_VERSION
        or payload.get("status") != "frozen"
    ):
        raise ProcessQualityFormalIntegrityError(
            "The process-quality scope identity changed."
        )
    source_bundle = _require_mapping(
        payload.get("source_bundle"), "source_bundle"
    )
    source_plan = _require_mapping(
        source_bundle.get("formal_execution_plan"),
        "source_bundle.formal_execution_plan",
    )
    source_plan_hash = _require_sha256(
        source_plan.get("internal_plan_sha256"),
        "source internal plan SHA-256",
    )
    expected_seed_selection = select_preregistered_seed(
        source_plan_hash
    ).to_dict()
    if payload.get("seed_selection") != expected_seed_selection:
        raise ProcessQualityFormalIntegrityError(
            "The preregistered seed-selection record changed."
        )

    conditions = _require_sequence(payload.get("conditions"), "scope.conditions")
    if len(conditions) != 3:
        raise ProcessQualityFormalIntegrityError(
            "Representative external validation must contain exactly 3 conditions."
        )
    epsilons = [
        float(_require_mapping(value, "condition")["epsilon"])
        for value in conditions
    ]
    if epsilons != [0.5, 2.0, 8.0]:
        raise ProcessQualityFormalIntegrityError(
            "Representative condition order must be epsilon 0.5, 2, and 8."
        )
    frozen = _require_mapping(payload.get("scope"), "scope.scope")
    if (
        frozen.get("dataset") != "BPIC2017"
        or frozen.get("method") != "DP-full"
        or frozen.get("budget_strategy") != "uniform"
        or frozen.get("seed") != 27
        or frozen.get("condition_count") != 3
        or frozen.get("top_k_variants") != 100
        or frozen.get("min_edge_support") != 10
        or _require_sha256(
            frozen.get("dataset_source_sha256"), "dataset source SHA-256"
        )
        != frozen.get("dataset_source_sha256")
    ):
        raise ProcessQualityFormalIntegrityError(
            "The representative process-quality scope changed."
        )
    for condition, epsilon in zip(conditions, (0.5, 2.0, 8.0)):
        row = _require_mapping(condition, "condition")
        epsilon_label = f"{epsilon:g}"
        formal_id = (
            "reference_k100_tau10|BPIC2017|DP-full|"
            f"epsilon={epsilon_label}|seed=27|mechanism=uniform"
        )
        if (
            row.get("formal_condition_id") != formal_id
            or row.get("checkpoint_key")
            != _sha256_bytes(formal_id.encode("utf-8"))
            or row.get("source_role") != "verified_DP_release_checkpoint"
        ):
            raise ProcessQualityFormalIntegrityError(
                "A representative source condition identity changed."
            )
        for field in (
            "checkpoint_manifest_sha256",
            "experiment_metadata_sha256",
            "experiment_result_sha256",
        ):
            _require_sha256(row.get(field), f"condition.{field}")

    expected_method = {
        "discovery_algorithm": "PM4Py Inductive Miner",
        "conformance_variant": "token_based",
        "inductive_miner_noise_threshold": 0.0,
        "metrics": ["fitness", "precision", "generalization"],
        "same_evaluation_log_for_all_conditions": True,
    }
    if payload.get("process_quality_method") != expected_method:
        raise ProcessQualityFormalIntegrityError(
            "The representative PM4Py method changed."
        )
    expected_synthetic_policy = {
        "input_boundary": "verified_DP_release_only",
        "raw_trace_access": False,
        "variant_driven": True,
        "other_excluded": True,
        "missing_temporal_edge_policy": "fail_and_record_missing_edge",
        "timestamp_start": "2026-01-01T00:00:00",
        "timestamp_direction": "nondecreasing_add_interval",
        "seed": 27,
    }
    if payload.get("synthetic_log_policy") != expected_synthetic_policy:
        raise ProcessQualityFormalIntegrityError(
            "The representative synthetic-log policy changed."
        )
    expected_boundary = {
        "included_core_tasks": [
            "main_BUL_numeric_matrix",
            "BUL_equal_and_fixed_reference_BUL_entropy",
            "dimension_ablation",
            "parameter_sensitivity",
            "cross_dataset_summary",
        ],
        "included_core_tasks_are_not_completion_claims": True,
        "external_validity_exception": {
            "dataset": "BPIC2017",
            "method": "DP-full",
            "budget_strategy": "uniform",
            "epsilon_values": [0.5, 2.0, 8.0],
            "seed": 27,
            "condition_count": 3,
            "role": EXTERNAL_VALIDITY_ROLE,
        },
        "removed_non_bul_tasks": [
            "full_PM4Py_condition_matrix",
            "all_budget_strategies_30_seeds_multi_dataset_PM4Py",
            "DP_edge_only_PM4Py",
            "3150_condition_synthetic_logs_and_Petri_nets",
            "PM4Py_inferential_statistics",
            "PM4Py_trend_or_correlation_analysis",
        ],
        "scope_semantics": (
            "Defines Phase 6 membership and exclusions; it does not mark "
            "an included but unfinished BUL task as complete."
        ),
    }
    if payload.get("phase6_scope_boundary") != expected_boundary:
        raise ProcessQualityFormalIntegrityError(
            "The reduced Phase 6 task boundary changed."
        )
    expected_exclusions = {
        "datasets": ["BPIC2012", "SEPSIS"],
        "methods": ["DP-edge-only"],
        "epsilon_values": [1.0, 4.0],
        "budget_strategies": [
            "activity_focus",
            "edge_focus",
            "variant_focus",
            "time_focus",
        ],
        "seeds": [seed for seed in range(30) if seed != 27],
        "metrics": ["simplicity"],
        "dp_edge_only_quality_fields": "NA",
        "other_variant_in_synthetic_generation": False,
    }
    if payload.get("explicit_exclusions") != expected_exclusions:
        raise ProcessQualityFormalIntegrityError(
            "The representative external-validity exclusions changed."
        )
    expected_claim = {
        "role": EXTERNAL_VALIDITY_ROLE,
        "claim_strength": EXTERNAL_VALIDITY_CLAIM_STRENGTH,
        "not_a_30_seed_estimate": True,
        "not_budget_strategy_robustness_evidence": True,
        "not_cross_dataset_evidence": True,
        "synthetic_log_is_DP_post_processing": True,
        "quality_metrics_are_evaluation_artifacts_not_DP_releases": True,
    }
    if payload.get("claim_boundary") != expected_claim:
        raise ProcessQualityFormalIntegrityError(
            "The representative external-validity claim boundary changed."
        )
    return payload


def validate_disjoint_directories(
    reference_dir: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Resolve and require non-overlapping read-only/input and output trees."""

    reference = Path(reference_dir).resolve()
    output = Path(output_dir).resolve()
    if (
        output == reference
        or output.is_relative_to(reference)
        or reference.is_relative_to(output)
    ):
        raise ProcessQualityFormalIntegrityError(
            "output_dir and the read-only reference_dir must be disjoint."
        )
    return reference, output


@contextmanager
def _exclusive_run_lock(path: Path):
    """Hold one OS-released non-blocking lock for the output directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ProcessQualityFormalIntegrityError(
                "Another process-quality execution already holds the output lock."
            ) from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


@dataclass(frozen=True)
class ProcessQualityExecutionPlan:
    """Immutable three-condition execution plan."""

    payload: Mapping[str, Any]

    @property
    def plan_sha256(self) -> str:
        """Build sha256.

        Inputs:
            None.
        Outputs:
            A value of type `str` produced by the operation.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        return _require_sha256(
            self.payload.get("plan_sha256"), "execution plan_sha256"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Inputs:
            None.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        return json.loads(json.dumps(dict(self.payload), allow_nan=False))


def build_process_quality_execution_plan(
    scope: FrozenProcessQualityScope,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> ProcessQualityExecutionPlan:
    """Build a deterministic plan without reading an XES evaluation log."""

    scope_payload = _validated_scope_payload(scope)
    root = Path(project_root).resolve()
    core = {
        "name": "BUL_representative_process_quality_execution_plan",
        "original_phase": 6,
        "status": "planned",
        "scope": "appendix_external_validity_representative3",
        "scope_file": _SCOPE_FILENAME,
        "scope_sha256": scope.scope_sha256,
        "conditions": scope_payload["conditions"],
        "counts": {"conditions": 3, "datasets": 1, "seeds": 1},
        "process_quality_method": scope_payload["process_quality_method"],
        "evaluation_log_source": scope_payload["evaluation_log_source"],
        "evaluation_log": scope_payload["evaluation_log"],
        "evaluation_sampling": scope_payload["evaluation_sampling"],
        "claim_boundary": scope_payload["claim_boundary"],
        "explicit_exclusions": scope_payload["explicit_exclusions"],
        "phase6_scope_boundary": scope_payload.get("phase6_scope_boundary"),
        "code_sha256": _code_manifest(root),
        "runtime": _runtime_metadata(),
        "artifact_contract": {
            "per_condition": [
                "synthetic_log.xes.gz",
                "petri_net.pnml",
                "process_quality_result.csv",
                "process_quality_metadata.json",
                _CHECKPOINT_MANIFEST,
            ],
            "final": [_RESULT_FILENAME, _METADATA_FILENAME, _FINAL_MANIFEST],
            "checkpoint_manifest_written_last": True,
        },
        "raw_trace_access_during_synthesis": False,
        "source_DP_release_reused_without_renoising": True,
    }
    payload = {**core, "plan_sha256": _sha256_bytes(_json_bytes(core))}
    return ProcessQualityExecutionPlan(payload=payload)


def write_process_quality_execution_plan(
    plan: ProcessQualityExecutionPlan,
    output_dir: str | Path,
) -> Path:
    """Write an immutable plan; reject an existing byte-different plan."""

    if not isinstance(plan, ProcessQualityExecutionPlan):
        raise TypeError("plan must be a ProcessQualityExecutionPlan.")
    destination = Path(output_dir).resolve() / _PLAN_FILENAME
    payload = _json_bytes(plan.to_dict())
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ProcessQualityFormalIntegrityError(
                "Existing process-quality execution plan differs and is immutable."
            )
        return destination
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        if destination.read_bytes() != payload:
            raise ProcessQualityFormalIntegrityError(
                "Concurrent process-quality plan differs."
            ) from exc
    return destination


@dataclass(frozen=True)
class ProcessQualityFormalSummary:
    """One invocation's resumable execution status."""

    status: str
    expected_conditions: int
    completed_conditions: int
    executed_this_invocation: int
    skipped_this_invocation: int
    failed_conditions: int
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Inputs:
            None.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        return {
            "original_phase": 6,
            "status": self.status,
            "expected_conditions": self.expected_conditions,
            "completed_conditions": self.completed_conditions,
            "executed_this_invocation": self.executed_this_invocation,
            "skipped_this_invocation": self.skipped_this_invocation,
            "failed_conditions": self.failed_conditions,
            "output_dir": self.output_dir,
        }


EvaluatorFactory = Callable[[], ProcessModelEvaluator]
EvaluationEventsLoader = Callable[[], pd.DataFrame]
ScopeSourceValidator = Callable[..., FrozenProcessQualityScope]


LOGGER = logging.getLogger(__name__)


class RepresentativeProcessQualityRunner:
    """Execute, checkpoint, and aggregate the exact three frozen conditions."""

    def __init__(
        self,
        scope: FrozenProcessQualityScope,
        reference_dir: str | Path,
        evaluation_events: pd.DataFrame | EvaluationEventsLoader,
        output_dir: str | Path,
        *,
        config_path: str | Path,
        evaluator_factory: EvaluatorFactory | None = None,
        project_root: str | Path = PROJECT_ROOT,
        _scope_source_validator: ScopeSourceValidator | None = None,
    ) -> None:
        """Initialize the instance and validate its construction inputs.

        Inputs:
            scope: Input value used by this operation.
            reference_dir: Input value used by this operation.
            evaluation_events: Input value used by this operation.
            output_dir: Input value used by this operation.
            config_path: Input value used by this operation.
            evaluator_factory: Input value used by this operation.
            project_root: Input value used by this operation.
            _scope_source_validator: Input value used by this operation.
        Outputs:
            None.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        self.reference_dir, self.output_dir = validate_disjoint_directories(
            reference_dir, output_dir
        )
        validator = _scope_source_validator or validate_process_quality_scope
        validated_scope = validator(
            scope,
            self.reference_dir / "formal_execution_plan.json",
            self.reference_dir / "formal_run_manifest.json",
            Path(config_path).resolve(),
            checkpoint_root=self.reference_dir / "checkpoints",
        )
        if (
            not isinstance(validated_scope, FrozenProcessQualityScope)
            or validated_scope.to_dict() != scope.to_dict()
        ):
            raise ProcessQualityFormalIntegrityError(
                "The Runner scope does not match its authenticated source bundle."
            )
        self.scope = validated_scope
        self.scope_payload = _validated_scope_payload(validated_scope)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scope_path = self.output_dir / _SCOPE_FILENAME
        self._validate_scope_file()
        self.checkpoint_root = self.output_dir / "checkpoints"
        self.plan = build_process_quality_execution_plan(
            scope, project_root=project_root
        )
        self.plan_path = write_process_quality_execution_plan(
            self.plan, self.output_dir
        )
        self.conditions = tuple(
            _require_mapping(value, "scope.conditions[]")
            for value in self.scope_payload["conditions"]
        )
        if isinstance(evaluation_events, pd.DataFrame):
            self.evaluation_events: pd.DataFrame | None = (
                self._validate_evaluation_events(evaluation_events)
            )
            self.evaluation_events_loader: EvaluationEventsLoader | None = None
        elif callable(evaluation_events):
            self.evaluation_events = None
            self.evaluation_events_loader = evaluation_events
        else:
            raise TypeError(
                "evaluation_events must be a DataFrame or a zero-argument loader."
            )
        method = _require_mapping(
            self.scope_payload["process_quality_method"],
            "process_quality_method",
        )
        self.evaluator_factory = evaluator_factory or (
            lambda: ProcessModelEvaluator(
                conformance_variant=method["conformance_variant"],
                noise_threshold=float(method["inductive_miner_noise_threshold"]),
                show_progress=False,
            )
        )

    def _validate_scope_file(self) -> None:
        """Validate scope file.

        Inputs:
            None.
        Outputs:
            None.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        expected = _json_bytes(self.scope.to_dict())
        if not self.scope_path.is_file():
            raise ProcessQualityFormalIntegrityError(
                "The immutable on-disk process-quality scope is missing."
            )
        if self.scope_path.read_bytes() != expected:
            raise ProcessQualityFormalIntegrityError(
                "The on-disk process-quality scope differs from the validated scope."
            )

    def _materialize_evaluation_events(self) -> None:
        """Materialize evaluation events.

        Inputs:
            None.
        Outputs:
            None.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        if self.evaluation_events is not None:
            return
        if self.evaluation_events_loader is None:
            raise ProcessQualityFormalIntegrityError(
                "The frozen evaluation-log loader is unavailable."
            )
        LOGGER.info("Loading and validating the frozen BPIC2017 evaluation log")
        events = self.evaluation_events_loader()
        self.evaluation_events = self._validate_evaluation_events(events)

    def _validate_evaluation_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """Validate evaluation events.

        Inputs:
            events: Input value used by this operation.
        Outputs:
            A value of type `pd.DataFrame` produced by the operation.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        expected = _require_mapping(
            self.scope_payload["evaluation_log"], "evaluation_log"
        )
        actual = build_evaluation_log_metadata(
            events, str(expected["evaluation_log_id"])
        ).to_dict()
        if actual != expected:
            raise ProcessQualityFormalIntegrityError(
                "The supplied external evaluation log differs from the frozen log."
            )
        return events

    def _source_checkpoint_dir(self, condition: Mapping[str, Any]) -> Path:
        """Return source checkpoint dir.

        Inputs:
            condition: Input value used by this operation.
        Outputs:
            A value of type `Path` produced by the operation.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        return self.reference_dir / "checkpoints" / str(condition["checkpoint_key"])

    def _load_source_metadata(
        self, condition: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Load source metadata.

        Inputs:
            condition: Input value used by this operation.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        checkpoint_dir = self._source_checkpoint_dir(condition)
        manifest_path = checkpoint_dir / _CHECKPOINT_MANIFEST
        manifest = _read_authenticated_json(
            manifest_path,
            condition["checkpoint_manifest_sha256"],
            "source checkpoint manifest",
        )
        source_plan_hash = self.scope_payload["source_bundle"][
            "formal_execution_plan"
        ]["internal_plan_sha256"]
        if (
            manifest.get("name") != "BUL_formal_condition_checkpoint"
            or manifest.get("status") != "complete"
            or manifest.get("formal_condition_id")
            != condition["formal_condition_id"]
            or manifest.get("execution_plan_sha256") != source_plan_hash
        ):
            raise ProcessQualityFormalIntegrityError(
                "Source numerical checkpoint identity changed."
            )
        artifacts = _require_mapping(manifest.get("artifacts"), "source artifacts")
        hashes = _require_mapping(
            manifest.get("artifact_sha256"), "source artifact hashes"
        )
        if artifacts != {
            "experiment_result": "experiment_result.csv",
            "experiment_metadata": "experiment_metadata.json",
        }:
            raise ProcessQualityFormalIntegrityError(
                "Source checkpoint artifact contract changed."
            )
        metadata_path = checkpoint_dir / artifacts["experiment_metadata"]
        result_path = checkpoint_dir / artifacts["experiment_result"]
        metadata_hash = condition["experiment_metadata_sha256"]
        result_hash = condition["experiment_result_sha256"]
        if (
            hashes.get(metadata_path.name) != metadata_hash
            or hashes.get(result_path.name) != result_hash
            or not result_path.is_file()
            or sha256_file(result_path) != result_hash
        ):
            raise ProcessQualityFormalIntegrityError(
                "Authenticated source artifact changed."
            )
        metadata = _read_authenticated_json(
            metadata_path,
            metadata_hash,
            "source experiment metadata",
        )
        record = _require_mapping(metadata.get("result_record"), "result_record")
        if (
            record.get("dataset") != "BPIC2017"
            or record.get("method") != "DP-full"
            or record.get("budget_strategy") != "uniform"
            or record.get("seed") != 27
            or float(record.get("epsilon", math.nan)) != float(condition["epsilon"])
            or record.get("process_quality_evaluated") is not False
            or any(
                record.get(name) is not None
                for name in ("fitness", "precision", "generalization")
            )
        ):
            raise ProcessQualityFormalIntegrityError(
                "Source result record is not the frozen numerical condition."
            )
        return metadata

    def _condition_output_dir(self, condition: Mapping[str, Any]) -> Path:
        """Create output dir.

        Inputs:
            condition: Input value used by this operation.
        Outputs:
            A value of type `Path` produced by the operation.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        return self.checkpoint_root / str(condition["checkpoint_key"])

    def _condition_record(
        self,
        condition: Mapping[str, Any],
        source_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Create record.

        Inputs:
            condition: Input value used by this operation.
            source_metadata: Input value used by this operation.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        source = _require_mapping(source_metadata.get("result_record"), "result_record")
        return {
            "scenario_id": "reference_k100_tau10",
            "formal_condition_id": condition["formal_condition_id"],
            "dataset": "BPIC2017",
            "epsilon": float(condition["epsilon"]),
            "method": "DP-full",
            "seed": 27,
            "budget_strategy": "uniform",
            "top_k_variants": 100,
            "min_edge_support": 10,
            "L_structure": _unit_interval(source["L_structure"], "L_structure"),
            "L_temporal": _unit_interval(source["L_temporal"], "L_temporal"),
            "L_information": _unit_interval(
                source["L_information"], "L_information"
            ),
            "BUL_equal": _unit_interval(source["BUL"], "BUL"),
            "process_quality_in_BUL": False,
            "external_validation_role": EXTERNAL_VALIDITY_ROLE,
            "external_validation_claim_strength": (
                EXTERNAL_VALIDITY_CLAIM_STRENGTH
            ),
        }

    def _write_condition_manifest(
        self,
        condition: Mapping[str, Any],
        artifacts: Mapping[str, str],
        artifact_hashes: Mapping[str, str],
        xes: VerifiedArtifact,
        pnml: VerifiedArtifact,
    ) -> Path:
        """Write condition manifest.

        Inputs:
            condition: Input value used by this operation.
            artifacts: Input value used by this operation.
            artifact_hashes: Input value used by this operation.
            xes: Input value used by this operation.
            pnml: Input value used by this operation.
        Outputs:
            A value of type `Path` produced by the operation.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        directory = self._condition_output_dir(condition)
        payload = {
            "name": "BUL_representative_process_quality_checkpoint",
            "original_phase": 6,
            "status": "complete",
            "scope_sha256": self.scope.scope_sha256,
            "execution_plan_sha256": self.plan.plan_sha256,
            "formal_condition_id": condition["formal_condition_id"],
            "condition": dict(condition),
            "source_checkpoint": {
                "checkpoint_manifest_sha256": condition[
                    "checkpoint_manifest_sha256"
                ],
                "experiment_metadata_sha256": condition[
                    "experiment_metadata_sha256"
                ],
                "DP_release_reused_without_renoising": True,
            },
            "evaluation_log_source": self.scope_payload["evaluation_log_source"],
            "evaluation_log": self.scope_payload["evaluation_log"],
            "evaluation_sampling": self.scope_payload["evaluation_sampling"],
            "artifacts": dict(artifacts),
            "artifact_sha256": dict(artifact_hashes),
            "artifact_semantic_sha256": {
                xes.filename: xes.semantic_sha256,
                pnml.filename: pnml.semantic_sha256,
            },
            "synthetic_generation_raw_trace_access": False,
            "process_quality_in_BUL": False,
            "claim_role": EXTERNAL_VALIDITY_ROLE,
            "claim_strength": EXTERNAL_VALIDITY_CLAIM_STRENGTH,
        }
        path = directory / _CHECKPOINT_MANIFEST
        _write_json_atomic(path, payload)
        return path

    def _execute_condition(self, condition: Mapping[str, Any]) -> None:
        """Execute condition.

        Inputs:
            condition: Input value used by this operation.
        Outputs:
            None.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        directory = self._condition_output_dir(condition)
        directory.mkdir(parents=True, exist_ok=True)
        source_metadata = self._load_source_metadata(condition)
        frozen = _require_mapping(self.scope_payload["scope"], "scope.scope")
        release_result: ProcessQualityReleaseResult = (
            generate_process_quality_release(
                source_metadata,
                expected_epsilon=float(condition["epsilon"]),
                expected_seed=int(frozen["seed"]),
                frozen_t_max=float(frozen["t_max_seconds"]),
            )
        )
        LOGGER.info(
            "Starting PM4Py evaluation: epsilon=%s, seed=27, strategy=uniform",
            condition["epsilon"],
        )
        evaluator = self.evaluator_factory()
        if not isinstance(evaluator, ProcessModelEvaluator):
            raise TypeError("evaluator_factory must return ProcessModelEvaluator.")
        evaluation = _require_mapping(
            self.scope_payload["evaluation_log"], "evaluation_log"
        )
        if self.evaluation_events is None:
            raise ProcessQualityFormalIntegrityError(
                "Evaluation events were not materialized under the run lock."
            )
        quality: ProcessQualityResult = evaluator.evaluate(
            release_result.synthetic_log,
            self.evaluation_events,
            evaluation_log_id=str(evaluation["evaluation_log_id"]),
        )
        LOGGER.info(
            "PM4Py evaluation completed: epsilon=%s, fitness=%.6f, "
            "precision=%.6f, generalization=%.6f",
            condition["epsilon"],
            quality.fitness,
            quality.precision,
            quality.generalization,
        )
        if quality.metadata.get("evaluation_log_sha256") != evaluation[
            "evaluation_log_sha256"
        ]:
            raise ProcessQualityFormalIntegrityError(
                "PM4Py evaluator used a different evaluation log."
            )
        for name, value in (
            ("fitness", quality.fitness),
            ("precision", quality.precision),
            ("generalization", quality.generalization),
        ):
            _unit_interval(value, name)

        xes = write_synthetic_xes_atomic(
            release_result.synthetic_log,
            directory / "synthetic_log.xes.gz",
            overwrite=True,
        )
        pnml = write_petri_pnml_atomic(
            quality.model,
            directory / "petri_net.pnml",
            overwrite=True,
        )
        record = self._condition_record(condition, source_metadata)
        result_digest: FileDigest = write_process_quality_result_csv_atomic(
            record,
            quality,
            xes,
            pnml,
            directory / "process_quality_result.csv",
            overwrite=True,
        )
        metadata_digest: FileDigest = write_process_quality_metadata_json_atomic(
            record,
            release_result.synthetic_log,
            quality,
            xes,
            pnml,
            directory / "process_quality_metadata.json",
            temporal_metadata={
                "DP_release_bridge": release_result.metadata,
                "temporal_reconstruction": release_result.temporal_distributions.metadata,
            },
            overwrite=True,
        )
        artifacts = {
            "synthetic_log": xes.filename,
            "petri_net": pnml.filename,
            "process_quality_result": result_digest.filename,
            "process_quality_metadata": metadata_digest.filename,
        }
        artifact_hashes = {
            xes.filename: xes.sha256,
            pnml.filename: pnml.sha256,
            result_digest.filename: result_digest.sha256,
            metadata_digest.filename: metadata_digest.sha256,
        }
        failure = directory / "failure.json"
        if failure.exists():
            failure.unlink()
        self._write_condition_manifest(
            condition, artifacts, artifact_hashes, xes, pnml
        )

    def _load_checkpoint(
        self, condition: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Load checkpoint.

        Inputs:
            condition: Input value used by this operation.
        Outputs:
            A value of type `dict[str, Any] | None` produced by the operation.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        directory = self._condition_output_dir(condition)
        manifest_path = directory / _CHECKPOINT_MANIFEST
        if not manifest_path.exists():
            return None
        manifest = _read_json(manifest_path, "process-quality checkpoint manifest")
        if (
            manifest.get("name")
            != "BUL_representative_process_quality_checkpoint"
            or manifest.get("original_phase") != 6
            or manifest.get("status") != "complete"
            or manifest.get("scope_sha256") != self.scope.scope_sha256
            or manifest.get("execution_plan_sha256") != self.plan.plan_sha256
            or manifest.get("formal_condition_id")
            != condition["formal_condition_id"]
            or manifest.get("condition") != dict(condition)
            or manifest.get("evaluation_log")
            != self.scope_payload["evaluation_log"]
            or manifest.get("evaluation_log_source")
            != self.scope_payload["evaluation_log_source"]
            or manifest.get("evaluation_sampling")
            != self.scope_payload["evaluation_sampling"]
            or manifest.get("source_checkpoint")
            != {
                "checkpoint_manifest_sha256": condition[
                    "checkpoint_manifest_sha256"
                ],
                "experiment_metadata_sha256": condition[
                    "experiment_metadata_sha256"
                ],
                "DP_release_reused_without_renoising": True,
            }
            or manifest.get("synthetic_generation_raw_trace_access") is not False
            or manifest.get("process_quality_in_BUL") is not False
            or manifest.get("claim_role") != EXTERNAL_VALIDITY_ROLE
            or manifest.get("claim_strength")
            != EXTERNAL_VALIDITY_CLAIM_STRENGTH
        ):
            raise ProcessQualityFormalIntegrityError(
                "Committed process-quality checkpoint identity changed."
            )
        artifacts = _require_mapping(manifest.get("artifacts"), "artifacts")
        expected_artifacts = {
            "synthetic_log": "synthetic_log.xes.gz",
            "petri_net": "petri_net.pnml",
            "process_quality_result": "process_quality_result.csv",
            "process_quality_metadata": "process_quality_metadata.json",
        }
        if artifacts != expected_artifacts:
            raise ProcessQualityFormalIntegrityError(
                "Committed process-quality artifact contract changed."
            )
        hashes = _require_mapping(manifest.get("artifact_sha256"), "artifact hashes")
        if set(hashes) != set(expected_artifacts.values()):
            raise ProcessQualityFormalIntegrityError(
                "Committed process-quality artifact hash set changed."
            )
        for filename, expected_raw in hashes.items():
            expected = _require_sha256(expected_raw, filename)
            path = directory / filename
            if not path.is_file() or sha256_file(path) != expected:
                raise ProcessQualityFormalIntegrityError(
                    f"Committed process-quality artifact changed: {path}"
                )
        semantic_hashes = _require_mapping(
            manifest.get("artifact_semantic_sha256"),
            "artifact semantic hashes",
        )
        if set(semantic_hashes) != {"synthetic_log.xes.gz", "petri_net.pnml"}:
            raise ProcessQualityFormalIntegrityError(
                "Committed semantic-artifact hash set changed."
            )
        expected_xes_semantic = _require_sha256(
            semantic_hashes["synthetic_log.xes.gz"],
            "synthetic XES semantic SHA-256",
        )
        expected_pnml_semantic = _require_sha256(
            semantic_hashes["petri_net.pnml"],
            "Petri PNML semantic SHA-256",
        )
        try:
            imported_xes = pm4py.read_xes(
                str(directory / "synthetic_log.xes.gz"),
                return_legacy_log_object=False,
                encoding="utf-8",
                show_progress_bar=False,
            )
            if not isinstance(imported_xes, pd.DataFrame):
                raise TypeError("PM4Py XES import did not return a DataFrame.")
            xes_events = imported_xes.loc[
                :, ["case:concept:name", "concept:name", "time:timestamp"]
            ].rename(
                columns={
                    "case:concept:name": "case_id",
                    "concept:name": "activity",
                    "time:timestamp": "timestamp",
                }
            )
            actual_xes_semantic = synthetic_event_semantic_sha256(
                xes_events, timestamp_precision="microsecond"
            )
            net, initial_marking, final_marking = pm4py.read_pnml(
                str(directory / "petri_net.pnml"), encoding="utf-8"
            )
            actual_pnml_semantic = petri_net_semantic_sha256(
                net, initial_marking, final_marking
            )
        except Exception as exc:
            raise ProcessQualityFormalIntegrityError(
                "Committed XES/PNML artifact cannot be semantically revalidated."
            ) from exc
        if (
            actual_xes_semantic != expected_xes_semantic
            or actual_pnml_semantic != expected_pnml_semantic
        ):
            raise ProcessQualityFormalIntegrityError(
                "Committed XES/PNML semantic digest changed."
            )
        condition_metadata = _read_authenticated_json(
            directory / "process_quality_metadata.json",
            hashes["process_quality_metadata.json"],
            "process-quality condition metadata",
        )
        metadata_artifacts = _require_mapping(
            condition_metadata.get("artifacts"), "metadata.artifacts"
        )
        if (
            _require_mapping(
                metadata_artifacts.get("synthetic_xes"),
                "metadata.synthetic_xes",
            ).get("semantic_sha256")
            != expected_xes_semantic
            or _require_mapping(
                metadata_artifacts.get("petri_pnml"),
                "metadata.petri_pnml",
            ).get("semantic_sha256")
            != expected_pnml_semantic
        ):
            raise ProcessQualityFormalIntegrityError(
                "Condition metadata and semantic artifact hashes differ."
            )
        rows = self._read_result_rows(directory / "process_quality_result.csv")
        if len(rows) != 1:
            raise ProcessQualityFormalIntegrityError(
                "A process-quality checkpoint must contain exactly one row."
            )
        row = rows[0]
        if (
            row.get("formal_condition_id") != condition["formal_condition_id"]
            or float(row.get("epsilon", "nan")) != float(condition["epsilon"])
            or int(row.get("seed", "-1")) != 27
        ):
            raise ProcessQualityFormalIntegrityError(
                "Committed process-quality row identity changed."
            )
        for field in ("fitness", "precision", "generalization"):
            try:
                _unit_interval(float(row[field]), field)
            except (KeyError, ValueError) as exc:
                raise ProcessQualityFormalIntegrityError(
                    f"Committed {field} is invalid."
                ) from exc
        return manifest

    @staticmethod
    def _read_result_rows(path: Path) -> list[dict[str, str]]:
        """Read result rows.

        Inputs:
            path: Path to the input or output artifact.
        Outputs:
            A value of type `list[dict[str, str]]` produced by the operation.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
        except OSError as exc:
            raise ProcessQualityFormalIntegrityError(
                f"Cannot read process-quality result: {path}"
            ) from exc

    def _write_failure(
        self, condition: Mapping[str, Any], exc: Exception
    ) -> None:
        """Write failure.

        Inputs:
            condition: Input value used by this operation.
            exc: Input value used by this operation.
        Outputs:
            None.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        directory = self._condition_output_dir(condition)
        directory.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(
            directory / "failure.json",
            {
                "name": "BUL_representative_process_quality_failure",
                "original_phase": 6,
                "status": "failed",
                "scope_sha256": self.scope.scope_sha256,
                "execution_plan_sha256": self.plan.plan_sha256,
                "formal_condition_id": condition["formal_condition_id"],
                "condition": dict(condition),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "completion_marker_written": False,
            },
        )

    def _write_progress(
        self,
        *,
        status: str,
        completed: int,
        executed: int,
        skipped: int,
        failed: int,
    ) -> None:
        """Write progress.

        Inputs:
            status: Input value used by this operation.
            completed: Input value used by this operation.
            executed: Input value used by this operation.
            skipped: Input value used by this operation.
            failed: Input value used by this operation.
        Outputs:
            None.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        _write_json_atomic(
            self.output_dir / _PROGRESS_FILENAME,
            {
                "name": "BUL_representative_process_quality_progress",
                "original_phase": 6,
                "status": status,
                "scope_sha256": self.scope.scope_sha256,
                "execution_plan_sha256": self.plan.plan_sha256,
                "expected_conditions": 3,
                "completed_conditions": completed,
                "remaining_conditions": 3 - completed,
                "executed_this_invocation": executed,
                "skipped_this_invocation": skipped,
                "failed_this_invocation": failed,
                "authoritative_completion_marker": _FINAL_MANIFEST,
            },
        )

    def _aggregate(
        self,
        checkpoints: Mapping[str, Mapping[str, Any]],
        *,
        executed: int,
        skipped: int,
    ) -> None:
        """Aggregate.

        Inputs:
            checkpoints: Input value used by this operation.
            executed: Input value used by this operation.
            skipped: Input value used by this operation.
        Outputs:
            None.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        self._validate_scope_file()
        rows: list[dict[str, str]] = []
        metadata_rows: list[dict[str, Any]] = []
        checkpoint_hashes: dict[str, str] = {}
        for condition in self.conditions:
            directory = self._condition_output_dir(condition)
            rows.extend(self._read_result_rows(directory / "process_quality_result.csv"))
            metadata_rows.append(
                _read_json(
                    directory / "process_quality_metadata.json",
                    "process-quality metadata",
                )
            )
            checkpoint_hashes[str(condition["formal_condition_id"])] = sha256_file(
                directory / _CHECKPOINT_MANIFEST
            )
        if len(rows) != 3 or len(metadata_rows) != 3:
            raise ProcessQualityFormalIntegrityError(
                "Representative process-quality aggregation must contain 3 rows."
            )
        fieldnames = list(rows[0])
        if any(list(row) != fieldnames for row in rows):
            raise ProcessQualityFormalIntegrityError(
                "Process-quality result columns changed across conditions."
            )
        import io

        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        _write_bytes_atomic(
            self.output_dir / _RESULT_FILENAME,
            stream.getvalue().encode("utf-8"),
        )
        metadata_payload = b"".join(
            (
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            for value in metadata_rows
        )
        _write_bytes_atomic(
            self.output_dir / _METADATA_FILENAME, metadata_payload
        )
        artifact_hashes = {
            _SCOPE_FILENAME: sha256_file(self.output_dir / _SCOPE_FILENAME),
            _PLAN_FILENAME: sha256_file(self.plan_path),
            _RESULT_FILENAME: sha256_file(self.output_dir / _RESULT_FILENAME),
            _METADATA_FILENAME: sha256_file(self.output_dir / _METADATA_FILENAME),
        }
        manifest = {
            "name": "BUL_representative_process_quality_run",
            "original_phase": 6,
            "status": "complete",
            "scope": "appendix_external_validity_representative3",
            "scope_sha256": self.scope.scope_sha256,
            "execution_plan_sha256": self.plan.plan_sha256,
            "counts": {
                "expected_conditions": 3,
                "completed_conditions": 3,
                "failed_conditions": 0,
                "executed_finalizing_invocation": executed,
                "skipped_finalizing_invocation": skipped,
            },
            "artifacts": {
                "scope_freeze": _SCOPE_FILENAME,
                "execution_plan": _PLAN_FILENAME,
                "process_quality_result": _RESULT_FILENAME,
                "process_quality_run_metadata": _METADATA_FILENAME,
                "manifest": _FINAL_MANIFEST,
            },
            "artifact_sha256": artifact_hashes,
            "checkpoint_manifest_sha256": checkpoint_hashes,
            "evaluation_log_source": self.scope_payload["evaluation_log_source"],
            "evaluation_log": self.scope_payload["evaluation_log"],
            "evaluation_sampling": self.scope_payload["evaluation_sampling"],
            "claim_boundary": self.scope_payload["claim_boundary"],
            "phase6_scope_boundary": self.scope_payload.get(
                "phase6_scope_boundary"
            ),
            "process_quality_in_BUL": False,
            "raw_trace_access_during_synthesis": False,
            "source_DP_release_reused_without_renoising": True,
            "not_original_phase6_completion_marker": True,
        }
        _write_json_atomic(self.output_dir / _FINAL_MANIFEST, manifest)

    def _validate_final_manifest(self) -> None:
        """Validate final manifest.

        Inputs:
            None.
        Outputs:
            None.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        path = self.output_dir / _FINAL_MANIFEST
        manifest = _read_json(path, "process-quality final manifest")
        if (
            manifest.get("name") != "BUL_representative_process_quality_run"
            or manifest.get("original_phase") != 6
            or manifest.get("status") != "complete"
            or manifest.get("scope")
            != "appendix_external_validity_representative3"
            or manifest.get("scope_sha256") != self.scope.scope_sha256
            or manifest.get("execution_plan_sha256") != self.plan.plan_sha256
            or manifest.get("evaluation_log")
            != self.scope_payload["evaluation_log"]
            or manifest.get("evaluation_log_source")
            != self.scope_payload["evaluation_log_source"]
            or manifest.get("evaluation_sampling")
            != self.scope_payload["evaluation_sampling"]
            or manifest.get("process_quality_in_BUL") is not False
            or manifest.get("raw_trace_access_during_synthesis") is not False
            or manifest.get("source_DP_release_reused_without_renoising")
            is not True
            or manifest.get("claim_boundary")
            != self.scope_payload["claim_boundary"]
            or manifest.get("phase6_scope_boundary")
            != self.scope_payload["phase6_scope_boundary"]
            or manifest.get("not_original_phase6_completion_marker") is not True
        ):
            raise ProcessQualityFormalIntegrityError(
                "Representative process-quality final manifest changed."
            )
        expected_artifacts = {
            "scope_freeze": _SCOPE_FILENAME,
            "execution_plan": _PLAN_FILENAME,
            "process_quality_result": _RESULT_FILENAME,
            "process_quality_run_metadata": _METADATA_FILENAME,
            "manifest": _FINAL_MANIFEST,
        }
        if manifest.get("artifacts") != expected_artifacts:
            raise ProcessQualityFormalIntegrityError(
                "Representative final artifact mapping changed."
            )
        counts = _require_mapping(manifest.get("counts"), "final counts")
        if (
            counts.get("expected_conditions") != 3
            or counts.get("completed_conditions") != 3
            or counts.get("failed_conditions") != 0
            or int(counts.get("executed_finalizing_invocation", -1))
            + int(counts.get("skipped_finalizing_invocation", -1))
            != 3
        ):
            raise ProcessQualityFormalIntegrityError(
                "Representative process-quality completion counts changed."
            )
        hashes = _require_mapping(
            manifest.get("artifact_sha256"), "final artifact hashes"
        )
        expected_files = {
            _SCOPE_FILENAME,
            _PLAN_FILENAME,
            _RESULT_FILENAME,
            _METADATA_FILENAME,
        }
        if set(hashes) != expected_files:
            raise ProcessQualityFormalIntegrityError(
                "Representative final artifact set changed."
            )
        for filename, expected_raw in hashes.items():
            expected = _require_sha256(expected_raw, filename)
            path = self.output_dir / filename
            if not path.is_file() or sha256_file(path) != expected:
                raise ProcessQualityFormalIntegrityError(
                    f"Representative final artifact changed: {filename}"
                )
        checkpoint_hashes = _require_mapping(
            manifest.get("checkpoint_manifest_sha256"),
            "checkpoint manifest hashes",
        )
        expected_ids = {
            str(condition["formal_condition_id"]) for condition in self.conditions
        }
        if set(checkpoint_hashes) != expected_ids:
            raise ProcessQualityFormalIntegrityError(
                "Representative checkpoint set changed."
            )
        for condition in self.conditions:
            expected = _require_sha256(
                checkpoint_hashes[str(condition["formal_condition_id"])],
                "checkpoint manifest hash",
            )
            path = self._condition_output_dir(condition) / _CHECKPOINT_MANIFEST
            if not path.is_file() or sha256_file(path) != expected:
                raise ProcessQualityFormalIntegrityError(
                    "Representative checkpoint manifest changed."
                )

    def run(
        self, *, max_new_conditions: int | None = None
    ) -> ProcessQualityFormalSummary:
        """Execute under one output-level OS lock."""

        with _exclusive_run_lock(
            self.output_dir / ".process_quality_run.lock"
        ):
            return self._run_unlocked(
                max_new_conditions=max_new_conditions
            )

    def _run_unlocked(
        self, *, max_new_conditions: int | None = None
    ) -> ProcessQualityFormalSummary:
        """Execute missing conditions; commit final outputs only after 3/3 pass."""

        if max_new_conditions is not None and (
            isinstance(max_new_conditions, bool)
            or not isinstance(max_new_conditions, int)
            or max_new_conditions <= 0
        ):
            raise ValueError("max_new_conditions must be a positive integer.")
        self._validate_scope_file()
        checkpoints: dict[str, Mapping[str, Any]] = {}
        for condition in self.conditions:
            manifest = self._load_checkpoint(condition)
            if manifest is not None:
                checkpoints[str(condition["formal_condition_id"])] = manifest
        skipped = len(checkpoints)
        executed = 0
        failed = 0

        final_manifest = self.output_dir / _FINAL_MANIFEST
        if final_manifest.exists():
            if len(checkpoints) != 3:
                raise ProcessQualityFormalIntegrityError(
                    "A final manifest exists without all three committed checkpoints."
                )
            self._validate_final_manifest()
            self._write_progress(
                status="complete",
                completed=3,
                executed=0,
                skipped=3,
                failed=0,
            )
            return ProcessQualityFormalSummary(
                status="complete",
                expected_conditions=3,
                completed_conditions=3,
                executed_this_invocation=0,
                skipped_this_invocation=3,
                failed_conditions=0,
                output_dir=str(self.output_dir),
            )

        self._materialize_evaluation_events()

        for condition in self.conditions:
            condition_id = str(condition["formal_condition_id"])
            if condition_id in checkpoints:
                continue
            if max_new_conditions is not None and executed >= max_new_conditions:
                break
            try:
                self._write_progress(
                    status="running_condition",
                    completed=len(checkpoints),
                    executed=executed,
                    skipped=skipped,
                    failed=failed,
                )
                LOGGER.info(
                    "Starting representative condition %s",
                    condition_id,
                )
                self._execute_condition(condition)
                manifest = self._load_checkpoint(condition)
                if manifest is None:
                    raise RuntimeError(
                        "Process-quality condition finished without a commit manifest."
                    )
                checkpoints[condition_id] = manifest
                executed += 1
                self._write_progress(
                    status="running",
                    completed=len(checkpoints),
                    executed=executed,
                    skipped=skipped,
                    failed=failed,
                )
            except ProcessQualityFormalIntegrityError:
                self._write_progress(
                    status="blocked",
                    completed=len(checkpoints),
                    executed=executed,
                    skipped=skipped,
                    failed=failed,
                )
                raise
            except Exception as exc:
                failed += 1
                self._write_failure(condition, exc)
                self._write_progress(
                    status="failed",
                    completed=len(checkpoints),
                    executed=executed,
                    skipped=skipped,
                    failed=failed,
                )
                raise ProcessQualityFormalConditionError(
                    f"Representative process-quality condition failed: {condition_id}"
                ) from exc

        if len(checkpoints) == 3:
            self._aggregate(checkpoints, executed=executed, skipped=skipped)
            self._validate_final_manifest()
            self._write_progress(
                status="complete",
                completed=3,
                executed=executed,
                skipped=skipped,
                failed=0,
            )
            status = "complete"
        else:
            self._write_progress(
                status="partial",
                completed=len(checkpoints),
                executed=executed,
                skipped=skipped,
                failed=failed,
            )
            status = "partial"
        return ProcessQualityFormalSummary(
            status=status,
            expected_conditions=3,
            completed_conditions=len(checkpoints),
            executed_this_invocation=executed,
            skipped_this_invocation=skipped,
            failed_conditions=failed,
            output_dir=str(self.output_dir),
        )


__all__ = [
    "ProcessQualityExecutionPlan",
    "ProcessQualityFormalConditionError",
    "ProcessQualityFormalIntegrityError",
    "ProcessQualityFormalSummary",
    "RepresentativeProcessQualityRunner",
    "build_process_quality_execution_plan",
    "write_process_quality_execution_plan",
    "validate_disjoint_directories",
]

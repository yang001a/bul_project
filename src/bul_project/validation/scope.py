"""Immutable scope freeze for the three formal PM4Py validation conditions.

This module does not execute PM4Py and does not read an XES log.  It binds the
pre-registered external-validation scope to the already completed numerical
reference plan and to three committed DP-full checkpoints.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from bul_project.validation.evaluation_sample import (
    expected_evaluation_sample_metadata,
    expected_evaluation_sampling_policy,
)
from bul_project.config import (
    BUDGET_STRATEGIES,
    EPSILON_VALUES,
    EXPERIMENT_SEEDS,
    EXTERNAL_VALIDATION_BUDGET_STRATEGY,
    EXTERNAL_VALIDATION_DATASET,
    EXTERNAL_VALIDATION_EPSILON_VALUES,
    EXTERNAL_VALIDATION_METHOD,
    FROZEN_EDGE_TAU,
    FROZEN_TOP_K_VARIANTS,
)


SCOPE_VERSION = "PM4Py_scope_v1"
SCOPE_NAME = "BUL_formal_process_quality_scope"
SCOPE_DATASET = EXTERNAL_VALIDATION_DATASET
SCOPE_METHOD = EXTERNAL_VALIDATION_METHOD
SCOPE_BUDGET_STRATEGY = EXTERNAL_VALIDATION_BUDGET_STRATEGY
SCOPE_EPSILONS = EXTERNAL_VALIDATION_EPSILON_VALUES
SEED_CANDIDATES = EXPERIMENT_SEEDS
REFERENCE_TOP_K = FROZEN_TOP_K_VARIANTS
REFERENCE_EDGE_TAU = FROZEN_EDGE_TAU
EXTERNAL_VALIDITY_ROLE = "representative_external_validity_check_only"
EXTERNAL_VALIDITY_CLAIM_STRENGTH = "feasibility_only"

_EXPECTED_DATASETS = ("BPIC2012", "BPIC2017", "SEPSIS")
_EXPECTED_EPSILONS = EPSILON_VALUES
_EXPECTED_STRATEGIES = BUDGET_STRATEGIES
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProcessQualityScopeIntegrityError(RuntimeError):
    """Raised when a scope source, frozen field, or immutable file changed."""


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


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest of a file.

    Inputs:
        path: Path to the input or output artifact.
    Outputs:
        A value of type `str` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProcessQualityScopeIntegrityError(
            f"{label} must be one lowercase hexadecimal SHA-256 digest."
        )
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Validate and return a mapping value.

    Inputs:
        value: Candidate value to validate or transform.
        label: Input value used by this operation.
    Outputs:
        A value of type `Mapping[str, Any]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if not isinstance(value, Mapping):
        raise ProcessQualityScopeIntegrityError(f"{label} must be a mapping.")
    return value


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    """Validate and return a sequence value.

    Inputs:
        value: Candidate value to validate or transform.
        label: Input value used by this operation.
    Outputs:
        A value of type `Sequence[Any]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProcessQualityScopeIntegrityError(f"{label} must be a sequence.")
    return value


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    """Read and parse a JSON document.

    Inputs:
        path: Path to the input or output artifact.
        label: Input value used by this operation.
    Outputs:
        A value of type `Mapping[str, Any]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if not path.is_file():
        raise ProcessQualityScopeIntegrityError(f"{label} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProcessQualityScopeIntegrityError(
            f"{label} is not valid UTF-8 JSON: {path}"
        ) from exc
    return _require_mapping(value, label)


def _formal_plan_hash(plan: Mapping[str, Any]) -> str:
    """Return formal plan hash.

    Inputs:
        plan: Input value used by this operation.
    Outputs:
        A value of type `str` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    core = dict(plan)
    declared = _require_sha256(core.pop("plan_sha256", None), "plan.plan_sha256")
    actual = _sha256_bytes(_json_bytes(core))
    if actual != declared:
        raise ProcessQualityScopeIntegrityError(
            "The formal execution plan core differs from its declared plan hash."
        )
    return declared


def score_preregistered_seed(current_plan_hash: str, seed: int) -> str:
    """Return ``SHA256(plan_hash|PM4Py_scope_v1|seed=<seed>)``.

    Input
    -----
    current_plan_hash:
        The immutable internal SHA-256 of the completed numerical reference plan.
    seed:
        One candidate from the predeclared integer range 0--29.

    Output
    ------
    str
        Lowercase hexadecimal score used only for outcome-independent selection.

    Paper experiment step
    ---------------------
    Phase 6 external-validation pre-registration.  The score is calculated
    before PM4Py results exist and never depends on a quality metric.
    """

    plan_hash = _require_sha256(current_plan_hash, "current_plan_hash")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEED_CANDIDATES:
        raise ValueError("seed must be one of the predeclared candidates 0..29.")
    material = f"{plan_hash}|{SCOPE_VERSION}|seed={seed}".encode("utf-8")
    return _sha256_bytes(material)


@dataclass(frozen=True)
class PreregisteredSeedSelection:
    """Outcome-independent seed selection and its complete score table."""

    current_plan_hash: str
    selected_seed: int
    selected_score: str
    scores: tuple[tuple[int, str], ...]

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
            "algorithm": "minimum_SHA256_score",
            "scope_version": SCOPE_VERSION,
            "score_rule": (
                "SHA256(current_plan_hash|PM4Py_scope_v1|seed=<seed>)"
            ),
            "score_encoding": "UTF-8; lowercase hexadecimal digest",
            "candidate_seeds": list(SEED_CANDIDATES),
            "current_plan_hash": self.current_plan_hash,
            "selected_seed": self.selected_seed,
            "selected_score": self.selected_score,
            "scores": [
                {"seed": seed, "score": score} for seed, score in self.scores
            ],
            "selection_timing": "frozen_before_any_formal_PM4Py_result",
        }


def select_preregistered_seed(current_plan_hash: str) -> PreregisteredSeedSelection:
    """Select the 0--29 seed having the lexicographically lowest SHA-256 score."""

    plan_hash = _require_sha256(current_plan_hash, "current_plan_hash")
    scores = tuple(
        (seed, score_preregistered_seed(plan_hash, seed))
        for seed in SEED_CANDIDATES
    )
    score_values = [score for _, score in scores]
    if len(set(score_values)) != len(score_values):
        raise ProcessQualityScopeIntegrityError(
            "Seed selection encountered a SHA-256 score collision."
        )
    selected_seed, selected_score = min(scores, key=lambda item: item[1])
    return PreregisteredSeedSelection(
        current_plan_hash=plan_hash,
        selected_seed=selected_seed,
        selected_score=selected_score,
        scores=scores,
    )


@dataclass(frozen=True)
class FrozenProcessQualityScope:
    """JSON-compatible immutable-scope payload."""

    payload: Mapping[str, Any]

    @property
    def scope_sha256(self) -> str:
        """Return scope sha256.

        Inputs:
            None.
        Outputs:
            A value of type `str` produced by the operation.
        Paper role:
            Artifact provenance and representative external feasibility validation.
        """
        return _require_sha256(
            self.payload.get("scope_sha256"), "scope.scope_sha256"
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
        return copy.deepcopy(dict(self.payload))


def _dataset_by_name(
    config: Mapping[str, Any], dataset: str
) -> Mapping[str, Any]:
    """Return dataset by name.

    Inputs:
        config: Configuration controlling the operation.
        dataset: Dataset identifier or dataset-specific input.
    Outputs:
        A value of type `Mapping[str, Any]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    rows = _require_sequence(config.get("datasets"), "config.datasets")
    mappings = [_require_mapping(row, "config.datasets[]") for row in rows]
    names = [row.get("dataset") for row in mappings]
    if tuple(names) != _EXPECTED_DATASETS:
        raise ProcessQualityScopeIntegrityError(
            "The frozen dataset order/scope differs from BPIC2012/BPIC2017/SEPSIS."
        )
    selected = [row for row in mappings if row.get("dataset") == dataset]
    if len(selected) != 1:
        raise ProcessQualityScopeIntegrityError(
            f"Frozen config must contain exactly one {dataset} dataset entry."
        )
    return selected[0]


def _validate_reference_sources(
    plan_path: Path,
    formal_manifest_path: Path,
    config_path: Path,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    str,
    str,
    str,
    str,
]:
    """Validate reference sources.

    Inputs:
        plan_path: Input value used by this operation.
        formal_manifest_path: Input value used by this operation.
        config_path: Input value used by this operation.
    Outputs:
        A value of type `tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str, str, str, str]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    plan = _read_json(plan_path, "formal execution plan")
    manifest = _read_json(formal_manifest_path, "formal run manifest")
    config = _read_json(config_path, "Phase 7C final config")
    plan_sha256 = _formal_plan_hash(plan)
    plan_file_sha256 = _sha256_file(plan_path)
    manifest_file_sha256 = _sha256_file(formal_manifest_path)
    config_file_sha256 = _sha256_file(config_path)

    if (
        plan.get("name") != "BUL_formal_execution_plan"
        or plan.get("phase") != "8A"
        or plan.get("status") != "planned"
        or plan.get("scope") != "reference_numeric"
        or plan.get("scenario_id") != "reference_k100_tau10"
        or plan.get("process_quality_scope_status") != "deferred_not_frozen"
        or plan.get("process_quality_conditions") != 0
    ):
        raise ProcessQualityScopeIntegrityError(
            "The source plan is not the completed reference-numeric plan."
        )
    if (
        manifest.get("name") != "BUL_formal_run"
        or manifest.get("phase") != "8A"
        or manifest.get("status") != "complete"
        or manifest.get("scope") != "reference_numeric"
        or manifest.get("scenario_id") != plan.get("scenario_id")
        or manifest.get("execution_plan_sha256") != plan_sha256
        or manifest.get("process_quality_scope_status") != "deferred_not_frozen"
    ):
        raise ProcessQualityScopeIntegrityError(
            "The source completion manifest is not the matching complete run."
        )
    counts = _require_mapping(manifest.get("counts"), "manifest.counts")
    if counts.get("failed_conditions") != 0 or counts.get(
        "completed_conditions"
    ) != counts.get("dp_conditions"):
        raise ProcessQualityScopeIntegrityError(
            "The numerical reference run is not complete and failure-free."
        )
    artifacts = _require_mapping(
        manifest.get("artifact_sha256"), "manifest.artifact_sha256"
    )
    if artifacts.get(plan_path.name) != plan_file_sha256:
        raise ProcessQualityScopeIntegrityError(
            "The plan file hash differs from the completion manifest."
        )
    if (
        config.get("name") != "BUL_experiment_config_final"
        or config.get("phase") != "7C"
        or config.get("status") != "frozen"
        or plan.get("phase7c_final_config_sha256") != config_file_sha256
        or manifest.get("phase7c_final_config_sha256") != config_file_sha256
    ):
        raise ProcessQualityScopeIntegrityError(
            "The Phase 7C final config is not the one bound to the main plan."
        )
    return (
        plan,
        manifest,
        config,
        plan_sha256,
        plan_file_sha256,
        manifest_file_sha256,
        config_file_sha256,
    )


def _build_scope_core(
    plan_path: Path,
    formal_manifest_path: Path,
    config_path: Path,
    checkpoint_root: Path,
) -> dict[str, Any]:
    """Build scope core.

    Inputs:
        plan_path: Input value used by this operation.
        formal_manifest_path: Input value used by this operation.
        config_path: Input value used by this operation.
        checkpoint_root: Input value used by this operation.
    Outputs:
        A value of type `dict[str, Any]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    (
        plan,
        manifest,
        config,
        plan_sha256,
        plan_file_sha256,
        manifest_file_sha256,
        config_file_sha256,
    ) = _validate_reference_sources(plan_path, formal_manifest_path, config_path)

    seed_selection = select_preregistered_seed(plan_sha256)
    if seed_selection.selected_seed != 27:
        raise ProcessQualityScopeIntegrityError(
            "The predeclared PM4Py_scope_v1 rule no longer selects seed 27."
        )

    reference = _require_mapping(
        plan.get("reference_parameters"), "plan.reference_parameters"
    )
    manifest_reference = _require_mapping(
        manifest.get("reference_parameters"), "manifest.reference_parameters"
    )
    if (
        reference != manifest_reference
        or reference.get("top_k") != REFERENCE_TOP_K
        or reference.get("edge_tau") != REFERENCE_EDGE_TAU
    ):
        raise ProcessQualityScopeIntegrityError(
            "The main reference is not the frozen "
            f"k={REFERENCE_TOP_K}/tau={REFERENCE_EDGE_TAU} scenario."
        )

    matrix = _require_mapping(
        config.get("experiment_matrix"), "config.experiment_matrix"
    )
    if (
        tuple(float(value) for value in _require_sequence(
            matrix.get("epsilon_values"), "epsilon_values"
        ))
        != _EXPECTED_EPSILONS
        or tuple(_require_sequence(matrix.get("seeds"), "seeds"))
        != SEED_CANDIDATES
        or tuple(
            _require_sequence(matrix.get("budget_strategies"), "budget_strategies")
        )
        != _EXPECTED_STRATEGIES
    ):
        raise ProcessQualityScopeIntegrityError(
            "The frozen epsilon/seed/budget-strategy matrix changed."
        )

    dataset = _dataset_by_name(config, SCOPE_DATASET)
    if (
        dataset.get("top_k_variants") != REFERENCE_TOP_K
        or dataset.get("min_edge_support") != REFERENCE_EDGE_TAU
    ):
        raise ProcessQualityScopeIntegrityError(
            "BPIC2017 does not use the frozen reference k/tau values."
        )
    source_sha256 = _require_sha256(
        dataset.get("source_sha256"), "BPIC2017.source_sha256"
    )
    evaluation = _require_mapping(
        dataset.get("evaluation_log"), "BPIC2017.evaluation_log"
    )
    evaluation_sha256 = _require_sha256(
        evaluation.get("evaluation_log_sha256"),
        "BPIC2017.evaluation_log_sha256",
    )
    if (
        evaluation.get("evaluation_log_id")
        != "BPIC2017_phase1_complete_canonical_log"
        or evaluation.get("event_count") != 1202267
        or evaluation.get("case_count") != 31509
        or evaluation.get("core_columns")
        != ["case_id", "activity", "timestamp"]
        or evaluation.get("timestamp_policy")
        != "complete canonical timestamps required"
    ):
        raise ProcessQualityScopeIntegrityError(
            "The unified BPIC2017 evaluation-log contract changed."
        )

    quality = _require_mapping(
        config.get("process_quality"), "config.process_quality"
    )
    noise = quality.get("inductive_miner_noise_threshold")
    if (
        quality.get("datasets") != [SCOPE_DATASET]
        or quality.get("conformance_variant") != "token_based"
        or isinstance(noise, bool)
        or not isinstance(noise, (int, float))
        or not math.isclose(float(noise), 0.0, rel_tol=0.0, abs_tol=0.0)
    ):
        raise ProcessQualityScopeIntegrityError(
            "The frozen process-quality method contract changed."
        )

    plan_sources = _require_mapping(plan.get("source_sha256"), "plan.source_sha256")
    manifest_sources = _require_mapping(
        manifest.get("source_sha256"), "manifest.source_sha256"
    )
    config_sources = {
        row["dataset"]: row["source_sha256"]
        for row in _require_sequence(config.get("datasets"), "config.datasets")
    }
    if plan_sources != manifest_sources or dict(plan_sources) != config_sources:
        raise ProcessQualityScopeIntegrityError(
            "Dataset source hashes differ across plan, manifest, and config."
        )

    condition_plan = _require_sequence(
        plan.get("condition_plan"), "plan.condition_plan"
    )
    checkpoint_hashes = _require_mapping(
        manifest.get("checkpoint_manifest_sha256"),
        "manifest.checkpoint_manifest_sha256",
    )
    selected_conditions: list[dict[str, Any]] = []
    for epsilon in SCOPE_EPSILONS:
        matches = []
        for row_value in condition_plan:
            row = _require_mapping(row_value, "plan.condition_plan[]")
            condition = _require_mapping(row.get("condition"), "condition")
            if (
                condition.get("dataset") == SCOPE_DATASET
                and condition.get("method") == SCOPE_METHOD
                and condition.get("budget_strategy") == SCOPE_BUDGET_STRATEGY
                and condition.get("seed") == seed_selection.selected_seed
                and condition.get("epsilon") == epsilon
            ):
                matches.append(row)
        if len(matches) != 1:
            raise ProcessQualityScopeIntegrityError(
                f"Expected exactly one source condition for epsilon={epsilon:g}."
            )
        row = matches[0]
        condition = _require_mapping(row.get("condition"), "condition")
        expected_condition_id = (
            f"{SCOPE_DATASET}|{SCOPE_METHOD}|epsilon={epsilon:g}|"
            f"seed={seed_selection.selected_seed}|"
            f"mechanism={SCOPE_BUDGET_STRATEGY}"
        )
        formal_id = (
            f"reference_k100_tau10|{expected_condition_id}"
        )
        if (
            condition.get("condition_id") != expected_condition_id
            or condition.get("dp_edge_mode") is not None
            or condition.get("evaluate_process_quality") is not False
            or condition.get("conformance_variant") is not None
            or condition.get("process_model_noise_threshold") is not None
            or row.get("formal_condition_id") != formal_id
        ):
            raise ProcessQualityScopeIntegrityError(
                "A selected source condition differs from the numerical contract."
            )
        checkpoint_key = _sha256_bytes(formal_id.encode("utf-8"))
        if row.get("checkpoint_key") != checkpoint_key:
            raise ProcessQualityScopeIntegrityError(
                "A selected source condition has a changed checkpoint key."
            )
        expected_checkpoint_sha = _require_sha256(
            checkpoint_hashes.get(formal_id),
            f"checkpoint hash for {formal_id}",
        )
        checkpoint_dir = checkpoint_root / checkpoint_key
        checkpoint_path = checkpoint_dir / "checkpoint_manifest.json"
        if not checkpoint_path.is_file():
            raise ProcessQualityScopeIntegrityError(
                f"Selected checkpoint is unavailable: {checkpoint_path}"
            )
        checkpoint_sha = _sha256_file(checkpoint_path)
        if checkpoint_sha != expected_checkpoint_sha:
            raise ProcessQualityScopeIntegrityError(
                "A selected checkpoint manifest hash changed."
            )
        checkpoint = _read_json(checkpoint_path, "checkpoint manifest")
        if (
            checkpoint.get("name") != "BUL_formal_condition_checkpoint"
            or checkpoint.get("status") != "complete"
            or checkpoint.get("formal_condition_id") != formal_id
            or checkpoint.get("execution_plan_sha256") != plan_sha256
            or checkpoint.get("phase7c_final_config_sha256") != config_file_sha256
            or checkpoint.get("dataset_source_sha256") != source_sha256
            or checkpoint.get("evaluation_log_sha256") != evaluation_sha256
            or checkpoint.get("condition") != condition
            or checkpoint.get("raw_data_rows_serialized") is not False
        ):
            raise ProcessQualityScopeIntegrityError(
                "A selected checkpoint is not bound to the frozen source condition."
            )
        artifacts_map = _require_mapping(
            checkpoint.get("artifacts"), "checkpoint.artifacts"
        )
        artifact_hashes = _require_mapping(
            checkpoint.get("artifact_sha256"), "checkpoint.artifact_sha256"
        )
        expected_artifacts = {
            "experiment_metadata": "experiment_metadata.json",
            "experiment_result": "experiment_result.csv",
        }
        if artifacts_map != expected_artifacts or set(artifact_hashes) != set(
            expected_artifacts.values()
        ):
            raise ProcessQualityScopeIntegrityError(
                "A selected checkpoint artifact contract changed."
            )
        for file_name, declared_hash in artifact_hashes.items():
            artifact_path = checkpoint_dir / file_name
            if (
                not artifact_path.is_file()
                or _sha256_file(artifact_path)
                != _require_sha256(declared_hash, f"{file_name} SHA-256")
            ):
                raise ProcessQualityScopeIntegrityError(
                    f"Selected checkpoint artifact changed: {file_name}"
                )
        selected_conditions.append(
            {
                "epsilon": epsilon,
                "formal_condition_id": formal_id,
                "checkpoint_key": checkpoint_key,
                "checkpoint_manifest_sha256": checkpoint_sha,
                "experiment_metadata_sha256": artifact_hashes[
                    "experiment_metadata.json"
                ],
                "experiment_result_sha256": artifact_hashes[
                    "experiment_result.csv"
                ],
                "source_role": "verified_DP_release_checkpoint",
            }
        )

    synthetic_policy = _require_mapping(
        config.get("synthetic_log_policy"), "config.synthetic_log_policy"
    )
    if (
        synthetic_policy.get("variant_driven") is not True
        or synthetic_policy.get("other_excluded") is not True
        or synthetic_policy.get("same_generation_rule_across_conditions") is not True
        or synthetic_policy.get("timestamp_start") != "2026-01-01T00:00:00"
    ):
        raise ProcessQualityScopeIntegrityError(
            "The frozen DP synthetic-log policy changed."
        )

    return {
        "name": SCOPE_NAME,
        "original_phase": 6,
        "scope_version": SCOPE_VERSION,
        "status": "frozen",
        "source_bundle": {
            "formal_execution_plan": {
                "file_name": plan_path.name,
                "file_sha256": plan_file_sha256,
                "internal_plan_sha256": plan_sha256,
            },
            "formal_run_manifest": {
                "file_name": formal_manifest_path.name,
                "file_sha256": manifest_file_sha256,
            },
            "phase7c_final_config": {
                "file_name": config_path.name,
                "file_sha256": config_file_sha256,
            },
            "scenario_id": "reference_k100_tau10",
        },
        "seed_selection": seed_selection.to_dict(),
        "scope": {
            "dataset": SCOPE_DATASET,
            "method": SCOPE_METHOD,
            "budget_strategy": SCOPE_BUDGET_STRATEGY,
            "epsilon_values": list(SCOPE_EPSILONS),
            "seed": seed_selection.selected_seed,
            "condition_count": 3,
            "top_k_variants": REFERENCE_TOP_K,
            "min_edge_support": REFERENCE_EDGE_TAU,
            "l_max": dataset.get("l_max"),
            "t_max_seconds": dataset.get("t_max_seconds"),
            "dataset_source_sha256": source_sha256,
        },
        "evaluation_log_source": copy.deepcopy(dict(evaluation)),
        "evaluation_log": expected_evaluation_sample_metadata(),
        "evaluation_sampling": expected_evaluation_sampling_policy(),
        "process_quality_method": {
            "discovery_algorithm": "PM4Py Inductive Miner",
            "conformance_variant": "token_based",
            "inductive_miner_noise_threshold": 0.0,
            "metrics": ["fitness", "precision", "generalization"],
            "same_evaluation_log_for_all_conditions": True,
        },
        "synthetic_log_policy": {
            "input_boundary": "verified_DP_release_only",
            "raw_trace_access": False,
            "variant_driven": True,
            "other_excluded": True,
            "missing_temporal_edge_policy": "fail_and_record_missing_edge",
            "timestamp_start": "2026-01-01T00:00:00",
            "timestamp_direction": "nondecreasing_add_interval",
            "seed": seed_selection.selected_seed,
        },
        "conditions": selected_conditions,
        "phase6_scope_boundary": {
            "included_core_tasks": [
                "main_BUL_numeric_matrix",
                "BUL_equal_and_fixed_reference_BUL_entropy",
                "dimension_ablation",
                "parameter_sensitivity",
                "cross_dataset_summary",
            ],
            "included_core_tasks_are_not_completion_claims": True,
            "external_validity_exception": {
                "dataset": SCOPE_DATASET,
                "method": SCOPE_METHOD,
                "budget_strategy": SCOPE_BUDGET_STRATEGY,
                "epsilon_values": list(SCOPE_EPSILONS),
                "seed": seed_selection.selected_seed,
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
        },
        "explicit_exclusions": {
            "datasets": ["BPIC2012", "SEPSIS"],
            "methods": ["DP-edge-only"],
            "epsilon_values": [1.0, 4.0],
            "budget_strategies": list(_EXPECTED_STRATEGIES[1:]),
            "seeds": [seed for seed in SEED_CANDIDATES if seed != 27],
            "metrics": ["simplicity"],
            "dp_edge_only_quality_fields": "NA",
            "other_variant_in_synthetic_generation": False,
        },
        "claim_boundary": {
            "role": EXTERNAL_VALIDITY_ROLE,
            "claim_strength": EXTERNAL_VALIDITY_CLAIM_STRENGTH,
            "not_a_30_seed_estimate": True,
            "not_budget_strategy_robustness_evidence": True,
            "not_cross_dataset_evidence": True,
            "synthetic_log_is_DP_post_processing": True,
            "quality_metrics_are_evaluation_artifacts_not_DP_releases": True,
        },
    }


def build_process_quality_scope(
    plan_path: str | Path,
    formal_manifest_path: str | Path,
    config_path: str | Path,
    *,
    checkpoint_root: str | Path | None = None,
) -> FrozenProcessQualityScope:
    """Build the exact three-condition scope from verified frozen artifacts."""

    plan_file = Path(plan_path).resolve()
    manifest_file = Path(formal_manifest_path).resolve()
    config_file = Path(config_path).resolve()
    checkpoints = (
        plan_file.parent / "checkpoints"
        if checkpoint_root is None
        else Path(checkpoint_root).resolve()
    )
    core = _build_scope_core(
        plan_file,
        manifest_file,
        config_file,
        checkpoints,
    )
    payload = {**core, "scope_sha256": _sha256_bytes(_json_bytes(core))}
    return FrozenProcessQualityScope(payload=payload)


def validate_process_quality_scope(
    scope: FrozenProcessQualityScope | Mapping[str, Any],
    plan_path: str | Path,
    formal_manifest_path: str | Path,
    config_path: str | Path,
    *,
    checkpoint_root: str | Path | None = None,
) -> FrozenProcessQualityScope:
    """Fail closed unless a scope exactly matches all current frozen sources."""

    if isinstance(scope, FrozenProcessQualityScope):
        supplied = scope.to_dict()
    elif isinstance(scope, Mapping):
        supplied = copy.deepcopy(dict(scope))
    else:
        raise TypeError("scope must be FrozenProcessQualityScope or a mapping.")
    declared = _require_sha256(
        supplied.get("scope_sha256"), "scope.scope_sha256"
    )
    core = dict(supplied)
    core.pop("scope_sha256", None)
    if _sha256_bytes(_json_bytes(core)) != declared:
        raise ProcessQualityScopeIntegrityError(
            "The scope payload differs from its declared SHA-256."
        )
    expected = build_process_quality_scope(
        plan_path,
        formal_manifest_path,
        config_path,
        checkpoint_root=checkpoint_root,
    )
    if supplied != expected.to_dict():
        raise ProcessQualityScopeIntegrityError(
            "The supplied process-quality scope differs from the frozen sources."
        )
    return expected


def write_process_quality_scope(
    scope: FrozenProcessQualityScope | Mapping[str, Any],
    output_path: str | Path,
    plan_path: str | Path,
    formal_manifest_path: str | Path,
    config_path: str | Path,
    *,
    checkpoint_root: str | Path | None = None,
) -> Path:
    """Write once; an existing byte-different scope is never overwritten."""

    validated = validate_process_quality_scope(
        scope,
        plan_path,
        formal_manifest_path,
        config_path,
        checkpoint_root=checkpoint_root,
    )
    destination = Path(output_path).resolve()
    payload = _json_bytes(validated.to_dict())
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
            handle.flush()
    except FileExistsError:
        if destination.read_bytes() != payload:
            raise ProcessQualityScopeIntegrityError(
                "Existing process-quality scope differs and is immutable."
            )
    return destination


def load_process_quality_scope(
    scope_path: str | Path,
    plan_path: str | Path,
    formal_manifest_path: str | Path,
    config_path: str | Path,
    *,
    checkpoint_root: str | Path | None = None,
) -> FrozenProcessQualityScope:
    """Load and fully revalidate an immutable process-quality scope file."""

    path = Path(scope_path).resolve()
    payload = _read_json(path, "process-quality scope")
    return validate_process_quality_scope(
        payload,
        plan_path,
        formal_manifest_path,
        config_path,
        checkpoint_root=checkpoint_root,
    )


__all__ = [
    "FrozenProcessQualityScope",
    "EXTERNAL_VALIDITY_CLAIM_STRENGTH",
    "EXTERNAL_VALIDITY_ROLE",
    "PreregisteredSeedSelection",
    "ProcessQualityScopeIntegrityError",
    "SCOPE_VERSION",
    "build_process_quality_scope",
    "load_process_quality_scope",
    "score_preregistered_seed",
    "select_preregistered_seed",
    "validate_process_quality_scope",
    "write_process_quality_scope",
]

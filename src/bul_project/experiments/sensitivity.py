"""Resumable Top-k and edge-threshold BUL sensitivity experiment.

The approved compact scope is deliberately numerical only: BPIC2017,
DP-full/uniform, three epsilon values, thirty seeds, and the complete 3x3
Top-k by edge-tau grid.  It never requests PM4Py or serializes raw events.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bul_project.bul import calculate_BUL
from bul_project.config import (
    EXPERIMENT_SEEDS,
    FROZEN_EDGE_TAU,
    FROZEN_TOP_K_VARIANTS,
    PROJECT_ROOT,
    SENSITIVITY_BUDGET_STRATEGY,
    SENSITIVITY_DATASET,
    SENSITIVITY_EDGE_TAU_OPTIONS,
    SENSITIVITY_EPSILON_VALUES,
    SENSITIVITY_METHOD,
    SENSITIVITY_TOP_K_OPTIONS,
)
from bul_project.experiments.main_experiment import load_frozen_execution_spec
from bul_project.experiments.single_run import (
    SingleExperimentConfig,
    SingleExperimentResult,
    run_single_experiment,
)
from bul_project.preprocessing import Trace


SENSITIVITY_EPSILONS = SENSITIVITY_EPSILON_VALUES
SENSITIVITY_SEEDS = EXPERIMENT_SEEDS
SENSITIVITY_TOP_K = SENSITIVITY_TOP_K_OPTIONS
SENSITIVITY_EDGE_TAU = SENSITIVITY_EDGE_TAU_OPTIONS
REFERENCE_TOP_K = FROZEN_TOP_K_VARIANTS
REFERENCE_EDGE_TAU = FROZEN_EDGE_TAU
SENSITIVITY_SCENARIO = "bpic2017_dpfull_uniform_topk_tau_sensitivity"

_PLAN_FILE = "sensitivity_execution_plan.json"
_PROGRESS_FILE = "sensitivity_progress.json"
_RESULT_FILE = "sensitivity_result.csv"
_STAGE_FILE = "sensitivity_stage_result.csv"
_SUMMARY_FILE = "sensitivity_summary.csv"
_FINAL_MANIFEST = "phase6_sensitivity_manifest.json"
_CHECKPOINT_FILE = "sensitivity_checkpoint.json"

_SENSITIVITY_CODE_FILES = (
    "src/bul_project/experiments/__init__.py",
    "src/bul_project/experiments/sensitivity.py",
    "scripts/run_sensitivity_analysis.py",
)
_LOSS_FIELDS = ("L_structure", "L_temporal", "L_information", "BUL")
_STAGES = ("clipping", "dp_noise", "total")


class Phase6SensitivityError(RuntimeError):
    """Raised when a frozen input, checkpoint, or result contract is invalid."""


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


def _json_bytes(value: object) -> bytes:
    """Serialize JSON content to deterministic UTF-8 bytes.

    Inputs:
        value: Candidate value to validate or transform.
    Outputs:
        A value of type `bytes` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path, role: str) -> dict[str, Any]:
    """Read and parse a JSON document.

    Inputs:
        path: Path to the input or output artifact.
        role: Input value used by this operation.
    Outputs:
        A value of type `dict[str, Any]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase6SensitivityError(f"Cannot read {role}: {path}") from exc
    if not isinstance(value, dict):
        raise Phase6SensitivityError(f"{role} must be a JSON object.")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write an artifact atomically.

    Inputs:
        path: Path to the input or output artifact.
        payload: Structured content to validate, serialize, or write.
    Outputs:
        None.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_immutable(path: Path, payload: bytes) -> None:
    """Write an artifact without overwriting conflicting content.

    Inputs:
        path: Path to the input or output artifact.
        payload: Structured content to validate, serialize, or write.
    Outputs:
        None.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise Phase6SensitivityError(
                f"Immutable artifact differs from the frozen run: {path}"
            )
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise Phase6SensitivityError(
                    f"Concurrent immutable artifact differs: {path}"
                )
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize tabular content to deterministic UTF-8 CSV bytes.

    Inputs:
        frame: Tabular data processed by the operation.
    Outputs:
        A value of type `bytes` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    stream = io.StringIO(newline="")
    frame.to_csv(stream, index=False, na_rep="NA", lineterminator="\n")
    return stream.getvalue().encode("utf-8")


def _require_hash(path: Path, expected: str, role: str) -> str:
    """Return require hash.

    Inputs:
        path: Path to the input or output artifact.
        expected: Input value used by this operation.
        role: Input value used by this operation.
    Outputs:
        A value of type `str` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    if not path.is_file():
        raise Phase6SensitivityError(f"Missing {role}: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise Phase6SensitivityError(f"{role} SHA-256 changed: {path}")
    return actual


def _finite_loss(value: object, role: str) -> float:
    """Validate and return a finite loss value.

    Inputs:
        value: Candidate value to validate or transform.
        role: Input value used by this operation.
    Outputs:
        A value of type `float` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise Phase6SensitivityError(f"{role} is not numeric.") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise Phase6SensitivityError(f"{role} must be finite in [0,1].")
    return number


@dataclass(frozen=True, order=True)
class SensitivityCondition:
    """One immutable DP-full parameter-sensitivity condition."""

    top_k: int
    edge_tau: int
    epsilon: float
    seed: int

    @property
    def is_reference_reuse(self) -> bool:
        """Return whether reference reuse.

        Inputs:
            None.
        Outputs:
            A value of type `bool` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return self.top_k == REFERENCE_TOP_K and self.edge_tau == REFERENCE_EDGE_TAU

    @property
    def condition_id(self) -> str:
        """Create id.

        Inputs:
            None.
        Outputs:
            A value of type `str` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return (
            f"{SENSITIVITY_SCENARIO}|k={self.top_k}|tau={self.edge_tau}|"
            f"epsilon={self.epsilon:g}|seed={self.seed}"
        )

    @property
    def checkpoint_key(self) -> str:
        """Return checkpoint key.

        Inputs:
            None.
        Outputs:
            A value of type `str` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return _sha256_bytes(self.condition_id.encode("utf-8"))

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
            "condition_id": self.condition_id,
            "checkpoint_key": self.checkpoint_key,
            "dataset": SENSITIVITY_DATASET,
            "method": SENSITIVITY_METHOD,
            "budget_strategy": SENSITIVITY_BUDGET_STRATEGY,
            "top_k_variants": self.top_k,
            "min_edge_support": self.edge_tau,
            "epsilon": self.epsilon,
            "seed": self.seed,
            "execution_source": (
                "authenticated_reference_reuse"
                if self.is_reference_reuse
                else "new_sensitivity_execution"
            ),
        }


def build_sensitivity_conditions() -> tuple[SensitivityCondition, ...]:
    """Build the complete 810-condition compact parameter grid.

    Input
    -----
    None. The values are the Phase 7C-frozen sensitivity candidates.

    Output
    ------
    tuple[SensitivityCondition, ...]
        Nine Top-k/tau cells, three epsilons, and thirty seeds. Exactly ninety
        reference-cell conditions are marked for authenticated reuse.

    Paper experiment step
    ---------------------
    Original Phase 6 parameter robustness: full Top-k by edge-tau cross
    product on the main BPIC2017 dataset without expanding unrelated methods.
    """

    conditions = tuple(
        SensitivityCondition(k, tau, epsilon, seed)
        for k in SENSITIVITY_TOP_K
        for tau in SENSITIVITY_EDGE_TAU
        for epsilon in SENSITIVITY_EPSILONS
        for seed in SENSITIVITY_SEEDS
    )
    if len(conditions) != 810 or len({row.condition_id for row in conditions}) != 810:
        raise RuntimeError("Sensitivity plan must contain 810 unique conditions.")
    if sum(row.is_reference_reuse for row in conditions) != 90:
        raise RuntimeError("Sensitivity plan must reuse exactly 90 conditions.")
    return conditions


@dataclass(frozen=True)
class SensitivitySources:
    config_path: Path
    reference_dir: Path
    derived_dir: Path
    source_path: Path
    source_sha256: str
    l_max: int
    t_max: float
    normalization_scales: Mapping[tuple[str, str], float]
    entropy_weights: Mapping[str, float]
    reference_rows: Mapping[tuple[float, int], Mapping[str, Any]]
    source_metadata: Mapping[str, Any]
    scientific_code_sha256: Mapping[str, str]


def _load_reference_checkpoint_metadata(
    reference_dir: Path,
    manifest: Mapping[str, Any],
    condition_id: str,
) -> dict[str, Any]:
    """Load reference checkpoint metadata.

    Inputs:
        reference_dir: Input value used by this operation.
        manifest: Input value used by this operation.
        condition_id: Input value used by this operation.
    Outputs:
        A value of type `dict[str, Any]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    key = _sha256_bytes(condition_id.encode("utf-8"))
    checkpoint = reference_dir / "checkpoints" / key
    manifest_path = checkpoint / "checkpoint_manifest.json"
    expected_manifests = manifest.get("checkpoint_manifest_sha256")
    if not isinstance(expected_manifests, Mapping):
        raise Phase6SensitivityError("Formal checkpoint hashes are missing.")
    expected_manifest_hash = expected_manifests.get(condition_id)
    if not isinstance(expected_manifest_hash, str):
        raise Phase6SensitivityError(f"Reference checkpoint is unregistered: {condition_id}")
    _require_hash(manifest_path, expected_manifest_hash, "formal checkpoint manifest")
    checkpoint_manifest = _read_json(manifest_path, "formal checkpoint manifest")
    if (
        checkpoint_manifest.get("status") != "complete"
        or checkpoint_manifest.get("formal_condition_id") != condition_id
    ):
        raise Phase6SensitivityError("Formal checkpoint identity changed.")
    artifacts = checkpoint_manifest.get("artifacts")
    hashes = checkpoint_manifest.get("artifact_sha256")
    if not isinstance(artifacts, Mapping) or not isinstance(hashes, Mapping):
        raise Phase6SensitivityError("Formal checkpoint artifacts are missing.")
    filename = artifacts.get("experiment_metadata")
    if not isinstance(filename, str):
        raise Phase6SensitivityError("Formal checkpoint metadata name is invalid.")
    metadata_path = checkpoint / filename
    expected = hashes.get(filename)
    if not isinstance(expected, str):
        raise Phase6SensitivityError("Formal checkpoint metadata hash is missing.")
    _require_hash(metadata_path, expected, "formal checkpoint metadata")
    return _read_json(metadata_path, "formal checkpoint metadata")


def _reference_key_frame(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    """Return reference key frame.

    Inputs:
        frame: Tabular data processed by the operation.
        role: Input value used by this operation.
    Outputs:
        A value of type `pd.DataFrame` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    selected = frame.loc[
        frame["dataset"].eq(SENSITIVITY_DATASET)
        & frame["method"].eq(SENSITIVITY_METHOD)
        & frame["budget_strategy"].eq(SENSITIVITY_BUDGET_STRATEGY)
        & pd.to_numeric(frame["epsilon"], errors="coerce").isin(SENSITIVITY_EPSILONS)
        & pd.to_numeric(frame["seed"], errors="coerce").isin(SENSITIVITY_SEEDS)
    ].copy()
    selected["_epsilon"] = pd.to_numeric(selected["epsilon"], errors="coerce")
    selected["_seed"] = pd.to_numeric(selected["seed"], errors="coerce").astype(int)
    if role == "result":
        keys = list(zip(selected["_epsilon"], selected["_seed"]))
        expected_count = 90
    else:
        keys = list(zip(selected["_epsilon"], selected["_seed"], selected["stage"]))
        expected_count = 270
    if len(selected) != expected_count or len(set(keys)) != expected_count:
        raise Phase6SensitivityError(f"Formal reference {role} grid changed.")
    return selected


def load_sensitivity_sources(
    config_path: str | Path,
    reference_dir: str | Path,
    derived_dir: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> SensitivitySources:
    """Authenticate every frozen source without reading an XES event table.

    Input
    -----
    config_path, reference_dir, derived_dir:
        Phase 7C final config, completed numerical reference, and authoritative
        Phase 6 derived-v2 directory.

    Output
    ------
    SensitivitySources
        Lmax/Tmax, all 178 fixed edge normalization factors, fixed BPIC2017
        entropy weights, and ninety authenticated reusable result conditions.

    Paper experiment step
    ---------------------
    Original Phase 6 sensitivity preflight. Statistical domains and evaluation
    scales are frozen before every epsilon/seed run; no scale is re-estimated.
    """

    config = Path(config_path).resolve()
    reference = Path(reference_dir).resolve()
    derived = Path(derived_dir).resolve()
    root = Path(project_root).resolve()
    frozen_source_root = root / "provenance" / "frozen_source"
    code_root = frozen_source_root if frozen_source_root.is_dir() else root
    config_hash = _sha256_file(config)
    spec = load_frozen_execution_spec(config)
    dataset_spec = next(
        (item for item in spec.datasets if item.dataset == SENSITIVITY_DATASET), None
    )
    if dataset_spec is None or dataset_spec.source_path is None:
        raise Phase6SensitivityError("Frozen BPIC2017 source is unavailable.")
    if dataset_spec.l_max != 70 or not math.isclose(
        dataset_spec.t_max, 326449.43039999995, rel_tol=0.0, abs_tol=1e-9
    ):
        raise Phase6SensitivityError("Frozen BPIC2017 Lmax/Tmax changed.")

    formal_manifest_path = reference / "formal_run_manifest.json"
    formal_manifest_hash = _sha256_file(formal_manifest_path)
    formal_manifest = _read_json(formal_manifest_path, "formal manifest")
    if formal_manifest.get("status") != "complete":
        raise Phase6SensitivityError("Formal numerical reference is incomplete.")
    formal_hashes = formal_manifest.get("artifact_sha256")
    if not isinstance(formal_hashes, Mapping):
        raise Phase6SensitivityError("Formal artifact hashes are missing.")
    formal_result_hash = formal_hashes.get("experiment_result.csv")
    formal_stage_hash = formal_hashes.get("experiment_stage_result.csv")
    if not isinstance(formal_result_hash, str) or not isinstance(formal_stage_hash, str):
        raise Phase6SensitivityError("Formal artifact hashes are invalid.")
    if formal_manifest.get("phase7c_final_config_sha256") != config_hash:
        raise Phase6SensitivityError("Formal manifest and local rerun config differ.")
    result_path = reference / "experiment_result.csv"
    stage_path = reference / "experiment_stage_result.csv"
    _require_hash(result_path, formal_result_hash, "formal experiment result")
    _require_hash(stage_path, formal_stage_hash, "formal stage result")

    frozen_code = formal_manifest.get("code_sha256")
    if not isinstance(frozen_code, Mapping) or not frozen_code:
        raise Phase6SensitivityError("Formal scientific code lock is missing.")
    scientific_code: dict[str, str] = {}
    for relative, expected in frozen_code.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise Phase6SensitivityError("Formal code hash entry is invalid.")
        path = code_root / relative
        _require_hash(path, expected, f"frozen scientific code {relative}")
        scientific_code[relative] = expected

    config_value = _read_json(config, "Phase 7C final config")
    artifacts = config_value.get("artifact_sha256")
    if not isinstance(artifacts, Mapping):
        raise Phase6SensitivityError("Frozen edge-normalization hash is missing.")
    edge_normalization_hash = artifacts.get("edge_normalization.json")
    if not isinstance(edge_normalization_hash, str):
        raise Phase6SensitivityError("Frozen edge-normalization hash is invalid.")
    normalization_path = config.parent / "edge_normalization.json"
    _require_hash(
        normalization_path, edge_normalization_hash, "edge normalization factors"
    )
    normalization = _read_json(normalization_path, "edge normalization factors")
    datasets = normalization.get("datasets")
    rows = datasets.get(SENSITIVITY_DATASET) if isinstance(datasets, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 178:
        raise Phase6SensitivityError("BPIC2017 must retain all 178 frozen edge scales.")
    scales: dict[tuple[str, str], float] = {}
    selected_count = 0
    tau5_extra = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise Phase6SensitivityError("Normalization row is invalid.")
        edge = (row.get("source"), row.get("target"))
        if any(not isinstance(value, str) or not value for value in edge):
            raise Phase6SensitivityError("Normalization edge is invalid.")
        factor = row.get("normalization_factor_seconds")
        try:
            numeric = float(factor)
        except (TypeError, ValueError) as exc:
            raise Phase6SensitivityError("Normalization factor is not numeric.") from exc
        if not math.isfinite(numeric) or numeric <= 0 or edge in scales:
            raise Phase6SensitivityError("Normalization factors must be unique and positive.")
        scales[edge] = numeric
        selected = row.get("selected_for_experiment") is True
        selected_count += int(selected)
        count = int(row.get("raw_sample_count", 0))
        tau5_extra += int(not selected and count >= 5)
    if selected_count != 138 or tau5_extra <= 0:
        raise Phase6SensitivityError(
            "The all-edge factor set no longer supports the tau=5 sensitivity cell."
        )

    derived_manifest_path = derived / "phase6_bul_analysis_manifest.json"
    derived_manifest_hash = _sha256_file(derived_manifest_path)
    derived_manifest = _read_json(derived_manifest_path, "derived-v2 manifest")
    if (
        derived_manifest.get("status") != "complete"
        or derived_manifest.get("BUL_entropy_role") != "exploratory_robustness_only"
    ):
        raise Phase6SensitivityError("Authoritative derived-v2 policy changed.")
    derived_hashes = derived_manifest.get("artifact_sha256")
    if not isinstance(derived_hashes, Mapping):
        raise Phase6SensitivityError("Derived-v2 hashes are missing.")
    weights_path = derived / "entropy_weights.json"
    expected_weights_hash = derived_hashes.get("entropy_weights.json")
    if not isinstance(expected_weights_hash, str):
        raise Phase6SensitivityError("Entropy-weight hash is missing.")
    _require_hash(weights_path, expected_weights_hash, "fixed entropy weights")
    weights_value = _read_json(weights_path, "fixed entropy weights")
    weights_datasets = weights_value.get("datasets")
    bp_weights = (
        weights_datasets.get(SENSITIVITY_DATASET, {}).get("weights")
        if isinstance(weights_datasets, Mapping)
        else None
    )
    if not isinstance(bp_weights, Mapping) or set(bp_weights) != {
        "structure",
        "temporal",
        "information",
    }:
        raise Phase6SensitivityError("BPIC2017 fixed entropy weights are invalid.")
    checked_weights = {name: float(value) for name, value in bp_weights.items()}
    if not math.isclose(sum(checked_weights.values()), 1.0, abs_tol=1e-12):
        raise Phase6SensitivityError("BPIC2017 entropy weights do not sum to one.")

    result = _reference_key_frame(
        pd.read_csv(result_path, keep_default_na=False), "result"
    )
    stage = _reference_key_frame(
        pd.read_csv(stage_path, keep_default_na=False), "stage"
    )
    reference_rows: dict[tuple[float, int], dict[str, Any]] = {}
    for record in result.to_dict(orient="records"):
        epsilon = float(record["_epsilon"])
        seed = int(record["_seed"])
        condition_id = str(record["condition_id"])
        metadata = _load_reference_checkpoint_metadata(
            reference, formal_manifest, condition_id
        )
        stage_rows = stage.loc[
            stage["_epsilon"].eq(epsilon) & stage["_seed"].eq(seed)
        ]
        stages = {
            str(row["stage"]): {
                field: float(row[field]) for field in _LOSS_FIELDS
            }
            for row in stage_rows.to_dict(orient="records")
        }
        if set(stages) != set(_STAGES):
            raise Phase6SensitivityError("Reference stage set changed.")
        reference_rows[(epsilon, seed)] = {
            "source_condition_id": condition_id,
            "result_record": {
                key: value
                for key, value in record.items()
                if not key.startswith("_")
            },
            "stage_losses": stages,
            "experiment_metadata": metadata,
        }
    if len(reference_rows) != 90:
        raise Phase6SensitivityError("Exactly 90 reference rows are required.")

    source_path = Path(dataset_spec.source_path).resolve()
    source_metadata = {
        "phase7c_config_sha256": config_hash,
        "formal_manifest_sha256": formal_manifest_hash,
        "formal_result_sha256": formal_result_hash,
        "formal_stage_result_sha256": formal_stage_hash,
        "edge_normalization_sha256": edge_normalization_hash,
        "derived_v2_manifest_sha256": derived_manifest_hash,
        "entropy_weights_sha256": expected_weights_hash,
        "dataset_source_sha256": dataset_spec.source_sha256,
        "normalization_factor_count": 178,
        "reference_selected_factor_count": 138,
        "reference_reuse_rows": 90,
    }
    return SensitivitySources(
        config_path=config,
        reference_dir=reference,
        derived_dir=derived,
        source_path=source_path,
        source_sha256=dataset_spec.source_sha256,
        l_max=dataset_spec.l_max,
        t_max=dataset_spec.t_max,
        normalization_scales=scales,
        entropy_weights=checked_weights,
        reference_rows=reference_rows,
        source_metadata=source_metadata,
        scientific_code_sha256=scientific_code,
    )


def _sensitivity_code_hashes(project_root: Path) -> dict[str, str]:
    """Return sensitivity code hashes.

    Inputs:
        project_root: Input value used by this operation.
    Outputs:
        A value of type `dict[str, str]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    result: dict[str, str] = {}
    for relative in _SENSITIVITY_CODE_FILES:
        path = project_root / relative
        if not path.is_file():
            raise Phase6SensitivityError(f"Missing sensitivity code: {path}")
        result[relative] = _sha256_file(path)
    return result


def build_sensitivity_plan(
    sources: SensitivitySources,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Build a source- and code-bound immutable sensitivity execution plan."""

    conditions = build_sensitivity_conditions()
    core = {
        "name": "BUL_phase6_compact_parameter_sensitivity_plan",
        "original_phase": 6,
        "status": "planned",
        "scenario_id": SENSITIVITY_SCENARIO,
        "scope": {
            "dataset": SENSITIVITY_DATASET,
            "method": SENSITIVITY_METHOD,
            "budget_strategy": SENSITIVITY_BUDGET_STRATEGY,
            "epsilons": list(SENSITIVITY_EPSILONS),
            "seeds": list(SENSITIVITY_SEEDS),
            "top_k": list(SENSITIVITY_TOP_K),
            "edge_tau": list(SENSITIVITY_EDGE_TAU),
            "full_cross_product": True,
            "process_quality": False,
            "dp_edge_only": False,
        },
        "counts": {
            "conditions": 810,
            "authenticated_reference_reuse": 90,
            "new_sensitivity_execution": 720,
            "summary_groups": 27,
        },
        "source": dict(sources.source_metadata),
        "scientific_code_sha256": dict(sources.scientific_code_sha256),
        "sensitivity_code_sha256": _sensitivity_code_hashes(
            Path(project_root).resolve()
        ),
        "normalization_policy": (
            "load all 178 Phase7C-frozen BPIC2017 factors; each tau selects "
            "its effective edge subset; never re-estimate by epsilon or seed"
        ),
        "BUL_entropy_policy": {
            "role": "exploratory_robustness_only",
            "weights_fixed_by_dataset": True,
            "recalculated_by_condition": False,
            "calibration_overlap": (
                f"k={REFERENCE_TOP_K},tau={REFERENCE_EDGE_TAU},"
                f"{SENSITIVITY_BUDGET_STRATEGY},seed={SENSITIVITY_SEEDS[0]},"
                f"epsilon in {{{SENSITIVITY_EPSILONS[0]},"
                f"{SENSITIVITY_EPSILONS[-1]}}} is NA"
            ),
        },
        "condition_plan": [condition.to_dict() for condition in conditions],
    }
    return {**core, "plan_sha256": _sha256_bytes(_json_bytes(core))}


def write_sensitivity_plan(plan: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Write sensitivity plan.

    Inputs:
        plan: Input value used by this operation.
        output_dir: Input value used by this operation.
    Outputs:
        A value of type `Path` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    path = Path(output_dir).resolve() / _PLAN_FILE
    _write_immutable(path, _json_bytes(dict(plan)))
    return path


def _condition_config(
    condition: SensitivityCondition,
    sources: SensitivitySources,
) -> SingleExperimentConfig:
    """Create config.

    Inputs:
        condition: Input value used by this operation.
        sources: Input value used by this operation.
    Outputs:
        A value of type `SingleExperimentConfig` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    return SingleExperimentConfig(
        dataset=SENSITIVITY_DATASET,
        method=SENSITIVITY_METHOD,
        epsilon=condition.epsilon,
        seed=condition.seed,
        l_max=sources.l_max,
        t_max=sources.t_max,
        top_k_variants=condition.top_k,
        min_edge_support=condition.edge_tau,
        budget_strategy=SENSITIVITY_BUDGET_STRATEGY,
        dp_edge_mode=None,
        evaluate_process_quality=False,
        conformance_variant=None,
        process_model_noise_threshold=None,
    )


ConditionExecutor = Callable[
    [Sequence[Trace], SingleExperimentConfig, Mapping[tuple[str, str], float]],
    SingleExperimentResult,
]


def _default_executor(
    traces: Sequence[Trace],
    config: SingleExperimentConfig,
    scales: Mapping[tuple[str, str], float],
) -> SingleExperimentResult:
    """Return default executor.

    Inputs:
        traces: Input value used by this operation.
        config: Configuration controlling the operation.
        scales: Input value used by this operation.
    Outputs:
        A value of type `SingleExperimentResult` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    return run_single_experiment(traces, config, normalization_scales=scales)


def _validate_metadata(
    condition: SensitivityCondition,
    sources: SensitivitySources,
    metadata: Mapping[str, Any],
) -> None:
    """Validate metadata.

    Inputs:
        condition: Input value used by this operation.
        sources: Input value used by this operation.
        metadata: Metadata associated with the artifact or experiment condition.
    Outputs:
        None.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    expected_config = _condition_config(condition, sources).to_dict()
    if (
        metadata.get("name") != "BUL_single_experiment"
        or metadata.get("config") != expected_config
        or metadata.get("process_quality") is not None
        or metadata.get("synthetic_log_generation") is not None
    ):
        raise Phase6SensitivityError("Sensitivity checkpoint metadata changed.")
    result = metadata.get("result_record")
    stages = metadata.get("stage_losses")
    release = metadata.get("release")
    analysis = metadata.get("analysis_metadata")
    if not all(isinstance(value, Mapping) for value in (result, stages, release, analysis)):
        raise Phase6SensitivityError("Sensitivity checkpoint sections are missing.")
    if set(stages) != set(_STAGES):
        raise Phase6SensitivityError("Sensitivity checkpoint stages changed.")
    for stage_name in _STAGES:
        losses = stages[stage_name]
        if not isinstance(losses, Mapping) or set(losses) != set(_LOSS_FIELDS):
            raise Phase6SensitivityError("Sensitivity stage loss fields changed.")
        values = [_finite_loss(losses[field], f"{stage_name}.{field}") for field in _LOSS_FIELDS]
        if not math.isclose(values[3], sum(values[:3]) / 3.0, abs_tol=1e-12):
            raise Phase6SensitivityError("Stage BUL differs from the equal mean.")
    for field, expected in {
        "dataset": SENSITIVITY_DATASET,
        "method": SENSITIVITY_METHOD,
        "epsilon": condition.epsilon,
        "seed": condition.seed,
        "budget_strategy": SENSITIVITY_BUDGET_STRATEGY,
        "dp_edge_mode": None,
        "process_quality_evaluated": False,
    }.items():
        if result.get(field) != expected:
            raise Phase6SensitivityError(f"Sensitivity result field changed: {field}")
    total = stages["total"]
    for field in _LOSS_FIELDS:
        if not math.isclose(float(result[field]), float(total[field]), abs_tol=1e-12):
            raise Phase6SensitivityError("Result and total-stage losses differ.")
    if (
        release.get("method") != SENSITIVITY_METHOD
        or release.get("epsilon_total") != condition.epsilon
        or release.get("seed") != condition.seed
        or release.get("budget_strategy") != SENSITIVITY_BUDGET_STRATEGY
        or release.get("composition") != "sequential"
    ):
        raise Phase6SensitivityError("Sensitivity DP release metadata changed.")
    domain = release.get("release_domain")
    if not isinstance(domain, Mapping) or (
        domain.get("top_k_variants") != condition.top_k
        or domain.get("min_edge_support") != condition.edge_tau
    ):
        raise Phase6SensitivityError("Sensitivity release domain changed.")
    if analysis.get("statistical_spaces_frozen_before_noise") is not True:
        raise Phase6SensitivityError("Statistical domains were not frozen before noise.")


def _result_record(
    condition: SensitivityCondition,
    sources: SensitivitySources,
    metadata: Mapping[str, Any],
    *,
    execution_source: str,
    source_condition_id: str | None,
) -> dict[str, Any]:
    """Return result record.

    Inputs:
        condition: Input value used by this operation.
        sources: Input value used by this operation.
        metadata: Metadata associated with the artifact or experiment condition.
        execution_source: Input value used by this operation.
        source_condition_id: Input value used by this operation.
    Outputs:
        A value of type `dict[str, Any]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    result = metadata["result_record"]
    release = metadata["release"]
    analysis = metadata["analysis_metadata"]
    losses = {field: _finite_loss(result[field], field) for field in _LOSS_FIELDS}
    if not math.isclose(
        losses["BUL"],
        (losses["L_structure"] + losses["L_temporal"] + losses["L_information"]) / 3,
        abs_tol=1e-12,
    ):
        raise Phase6SensitivityError("BUL_equal differs from the dimension mean.")
    calibration_overlap = (
        condition.is_reference_reuse
        and condition.seed == 0
        and condition.epsilon in (0.5, 8.0)
    )
    entropy = None
    if not calibration_overlap:
        entropy = calculate_BUL(
            losses["L_structure"],
            losses["L_temporal"],
            losses["L_information"],
            weights=sources.entropy_weights,
        )
    other = release.get("variant_other")
    if not isinstance(other, Mapping):
        raise Phase6SensitivityError("DP Variant OTHER metadata is missing.")
    variant_other_count = float(other.get("count"))
    variant_other_probability = float(other.get("probability"))
    if (
        not math.isfinite(variant_other_count)
        or variant_other_count < 0
        or not math.isfinite(variant_other_probability)
        or not 0.0 <= variant_other_probability <= 1.0
    ):
        raise Phase6SensitivityError("DP Variant OTHER mass is invalid.")
    edge_domain = analysis.get("edge_domain")
    variant_domain = analysis.get("variant_domain")
    temporal_domain = analysis.get("temporal_domain")
    if not all(isinstance(value, list) for value in (edge_domain, variant_domain, temporal_domain)):
        raise Phase6SensitivityError("Sensitivity domain metadata is missing.")
    return {
        "scenario_id": SENSITIVITY_SCENARIO,
        "condition_id": condition.condition_id,
        "source_condition_id": source_condition_id,
        "dataset": SENSITIVITY_DATASET,
        "method": SENSITIVITY_METHOD,
        "budget_strategy": SENSITIVITY_BUDGET_STRATEGY,
        "epsilon": condition.epsilon,
        "seed": condition.seed,
        "top_k_variants": condition.top_k,
        "min_edge_support": condition.edge_tau,
        "L_structure": losses["L_structure"],
        "L_temporal": losses["L_temporal"],
        "L_information": losses["L_information"],
        "BUL_equal": losses["BUL"],
        "BUL_entropy": entropy,
        "BUL_entropy_eligible": not calibration_overlap,
        "BUL_entropy_exclusion_reason": (
            "calibration_overlap" if calibration_overlap else None
        ),
        "variant_other_count": variant_other_count,
        "variant_other_probability": variant_other_probability,
        "activity_domain_size": len(analysis.get("activity_domain", [])),
        "edge_domain_size": len(edge_domain),
        "variant_domain_size_including_other": len(variant_domain),
        "temporal_domain_size": len(temporal_domain),
        "runtime": float(result.get("runtime", 0.0)),
        "execution_source": execution_source,
        "process_quality_evaluated": False,
        "status": "success",
    }


def summarize_sensitivity_records(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Summarize the complete 810 rows into 27 thirty-seed cells.

    Input
    -----
    records:
        Complete sensitivity result records from the fixed grid.

    Output
    ------
    pandas.DataFrame
        One row per Top-k/tau/epsilon cell with seed counts, means, sample
        standard deviations, and explicit entropy-eligible counts.

    Paper experiment step
    ---------------------
    Original Phase 6 robustness reporting. It is descriptive only; Phase 7
    inferential statistics and plotting remain unstarted.
    """

    frame = pd.DataFrame(records)
    required = {
        "condition_id",
        "top_k_variants",
        "min_edge_support",
        "epsilon",
        "seed",
        "L_structure",
        "L_temporal",
        "L_information",
        "BUL_equal",
        "BUL_entropy",
        "variant_other_count",
        "variant_other_probability",
    }
    if len(frame) != 810 or required - set(frame):
        raise Phase6SensitivityError("Sensitivity aggregation requires 810 complete rows.")
    if frame["condition_id"].duplicated().any():
        raise Phase6SensitivityError("Sensitivity result contains duplicate conditions.")
    group_columns = ["top_k_variants", "min_edge_support", "epsilon"]
    measures = [
        "L_structure",
        "L_temporal",
        "L_information",
        "BUL_equal",
        "BUL_entropy",
        "variant_other_count",
        "variant_other_probability",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, sort=True):
        if len(group) != 30 or set(pd.to_numeric(group["seed"])) != set(range(30)):
            raise Phase6SensitivityError("Every sensitivity cell requires seeds 0-29.")
        row = dict(zip(group_columns, keys))
        row["seed_count"] = 30
        for measure in measures:
            values = pd.to_numeric(group[measure], errors="coerce").dropna()
            row[f"{measure}_n"] = int(len(values))
            row[f"{measure}_mean"] = None if values.empty else float(values.mean())
            row[f"{measure}_std"] = (
                None if len(values) < 2 else float(values.std(ddof=1))
            )
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(group_columns, kind="mergesort")
    if len(summary) != 27:
        raise Phase6SensitivityError("Sensitivity summary must contain 27 cells.")
    return summary.reset_index(drop=True)


@dataclass(frozen=True)
class SensitivityExecutionSummary:
    status: str
    expected_conditions: int
    completed_conditions: int
    reused_conditions: int
    executed_this_invocation: int
    skipped_checkpoints: int
    remaining_new_conditions: int
    output_dir: str

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
            "original_phase": 6,
            "status": self.status,
            "expected_conditions": self.expected_conditions,
            "completed_conditions": self.completed_conditions,
            "reused_conditions": self.reused_conditions,
            "executed_this_invocation": self.executed_this_invocation,
            "skipped_checkpoints": self.skipped_checkpoints,
            "remaining_new_conditions": self.remaining_new_conditions,
            "output_dir": self.output_dir,
        }


class SensitivityExperimentRunner:
    """Execute, checkpoint, resume, and aggregate the 720 new conditions."""

    def __init__(
        self,
        sources: SensitivitySources,
        output_dir: str | Path,
        *,
        executor: ConditionExecutor | None = None,
        project_root: str | Path = PROJECT_ROOT,
    ) -> None:
        """Initialize the instance and validate its construction inputs.

        Inputs:
            sources: Input value used by this operation.
            output_dir: Input value used by this operation.
            executor: Input value used by this operation.
            project_root: Input value used by this operation.
        Outputs:
            None.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        self.sources = sources
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.conditions = build_sensitivity_conditions()
        self.plan = build_sensitivity_plan(sources, project_root=project_root)
        self.plan_path = write_sensitivity_plan(self.plan, self.output_dir)
        self.plan_sha256 = str(self.plan["plan_sha256"])
        self.executor = executor or _default_executor
        self.checkpoint_root = self.output_dir / "checkpoints"

    def _checkpoint_path(self, condition: SensitivityCondition) -> Path:
        """Return checkpoint path.

        Inputs:
            condition: Input value used by this operation.
        Outputs:
            A value of type `Path` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return self.checkpoint_root / condition.checkpoint_key / _CHECKPOINT_FILE

    def _load_checkpoint(
        self, condition: SensitivityCondition
    ) -> dict[str, Any] | None:
        """Load checkpoint.

        Inputs:
            condition: Input value used by this operation.
        Outputs:
            A value of type `dict[str, Any] | None` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        path = self._checkpoint_path(condition)
        if not path.exists():
            return None
        checkpoint = _read_json(path, "sensitivity checkpoint")
        if (
            checkpoint.get("name") != "BUL_phase6_sensitivity_checkpoint"
            or checkpoint.get("status") != "complete"
            or checkpoint.get("execution_plan_sha256") != self.plan_sha256
            or checkpoint.get("condition") != condition.to_dict()
            or checkpoint.get("dataset_source_sha256") != self.sources.source_sha256
        ):
            raise Phase6SensitivityError(
                f"Sensitivity checkpoint identity changed: {condition.condition_id}"
            )
        metadata = checkpoint.get("experiment_metadata")
        if not isinstance(metadata, Mapping):
            raise Phase6SensitivityError("Sensitivity checkpoint metadata is missing.")
        _validate_metadata(condition, self.sources, metadata)
        return checkpoint

    def completed_new_conditions(self) -> int:
        """Return completed new conditions.

        Inputs:
            None.
        Outputs:
            A value of type `int` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return sum(
            self._load_checkpoint(condition) is not None
            for condition in self.conditions
            if not condition.is_reference_reuse
        )

    def remaining_new_conditions(self) -> int:
        """Return remaining new conditions.

        Inputs:
            None.
        Outputs:
            A value of type `int` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return 720 - self.completed_new_conditions()

    def _write_progress(
        self,
        *,
        status: str,
        executed: int,
        skipped: int,
        completed_new: int,
    ) -> None:
        """Write progress.

        Inputs:
            status: Input value used by this operation.
            executed: Input value used by this operation.
            skipped: Input value used by this operation.
            completed_new: Input value used by this operation.
        Outputs:
            None.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        payload = {
            "name": "BUL_phase6_sensitivity_progress",
            "original_phase": 6,
            "status": status,
            "execution_plan_sha256": self.plan_sha256,
            "expected_conditions": 810,
            "reused_conditions": 90,
            "completed_new_conditions": completed_new,
            "completed_conditions": 90 + completed_new,
            "remaining_new_conditions": 720 - completed_new,
            "executed_this_invocation": executed,
            "skipped_checkpoints_this_invocation": skipped,
            "authoritative_completion_marker": _FINAL_MANIFEST,
        }
        _atomic_write(self.output_dir / _PROGRESS_FILE, _json_bytes(payload))

    def _execute_condition(
        self,
        condition: SensitivityCondition,
        traces: Sequence[Trace],
    ) -> None:
        """Execute condition.

        Inputs:
            condition: Input value used by this operation.
            traces: Input value used by this operation.
        Outputs:
            None.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        config = _condition_config(condition, self.sources)
        result = self.executor(traces, config, self.sources.normalization_scales)
        if not isinstance(result, SingleExperimentResult):
            raise Phase6SensitivityError("Sensitivity executor returned an invalid result.")
        metadata = result.to_dict()
        _validate_metadata(condition, self.sources, metadata)
        checkpoint = {
            "name": "BUL_phase6_sensitivity_checkpoint",
            "original_phase": 6,
            "status": "complete",
            "execution_plan_sha256": self.plan_sha256,
            "condition": condition.to_dict(),
            "dataset_source_sha256": self.sources.source_sha256,
            "normalization_factors_sha256": self.sources.source_metadata.get(
                "edge_normalization_sha256", "fixture_unlocked"
            ),
            "experiment_metadata": metadata,
            "raw_event_rows_serialized": False,
            "process_quality_executed": False,
        }
        _write_immutable(self._checkpoint_path(condition), _json_bytes(checkpoint))

    def _all_records(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
        """Return all records.

        Inputs:
            None.
        Outputs:
            A value of type `tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        records: list[dict[str, Any]] = []
        stages: list[dict[str, Any]] = []
        checkpoint_hashes: dict[str, str] = {}
        for condition in self.conditions:
            source_condition_id = None
            if condition.is_reference_reuse:
                reused = self.sources.reference_rows[(condition.epsilon, condition.seed)]
                metadata = reused["experiment_metadata"]
                source_condition_id = str(reused["source_condition_id"])
                stage_losses = reused["stage_losses"]
                execution_source = "authenticated_reference_reuse"
            else:
                checkpoint = self._load_checkpoint(condition)
                if checkpoint is None:
                    raise Phase6SensitivityError("Cannot aggregate before every checkpoint exists.")
                metadata = checkpoint["experiment_metadata"]
                stage_losses = metadata["stage_losses"]
                execution_source = "new_sensitivity_execution"
                checkpoint_hashes[condition.condition_id] = _sha256_file(
                    self._checkpoint_path(condition)
                )
            records.append(
                _result_record(
                    condition,
                    self.sources,
                    metadata,
                    execution_source=execution_source,
                    source_condition_id=source_condition_id,
                )
            )
            for stage_name in _STAGES:
                losses = stage_losses[stage_name]
                stages.append(
                    {
                        "condition_id": condition.condition_id,
                        "source_condition_id": source_condition_id,
                        "dataset": SENSITIVITY_DATASET,
                        "method": SENSITIVITY_METHOD,
                        "budget_strategy": SENSITIVITY_BUDGET_STRATEGY,
                        "top_k_variants": condition.top_k,
                        "min_edge_support": condition.edge_tau,
                        "epsilon": condition.epsilon,
                        "seed": condition.seed,
                        "stage": stage_name,
                        **{field: float(losses[field]) for field in _LOSS_FIELDS},
                        "execution_source": execution_source,
                    }
                )
        return records, stages, checkpoint_hashes

    def _aggregate(self, *, executed: int, skipped: int) -> None:
        """Aggregate.

        Inputs:
            executed: Input value used by this operation.
            skipped: Input value used by this operation.
        Outputs:
            None.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        final_path = self.output_dir / _FINAL_MANIFEST
        if final_path.exists():
            manifest = _read_json(final_path, "final sensitivity manifest")
            if (
                manifest.get("name")
                != "BUL_phase6_compact_parameter_sensitivity"
                or manifest.get("status") != "complete"
                or manifest.get("execution_plan_sha256") != self.plan_sha256
            ):
                raise Phase6SensitivityError(
                    "Final sensitivity manifest identity changed."
                )
            artifact_hashes = manifest.get("artifact_sha256")
            if not isinstance(artifact_hashes, Mapping):
                raise Phase6SensitivityError(
                    "Final sensitivity artifact hashes are missing."
                )
            for filename in (_RESULT_FILE, _STAGE_FILE, _SUMMARY_FILE, _PLAN_FILE):
                expected = artifact_hashes.get(filename)
                if not isinstance(expected, str):
                    raise Phase6SensitivityError(
                        f"Final sensitivity hash is missing: {filename}"
                    )
                _require_hash(
                    self.output_dir / filename,
                    expected,
                    f"final sensitivity artifact {filename}",
                )
            return

        records, stages, checkpoint_hashes = self._all_records()
        summary = summarize_sensitivity_records(records)
        result_frame = pd.DataFrame(records).sort_values(
            ["top_k_variants", "min_edge_support", "epsilon", "seed"],
            kind="mergesort",
        )
        stage_frame = pd.DataFrame(stages).sort_values(
            ["top_k_variants", "min_edge_support", "epsilon", "seed", "stage"],
            kind="mergesort",
        )
        artifacts = {
            _RESULT_FILE: _csv_bytes(result_frame),
            _STAGE_FILE: _csv_bytes(stage_frame),
            _SUMMARY_FILE: _csv_bytes(summary),
        }
        for filename, payload in artifacts.items():
            _write_immutable(self.output_dir / filename, payload)
        hashes = {name: _sha256_file(self.output_dir / name) for name in artifacts}
        hashes[_PLAN_FILE] = _sha256_file(self.plan_path)
        manifest = {
            "name": "BUL_phase6_compact_parameter_sensitivity",
            "original_phase": 6,
            "status": "complete",
            "scenario_id": SENSITIVITY_SCENARIO,
            "execution_plan_sha256": self.plan_sha256,
            "counts": {
                "conditions": 810,
                "authenticated_reference_reuse": 90,
                "new_sensitivity_execution": 720,
                "result_rows": 810,
                "stage_rows": 2430,
                "summary_groups": 27,
                "seeds_per_group": 30,
                "failed_conditions": 0,
                "executed_finalizing_invocation": executed,
                "skipped_finalizing_invocation": skipped,
            },
            "source": dict(self.sources.source_metadata),
            "artifacts": {
                "sensitivity_result": _RESULT_FILE,
                "sensitivity_stage_result": _STAGE_FILE,
                "sensitivity_summary": _SUMMARY_FILE,
                "sensitivity_execution_plan": _PLAN_FILE,
            },
            "artifact_sha256": hashes,
            "checkpoint_sha256": checkpoint_hashes,
            "BUL_equal_role": "main",
            "BUL_entropy_role": "exploratory_robustness_only",
            "BUL_entropy_calibration_overlap_rows": 2,
            "normalization_factor_count": 178,
            "raw_xes_read_count_per_execution_invocation": 1,
            "raw_event_rows_serialized": False,
            "process_quality_executed": False,
            "synthetic_logs_saved": False,
            "dp_edge_only_executed": False,
            "not_original_phase7_statistics": True,
        }
        _write_immutable(final_path, _json_bytes(manifest))

    def run(
        self,
        traces: Sequence[Trace] | None,
        *,
        max_new_conditions: int | None = None,
    ) -> SensitivityExecutionSummary:
        """Execute or resume new conditions and aggregate only when complete.

        Input
        -----
        traces:
            Verified BPIC2017 Phase 1 traces. They may be ``None`` only when
            every new checkpoint already exists.
        max_new_conditions:
            Optional positive chunk limit for safe resumable execution.

        Output
        ------
        SensitivityExecutionSummary
            Current completed/reused/skipped/executed counts and status.

        Paper experiment step
        ---------------------
        Original Phase 6 compact parameter sensitivity. No PM4Py, synthetic
        artifact export, inferential statistics, or plotting is performed.
        """

        if max_new_conditions is not None and (
            isinstance(max_new_conditions, bool)
            or not isinstance(max_new_conditions, int)
            or max_new_conditions <= 0
        ):
            raise ValueError("max_new_conditions must be a positive integer.")
        if traces is not None:
            checked_traces = tuple(traces)
            if not checked_traces or any(not isinstance(row, Trace) for row in checked_traces):
                raise Phase6SensitivityError("Execution requires valid Phase 1 traces.")
        else:
            checked_traces = ()

        executed = 0
        skipped = 0
        completed_new = 0
        started = time.perf_counter()
        for condition in self.conditions:
            if condition.is_reference_reuse:
                continue
            if self._load_checkpoint(condition) is not None:
                skipped += 1
                completed_new += 1
                continue
            if max_new_conditions is not None and executed >= max_new_conditions:
                continue
            if not checked_traces:
                raise Phase6SensitivityError(
                    "Unfinished sensitivity conditions require verified BPIC2017 traces."
                )
            self._execute_condition(condition, checked_traces)
            executed += 1
            completed_new += 1
            self._write_progress(
                status="running",
                executed=executed,
                skipped=skipped,
                completed_new=completed_new,
            )

        status = "complete" if completed_new == 720 else "partial"
        if status == "complete":
            self._aggregate(executed=executed, skipped=skipped)
        self._write_progress(
            status=status,
            executed=executed,
            skipped=skipped,
            completed_new=completed_new,
        )
        _ = time.perf_counter() - started
        return SensitivityExecutionSummary(
            status=status,
            expected_conditions=810,
            completed_conditions=90 + completed_new,
            reused_conditions=90,
            executed_this_invocation=executed,
            skipped_checkpoints=skipped,
            remaining_new_conditions=720 - completed_new,
            output_dir=str(self.output_dir),
        )


# Natural release name retained alongside the historical compatibility name.
SensitivityAnalysisError = Phase6SensitivityError


__all__ = [
    "Phase6SensitivityError",
    "SensitivityAnalysisError",
    "SensitivityCondition",
    "SensitivityExecutionSummary",
    "SensitivityExperimentRunner",
    "SensitivitySources",
    "build_sensitivity_conditions",
    "build_sensitivity_plan",
    "load_sensitivity_sources",
    "summarize_sensitivity_records",
    "write_sensitivity_plan",
]

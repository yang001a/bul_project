"""Authenticated, deterministic BUL post-processing for original Phase 6.

This module never reads XES files and never regenerates DP noise.  It derives
fixed-entropy BUL, dimension-level ablations, and cross-dataset descriptive
summaries from the completed 3150-condition numerical reference matrix.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bul_project.bul import calculate_BUL, calculate_fixed_entropy_weights
from bul_project.config import PROJECT_ROOT


DATASET_ORDER = ("BPIC2012", "BPIC2017", "SEPSIS")
DIMENSION_COLUMNS = ("L_structure", "L_temporal", "L_information")
REFERENCE_EPSILONS = (0.5, 8.0)
REFERENCE_STAGES = ("clipping", "dp_noise", "total")

_MAIN_RESULT = "experiment_result.csv"
_MAIN_STAGE_RESULT = "experiment_stage_result.csv"
_MAIN_PLAN = "formal_execution_plan.json"
_MAIN_MANIFEST = "formal_run_manifest.json"
_PHASE7C_CONFIG = "experiment_config_final.json"
_SMOKE_STAGE_RESULT = "smoke/experiment_stage_result.csv"

_ENRICHED_RESULT = "bul_enriched_result.csv"
_ABLATION_RESULT = "dimension_ablation.csv"
_CROSS_DATASET_RESULT = "cross_dataset_summary.csv"
_ENTROPY_WEIGHTS = "entropy_weights.json"
_ANALYSIS_MANIFEST = "phase6_bul_analysis_manifest.json"

_ENTROPY_ELIGIBILITY = "BUL_entropy_eligible"

_ANALYSIS_CODE_FILES = (
    "src/bul_project/config.py",
    "src/bul_project/bul/__init__.py",
    "src/bul_project/bul/_utils.py",
    "src/bul_project/bul/bul_score.py",
    "src/bul_project/experiments/bul_analysis.py",
    "scripts/derive_bul_results.py",
)

_SOURCE_LOCK_FIELDS = (
    "formal_manifest_sha256",
    "formal_result_sha256",
    "formal_stage_result_sha256",
    "formal_plan_file_sha256",
    "formal_internal_plan_sha256",
    "phase7c_config_sha256",
    "phase7c_smoke_stage_sha256",
)


class Phase6DerivedAnalysisError(RuntimeError):
    """Raised when a frozen source or derived-analysis contract is invalid."""


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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
        raise Phase6DerivedAnalysisError(f"Cannot read {role}: {path}") from exc
    if not isinstance(value, dict):
        raise Phase6DerivedAnalysisError(f"{role} must be a JSON object.")
    return value


def _require_mapping(value: object, role: str) -> Mapping[str, Any]:
    """Validate and return a mapping value.

    Inputs:
        value: Candidate value to validate or transform.
        role: Input value used by this operation.
    Outputs:
        A value of type `Mapping[str, Any]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    if not isinstance(value, Mapping):
        raise Phase6DerivedAnalysisError(f"{role} must be a mapping.")
    return value


def _require_sha256(value: object, role: str) -> str:
    """Validate and return a SHA-256 digest.

    Inputs:
        value: Candidate value to validate or transform.
        role: Input value used by this operation.
    Outputs:
        A value of type `str` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Phase6DerivedAnalysisError(f"{role} is not a lowercase SHA-256.")
    return value


def _verify_file_hash(path: Path, expected: object, role: str) -> str:
    """Verify file hash.

    Inputs:
        path: Path to the input or output artifact.
        expected: Input value used by this operation.
        role: Input value used by this operation.
    Outputs:
        A value of type `str` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    declared = _require_sha256(expected, role)
    if not path.is_file():
        raise Phase6DerivedAnalysisError(f"Missing frozen source: {path}")
    actual = _sha256_file(path)
    if actual != declared:
        raise Phase6DerivedAnalysisError(
            f"{role} differs from its frozen SHA-256: {path}"
        )
    return actual


def build_rerun_source_lock(
    reference_dir: str | Path,
    phase7c_dir: str | Path,
) -> dict[str, str]:
    """Create a source lock from the completed local rerun artifacts.

    Inputs:
        reference_dir: Directory containing the completed main experiment.
        phase7c_dir: Directory containing the local rerun parameter artifacts.
    Outputs:
        SHA-256 values used to authenticate the downstream derivation inputs.
    Paper role:
        Reproducible provenance binding; this does not alter DP or BUL methods.
    """
    reference = Path(reference_dir).resolve()
    phase7c = Path(phase7c_dir).resolve()
    manifest_path = reference / _MAIN_MANIFEST
    manifest = _read_json(manifest_path, "formal numerical manifest")
    hashes = _require_mapping(manifest.get("artifact_sha256"), "formal hashes")
    plan = _read_json(reference / _MAIN_PLAN, "formal execution plan")
    config_path = phase7c / _PHASE7C_CONFIG
    config = _read_json(config_path, "Phase 7C final config")
    smoke = _require_mapping(config.get("smoke_evidence"), "smoke_evidence")
    smoke_hashes = _require_mapping(
        smoke.get("artifact_sha256"), "smoke artifact hashes"
    )
    lock = {
        "formal_manifest_sha256": _sha256_file(manifest_path),
        "formal_result_sha256": _require_sha256(
            hashes.get(_MAIN_RESULT), f"formal hash for {_MAIN_RESULT}"
        ),
        "formal_stage_result_sha256": _require_sha256(
            hashes.get(_MAIN_STAGE_RESULT), f"formal hash for {_MAIN_STAGE_RESULT}"
        ),
        "formal_plan_file_sha256": _require_sha256(
            hashes.get(_MAIN_PLAN), f"formal hash for {_MAIN_PLAN}"
        ),
        "formal_internal_plan_sha256": _require_sha256(
            plan.get("plan_sha256"), "formal internal plan SHA-256"
        ),
        "phase7c_config_sha256": _sha256_file(config_path),
        "phase7c_smoke_stage_sha256": _require_sha256(
            smoke_hashes.get("experiment_stage_result"),
            "smoke stage-result SHA-256",
        ),
    }
    return lock


def _write_immutable(path: Path, payload: bytes) -> Path:
    """Write an artifact without overwriting conflicting content.

    Inputs:
        path: Path to the input or output artifact.
        payload: Structured content to validate, serialize, or write.
    Outputs:
        A value of type `Path` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise Phase6DerivedAnalysisError(
                f"Immutable derived artifact already exists with different bytes: {path}"
            )
        return path
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
                raise Phase6DerivedAnalysisError(
                    "Concurrent immutable artifact differs from this run: "
                    f"{path}"
                )
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


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


def _as_loss_matrix(frame: pd.DataFrame, role: str) -> np.ndarray:
    """Convert component losses to a validated loss matrix.

    Inputs:
        frame: Tabular data processed by the operation.
        role: Input value used by this operation.
    Outputs:
        A value of type `np.ndarray` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    try:
        matrix = frame.loc[:, DIMENSION_COLUMNS].astype(float).to_numpy()
    except (KeyError, TypeError, ValueError) as exc:
        raise Phase6DerivedAnalysisError(f"{role} losses are not numeric.") from exc
    if (
        matrix.ndim != 2
        or matrix.shape[1] != 3
        or not np.all(np.isfinite(matrix))
        or np.any(matrix < 0.0)
        or np.any(matrix > 1.0)
    ):
        raise Phase6DerivedAnalysisError(f"{role} losses must be finite in [0,1].")
    return matrix


def _validate_main_result(frame: pd.DataFrame) -> None:
    """Validate main result.

    Inputs:
        frame: Tabular data processed by the operation.
    Outputs:
        None.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    required = {
        "dataset",
        "epsilon",
        "method",
        "seed",
        "budget_strategy",
        "dp_edge_mode",
        "status",
        "condition_id",
        "scenario_id",
        "top_k_variants",
        "min_edge_support",
        "BUL",
        *DIMENSION_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise Phase6DerivedAnalysisError(
            f"The numerical result is missing columns: {sorted(missing)!r}"
        )
    if len(frame) != 3156 or frame["condition_id"].duplicated().any():
        raise Phase6DerivedAnalysisError(
            "The numerical result must contain 3156 unique rows."
        )
    if set(frame["dataset"]) != set(DATASET_ORDER):
        raise Phase6DerivedAnalysisError("The numerical dataset scope changed.")
    if set(frame["status"]) != {"success"}:
        raise Phase6DerivedAnalysisError("The numerical result contains failures.")
    expected_counts = {
        "Raw": 3,
        "ClippedRaw": 3,
        "DP-full": 2250,
        "DP-edge-only": 900,
    }
    if frame["method"].value_counts().to_dict() != expected_counts:
        raise Phase6DerivedAnalysisError("The numerical method counts changed.")
    if set(frame["scenario_id"]) != {"reference_k100_tau10"}:
        raise Phase6DerivedAnalysisError("The numerical reference scenario changed.")
    if set(pd.to_numeric(frame["top_k_variants"], errors="coerce")) != {100}:
        raise Phase6DerivedAnalysisError("The numerical Top-k reference changed.")
    if set(pd.to_numeric(frame["min_edge_support"], errors="coerce")) != {10}:
        raise Phase6DerivedAnalysisError("The numerical edge-tau reference changed.")

    baseline = frame[frame["method"].isin(("Raw", "ClippedRaw"))]
    expected_baselines = {
        (dataset, method)
        for dataset in DATASET_ORDER
        for method in ("Raw", "ClippedRaw")
    }
    if (
        set(zip(baseline["dataset"], baseline["method"])) != expected_baselines
        or len(baseline) != len(expected_baselines)
    ):
        raise Phase6DerivedAnalysisError("The six numerical baselines changed.")

    full = frame[frame["method"].eq("DP-full")].copy()
    full["_epsilon"] = pd.to_numeric(full["epsilon"], errors="coerce")
    full["_seed"] = pd.to_numeric(full["seed"], errors="coerce")
    full_keys = set(
        zip(
            full["dataset"],
            full["_epsilon"],
            full["_seed"],
            full["budget_strategy"],
        )
    )
    expected_full_keys = {
        (dataset, epsilon, float(seed), strategy)
        for dataset in DATASET_ORDER
        for epsilon in (0.5, 1.0, 2.0, 4.0, 8.0)
        for seed in range(30)
        for strategy in (
            "uniform",
            "activity_focus",
            "edge_focus",
            "variant_focus",
            "time_focus",
        )
    }
    if full_keys != expected_full_keys or not full["dp_edge_mode"].eq("NA").all():
        raise Phase6DerivedAnalysisError("The DP-full Cartesian grid changed.")

    edge = frame[frame["method"].eq("DP-edge-only")].copy()
    edge["_epsilon"] = pd.to_numeric(edge["epsilon"], errors="coerce")
    edge["_seed"] = pd.to_numeric(edge["seed"], errors="coerce")
    edge_keys = set(
        zip(edge["dataset"], edge["_epsilon"], edge["_seed"], edge["dp_edge_mode"])
    )
    expected_edge_keys = {
        (dataset, epsilon, float(seed), mode)
        for dataset in DATASET_ORDER
        for epsilon in (0.5, 1.0, 2.0, 4.0, 8.0)
        for seed in range(30)
        for mode in ("component", "mechanism")
    }
    if edge_keys != expected_edge_keys or not edge["budget_strategy"].eq("NA").all():
        raise Phase6DerivedAnalysisError("The DP-edge-only Cartesian grid changed.")

    complete = frame["method"].isin(("Raw", "ClippedRaw", "DP-full"))
    matrix = _as_loss_matrix(frame.loc[complete], "complete numerical")
    bul = pd.to_numeric(frame.loc[complete, "BUL"], errors="coerce").to_numpy()
    expected_bul = matrix.mean(axis=1)
    if not np.all(np.isfinite(bul)) or not np.allclose(
        bul, expected_bul, rtol=0.0, atol=1e-12
    ):
        raise Phase6DerivedAnalysisError(
            "Stored BUL is not the equal mean of the three loss dimensions."
        )
    edge_only = frame["method"].eq("DP-edge-only")
    edge_structure = pd.to_numeric(
        frame.loc[edge_only, "L_structure"], errors="coerce"
    ).to_numpy()
    if (
        not np.all(np.isfinite(edge_structure))
        or np.any(edge_structure < 0.0)
        or np.any(edge_structure > 1.0)
    ):
        raise Phase6DerivedAnalysisError(
            "DP-edge-only L_structure must be finite in [0,1]."
        )
    if not (
        frame.loc[edge_only, "L_temporal"].eq("NA").all()
        and frame.loc[edge_only, "L_information"].eq("NA").all()
        and frame.loc[edge_only, "BUL"].eq("NA").all()
    ):
        raise Phase6DerivedAnalysisError(
            "DP-edge-only missing BUL dimensions must remain NA."
        )


def calculate_frozen_entropy_weights(
    smoke_stage: pd.DataFrame,
    *,
    source_sha256: str,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Fit one dataset-specific entropy weight vector from pre-formal smoke rows.

    Input
    -----
    smoke_stage:
        Authenticated locally generated Phase 7C stage-result table. Only
        DP-full, uniform, seed 0, epsilon 0.5/8 rows are used.
    source_sha256:
        SHA-256 of the local Phase 7C smoke stage table.

    Output
    ------
    tuple
        Dataset-to-weight mapping and complete reference-matrix audit metadata.

    Paper experiment step
    ---------------------
    Phase 5/6 robustness aggregation: calculate ``BUL_entropy`` weights once
    per dataset on the frozen Phase 7C smoke calibration matrix, then reuse
    them unchanged. Formal rows numerically overlapping this calibration set
    are excluded from entropy-weight evaluation by ``enrich_bul_results``.
    """

    required = {
        "dataset",
        "epsilon",
        "method",
        "seed",
        "budget_strategy",
        "stage",
        *DIMENSION_COLUMNS,
    }
    missing = required - set(smoke_stage.columns)
    if missing:
        raise Phase6DerivedAnalysisError(
            f"The smoke stage table is missing columns: {sorted(missing)!r}"
        )
    weights_by_dataset: dict[str, dict[str, float]] = {}
    dataset_metadata: dict[str, Any] = {}
    stage_order = {stage: index for index, stage in enumerate(REFERENCE_STAGES)}
    for dataset in DATASET_ORDER:
        selected = smoke_stage.loc[
            smoke_stage["dataset"].eq(dataset)
            & smoke_stage["method"].eq("DP-full")
            & smoke_stage["budget_strategy"].eq("uniform")
            & pd.to_numeric(smoke_stage["seed"], errors="coerce").eq(0)
            & smoke_stage["epsilon"].astype(float).isin(REFERENCE_EPSILONS)
            & smoke_stage["stage"].isin(REFERENCE_STAGES)
        ].copy()
        selected["_stage_order"] = selected["stage"].map(stage_order)
        selected = selected.sort_values(
            ["epsilon", "_stage_order"], kind="mergesort"
        ).drop(columns="_stage_order")
        reference_keys = set(
            zip(selected["epsilon"].astype(float), selected["stage"])
        )
        expected_reference_keys = {
            (epsilon, stage)
            for epsilon in REFERENCE_EPSILONS
            for stage in REFERENCE_STAGES
        }
        if len(selected) != 6 or reference_keys != expected_reference_keys:
            raise Phase6DerivedAnalysisError(
                f"{dataset} entropy reference matrix must contain exactly 6 rows."
            )
        matrix = _as_loss_matrix(selected, f"{dataset} entropy reference")
        weights = calculate_fixed_entropy_weights(matrix)
        weights_by_dataset[dataset] = weights
        rows = []
        for record in selected.to_dict(orient="records"):
            rows.append(
                {
                    "epsilon": float(record["epsilon"]),
                    "stage": str(record["stage"]),
                    "L_structure": float(record["L_structure"]),
                    "L_temporal": float(record["L_temporal"]),
                    "L_information": float(record["L_information"]),
                }
            )
        dataset_metadata[dataset] = {
            "weights": weights,
            "reference_matrix_rows": rows,
            "reference_row_count": 6,
        }
    metadata = {
        "name": "BUL_fixed_entropy_weights",
        "status": "frozen",
        "scope": "dataset_specific_and_constant_across_epsilon_strategy_seed",
        "source_role": "phase7c_smoke_calibration_matrix",
        "source_file": _SMOKE_STAGE_RESULT,
        "source_sha256": _require_sha256(source_sha256, "smoke source SHA-256"),
        "selection_freeze_timing": "post_execution_pre_phase7_statistical_analysis",
        "predeclared_in_phase7c_final_config": False,
        "overlap_policy": (
            "matching formal total rows and their ClippedRaw calibration "
            "baseline are ineligible for BUL_entropy evaluation"
        ),
        "selection": {
            "method": "DP-full",
            "budget_strategy": "uniform",
            "seed": 0,
            "epsilon_values": list(REFERENCE_EPSILONS),
            "stages": list(REFERENCE_STAGES),
            "rows_per_dataset": 6,
            "recalculated_per_epsilon_or_seed": False,
        },
        "datasets": dataset_metadata,
    }
    return weights_by_dataset, metadata


def enrich_bul_results(
    main_result: pd.DataFrame,
    weights_by_dataset: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Add fixed-entropy BUL and dimension-ablation scores without rerunning DP.

    Input
    -----
    main_result:
        Authenticated 3156-row formal result including six baselines.
    weights_by_dataset:
        One pre-frozen structure/temporal/information weight vector per dataset.

    Output
    ------
    pandas.DataFrame
        Original result columns plus ``BUL_equal``, ``BUL_entropy`` and three
        leave-one-dimension-out scores. DP-edge-only composite fields remain NA.

    Paper experiment step
    ---------------------
    Phase 5/6 BUL robustness and dimension-level ablation. Removing a dimension
    renormalizes the two retained equal weights to one half each.
    """

    _validate_main_result(main_result)
    if set(weights_by_dataset) != set(DATASET_ORDER):
        raise Phase6DerivedAnalysisError("Entropy weights must cover all datasets.")
    frame = main_result.copy()
    for column in (*DIMENSION_COLUMNS, "BUL"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in (
        "BUL_equal",
        "BUL_entropy",
        "BUL_wo_structure",
        "BUL_wo_temporal",
        "BUL_wo_information",
        "entropy_weight_structure",
        "entropy_weight_temporal",
        "entropy_weight_information",
    ):
        frame[column] = np.nan

    numeric_epsilon = pd.to_numeric(frame["epsilon"], errors="coerce")
    numeric_seed = pd.to_numeric(frame["seed"], errors="coerce")
    calibration_overlap = frame["method"].eq("ClippedRaw") | (
        frame["method"].eq("DP-full")
        & frame["budget_strategy"].eq("uniform")
        & numeric_seed.eq(0)
        & numeric_epsilon.isin(REFERENCE_EPSILONS)
    )
    complete = frame.loc[:, DIMENSION_COLUMNS].notna().all(axis=1)
    frame[_ENTROPY_ELIGIBILITY] = complete & ~calibration_overlap

    for index in frame.index[complete]:
        dataset = str(frame.at[index, "dataset"])
        weights = weights_by_dataset[dataset]
        structure = float(frame.at[index, "L_structure"])
        temporal = float(frame.at[index, "L_temporal"])
        information = float(frame.at[index, "L_information"])
        equal = calculate_BUL(structure, temporal, information)
        if not math.isclose(equal, float(frame.at[index, "BUL"]), abs_tol=1e-12):
            raise Phase6DerivedAnalysisError("Stored and recomputed BUL_equal differ.")
        frame.at[index, "BUL_equal"] = equal
        if bool(frame.at[index, _ENTROPY_ELIGIBILITY]):
            frame.at[index, "BUL_entropy"] = calculate_BUL(
                structure, temporal, information, weights=weights
            )
        frame.at[index, "BUL_wo_structure"] = calculate_BUL(
            structure,
            temporal,
            information,
            weights={"structure": 0.0, "temporal": 0.5, "information": 0.5},
        )
        frame.at[index, "BUL_wo_temporal"] = calculate_BUL(
            structure,
            temporal,
            information,
            weights={"structure": 0.5, "temporal": 0.0, "information": 0.5},
        )
        frame.at[index, "BUL_wo_information"] = calculate_BUL(
            structure,
            temporal,
            information,
            weights={"structure": 0.5, "temporal": 0.5, "information": 0.0},
        )
        if bool(frame.at[index, _ENTROPY_ELIGIBILITY]):
            for dimension in ("structure", "temporal", "information"):
                frame.at[index, f"entropy_weight_{dimension}"] = float(
                    weights[dimension]
                )

    edge_only = frame["method"].eq("DP-edge-only")
    derived = [
        "BUL_equal",
        "BUL_entropy",
        "BUL_wo_structure",
        "BUL_wo_temporal",
        "BUL_wo_information",
    ]
    if frame.loc[edge_only, derived].notna().any().any():
        raise Phase6DerivedAnalysisError(
            "DP-edge-only composite and ablation scores must remain NA."
        )
    if int(calibration_overlap.sum()) != 9:
        raise Phase6DerivedAnalysisError(
            "Entropy calibration overlap must contain 6 formal total rows "
            "and 3 ClippedRaw rows."
        )
    if frame.loc[calibration_overlap, "BUL_entropy"].notna().any():
        raise Phase6DerivedAnalysisError(
            "Entropy calibration rows cannot evaluate BUL_entropy."
        )
    return frame


def build_dimension_ablation(enriched: pd.DataFrame) -> pd.DataFrame:
    """Return the 2250 DP-full rows needed for dimension-level ablation."""

    selected = enriched.loc[enriched["method"].eq("DP-full")].copy()
    if len(selected) != 2250:
        raise Phase6DerivedAnalysisError("Dimension ablation requires 2250 DP-full rows.")
    columns = [
        "scenario_id",
        "condition_id",
        "dataset",
        "epsilon",
        "seed",
        "budget_strategy",
        *DIMENSION_COLUMNS,
        "BUL_equal",
        "BUL_entropy",
        "BUL_wo_structure",
        "BUL_wo_temporal",
        "BUL_wo_information",
    ]
    return selected.loc[:, columns].sort_values(
        ["dataset", "epsilon", "budget_strategy", "seed"], kind="mergesort"
    ).reset_index(drop=True)


def build_cross_dataset_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 30-seed BUL dimensions by dataset, epsilon, and mechanism."""

    selected = enriched.loc[
        enriched["method"].isin(("DP-full", "DP-edge-only"))
    ].copy()
    measures = [
        *DIMENSION_COLUMNS,
        "BUL_equal",
        "BUL_entropy",
        "BUL_wo_structure",
        "BUL_wo_temporal",
        "BUL_wo_information",
    ]
    group_columns = [
        "dataset",
        "method",
        "epsilon",
        "budget_strategy",
        "dp_edge_mode",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in selected.groupby(group_columns, sort=True, dropna=False):
        if len(group) != 30 or group["seed"].nunique() != 30:
            raise Phase6DerivedAnalysisError(
                "Every cross-dataset summary group must contain 30 unique seeds."
            )
        row = dict(zip(group_columns, keys))
        row["seed_count"] = 30
        for measure in measures:
            values = pd.to_numeric(group[measure], errors="coerce").dropna()
            row[f"{measure}_n"] = int(len(values))
            row[f"{measure}_mean"] = (
                np.nan if values.empty else float(values.mean())
            )
            row[f"{measure}_std"] = (
                np.nan if len(values) < 2 else float(values.std(ddof=1))
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    if len(summary) != 105:
        raise Phase6DerivedAnalysisError(
            "Cross-dataset summary must contain 105 mechanism groups."
        )
    return summary.sort_values(group_columns, kind="mergesort").reset_index(drop=True)


def _authenticate_sources(
    reference_dir: Path,
    phase7c_dir: Path,
    source_lock: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Authenticate sources.

    Inputs:
        reference_dir: Input value used by this operation.
        phase7c_dir: Input value used by this operation.
        source_lock: Input value used by this operation.
    Outputs:
        A value of type `tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    manifest_path = reference_dir / _MAIN_MANIFEST
    _verify_file_hash(
        manifest_path,
        source_lock.get("formal_manifest_sha256"),
        "externally locked formal manifest SHA-256",
    )
    manifest = _read_json(manifest_path, "formal numerical manifest")
    counts = _require_mapping(manifest.get("counts"), "formal counts")
    if (
        manifest.get("status") != "complete"
        or counts.get("completed_conditions") != 3150
        or counts.get("failed_conditions") != 0
        or counts.get("experiment_rows") != 3156
        or counts.get("stage_rows") != 9450
    ):
        raise Phase6DerivedAnalysisError("The formal numerical reference is incomplete.")
    hashes = _require_mapping(manifest.get("artifact_sha256"), "formal hashes")
    locked_artifacts = {
        _MAIN_RESULT: "formal_result_sha256",
        _MAIN_STAGE_RESULT: "formal_stage_result_sha256",
        _MAIN_PLAN: "formal_plan_file_sha256",
    }
    for filename, lock_name in locked_artifacts.items():
        locked_hash = _require_sha256(source_lock.get(lock_name), lock_name)
        if hashes.get(filename) != locked_hash:
            raise Phase6DerivedAnalysisError(
                f"Formal manifest and external lock differ for {filename}."
            )
        _verify_file_hash(
            reference_dir / filename,
            locked_hash,
            f"formal {filename} SHA-256",
        )

    plan = _read_json(reference_dir / _MAIN_PLAN, "formal execution plan")
    internal_plan_hash = _require_sha256(
        plan.get("plan_sha256"), "formal internal plan SHA-256"
    )
    if internal_plan_hash != _require_sha256(
        source_lock.get("formal_internal_plan_sha256"),
        "locked formal internal plan SHA-256",
    ) or manifest.get("execution_plan_sha256") != internal_plan_hash:
        raise Phase6DerivedAnalysisError("Formal internal plan hash changed.")
    plan_core = dict(plan)
    plan_core.pop("plan_sha256", None)
    if hashlib.sha256(_json_bytes(plan_core)).hexdigest() != internal_plan_hash:
        raise Phase6DerivedAnalysisError("Formal execution plan content is invalid.")

    config_path = phase7c_dir / _PHASE7C_CONFIG
    locked_config_hash = _require_sha256(
        source_lock.get("phase7c_config_sha256"), "locked Phase 7C config SHA-256"
    )
    if manifest.get("phase7c_final_config_sha256") != locked_config_hash:
        raise Phase6DerivedAnalysisError("Formal manifest and Phase 7C lock differ.")
    config_hash = _verify_file_hash(
        config_path,
        locked_config_hash,
        "Phase 7C final config SHA-256",
    )
    config = _read_json(config_path, "Phase 7C final config")
    if config.get("status") != "frozen":
        raise Phase6DerivedAnalysisError("Phase 7C configuration is not frozen.")
    smoke = _require_mapping(config.get("smoke_evidence"), "smoke_evidence")
    if (
        smoke.get("passed") is not True
        or smoke.get("seed") != 0
        or smoke.get("budget_strategy") != "uniform"
        or smoke.get("epsilons") != [0.5, 8.0]
        or smoke.get("dp_conditions") != 12
    ):
        raise Phase6DerivedAnalysisError("Phase 7C smoke evidence changed.")
    smoke_hashes = _require_mapping(
        smoke.get("artifact_sha256"), "smoke artifact hashes"
    )
    smoke_stage_path = phase7c_dir / _SMOKE_STAGE_RESULT
    locked_smoke_hash = _require_sha256(
        source_lock.get("phase7c_smoke_stage_sha256"),
        "locked Phase 7C smoke stage SHA-256",
    )
    if smoke_hashes.get("experiment_stage_result") != locked_smoke_hash:
        raise Phase6DerivedAnalysisError("Smoke manifest and source lock differ.")
    smoke_stage_hash = _verify_file_hash(
        smoke_stage_path,
        locked_smoke_hash,
        "Phase 7C smoke stage SHA-256",
    )
    main = pd.read_csv(reference_dir / _MAIN_RESULT, keep_default_na=False)
    smoke_stage = pd.read_csv(smoke_stage_path, keep_default_na=False)
    _validate_main_result(main)
    source_metadata = {
        "formal_manifest_file": _MAIN_MANIFEST,
        "formal_manifest_sha256": _sha256_file(manifest_path),
        "formal_result_file": _MAIN_RESULT,
        "formal_result_sha256": hashes[_MAIN_RESULT],
        "formal_stage_result_sha256": hashes[_MAIN_STAGE_RESULT],
        "formal_execution_plan_sha256": hashes[_MAIN_PLAN],
        "formal_internal_plan_sha256": internal_plan_hash,
        "phase7c_final_config_file": _PHASE7C_CONFIG,
        "phase7c_final_config_sha256": config_hash,
        "phase7c_smoke_stage_file": _SMOKE_STAGE_RESULT,
        "phase7c_smoke_stage_sha256": smoke_stage_hash,
    }
    return main, smoke_stage, source_metadata


def run_phase6_derived_analysis(
    reference_dir: str | Path,
    phase7c_dir: str | Path,
    output_dir: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    source_lock: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Authenticate sources, derive BUL artifacts, and commit a final manifest."""

    reference = Path(reference_dir).resolve()
    phase7c = Path(phase7c_dir).resolve()
    output = Path(output_dir).resolve()
    if (
        output == reference
        or output.is_relative_to(reference)
        or output == phase7c
        or output.is_relative_to(phase7c)
    ):
        raise Phase6DerivedAnalysisError(
            "Derived output must be outside the immutable numerical reference."
        )
    lock = dict(
        build_rerun_source_lock(reference, phase7c)
        if source_lock is None
        else source_lock
    )
    if set(lock) != set(_SOURCE_LOCK_FIELDS):
        raise Phase6DerivedAnalysisError("Source lock fields are incomplete or changed.")
    main, smoke_stage, source_metadata = _authenticate_sources(
        reference, phase7c, lock
    )
    weights, weight_metadata = calculate_frozen_entropy_weights(
        smoke_stage,
        source_sha256=source_metadata["phase7c_smoke_stage_sha256"],
    )
    enriched = enrich_bul_results(main, weights)
    ablation = build_dimension_ablation(enriched)
    cross_dataset = build_cross_dataset_summary(enriched)

    root = Path(project_root).resolve()
    code_hashes = {}
    for relative in _ANALYSIS_CODE_FILES:
        path = root / relative
        if not path.is_file():
            raise Phase6DerivedAnalysisError(f"Missing analysis code file: {path}")
        code_hashes[relative] = _sha256_file(path)
    runtime = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    artifacts = {
        _ENRICHED_RESULT: _csv_bytes(enriched),
        _ABLATION_RESULT: _csv_bytes(ablation),
        _CROSS_DATASET_RESULT: _csv_bytes(cross_dataset),
        _ENTROPY_WEIGHTS: _json_bytes(weight_metadata),
    }
    for filename, payload in artifacts.items():
        _write_immutable(output / filename, payload)
    artifact_hashes = {
        filename: _sha256_file(output / filename) for filename in artifacts
    }

    for relative in _ANALYSIS_CODE_FILES:
        path = root / relative
        if _sha256_file(path) != code_hashes[relative]:
            raise Phase6DerivedAnalysisError(
                f"Analysis code changed during execution: {path}"
            )
    plan_core = {
        "source": source_metadata,
        "code_sha256": code_hashes,
        "runtime": runtime,
        "entropy_reference": weight_metadata["selection"],
        "entropy_overlap_exclusion": {
            "formal_total_rows": 6,
            "clippedraw_rows": 3,
            "eligibility_column": _ENTROPY_ELIGIBILITY,
        },
        "dimension_ablation": {
            "groups": [
                "full",
                "without_structure",
                "without_temporal",
                "without_information",
            ],
            "removed_dimension_weight": 0.0,
            "retained_dimension_weights": [0.5, 0.5],
        },
        "cross_dataset_role": "descriptive_summary_not_absolute_BUL_ranking",
    }
    plan_sha256 = hashlib.sha256(_json_bytes(plan_core)).hexdigest()
    manifest = {
        "name": "BUL_phase6_derived_core_analysis",
        "original_phase": 6,
        "status": "complete",
        "plan_sha256": plan_sha256,
        "source": source_metadata,
        "code_sha256": code_hashes,
        "runtime": runtime,
        "completed_tasks": [
            "BUL_equal_verified",
            "fixed_calibration_BUL_entropy_with_overlap_exclusion",
            "dimension_ablation",
            "cross_dataset_descriptive_summary",
        ],
        "pending_tasks": ["compact_top_k_edge_tau_sensitivity"],
        "counts": {
            "source_rows": len(main),
            "dp_conditions": 3150,
            "enriched_rows": len(enriched),
            "dimension_ablation_rows": len(ablation),
            "cross_dataset_summary_rows": len(cross_dataset),
            "entropy_weight_datasets": len(weights),
            "entropy_eligible_rows": int(enriched[_ENTROPY_ELIGIBILITY].sum()),
            "entropy_calibration_excluded_rows": int(
                (~enriched[_ENTROPY_ELIGIBILITY] & enriched["method"].ne("DP-edge-only")).sum()
            ),
        },
        "artifacts": {filename: filename for filename in artifacts},
        "artifact_sha256": artifact_hashes,
        "dp_edge_only_composite_fields": "NA",
        "raw_log_access": False,
        "dp_noise_regenerated": False,
        "process_quality_in_BUL": False,
        "not_original_phase7_statistics": True,
        "supersedes_non_authoritative_output": "results/phase6/bul_core_derived_v1",
        "BUL_entropy_role": "exploratory_robustness_only",
        "BUL_entropy_cross_dataset_absolute_ranking_allowed": False,
        "BUL_entropy_formal_stage_calibration_inference_allowed": False,
    }
    _write_immutable(output / _ANALYSIS_MANIFEST, _json_bytes(manifest))
    return manifest


# Natural public names for the release artifact. Historical names remain
# import-compatible for frozen manifests and existing analysis notebooks.
DerivedAnalysisError = Phase6DerivedAnalysisError
run_derived_analysis = run_phase6_derived_analysis


__all__ = [
    "DerivedAnalysisError",
    "Phase6DerivedAnalysisError",
    "build_cross_dataset_summary",
    "build_dimension_ablation",
    "build_rerun_source_lock",
    "calculate_frozen_entropy_weights",
    "enrich_bul_results",
    "run_derived_analysis",
    "run_phase6_derived_analysis",
]

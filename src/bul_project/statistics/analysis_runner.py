"""Authenticated end-to-end runner for original Phase 7.

The runner is read-only with respect to Phase 1-6.  It verifies upstream
artifact hashes, computes statistics from the frozen tables, and writes only
new Phase 7 tables, figures, scope metadata, and a reproducibility manifest.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import scipy

from bul_project.config import PROJECT_ROOT

from .plotting import build_figure_payloads, figure_contract
from .statistical_tests import (
    MAIN_METRICS,
    Phase7StatisticsError,
    build_ablation_ranking_consistency,
    build_ablation_summary,
    build_descriptive_summary,
    build_friedman_results,
    build_sensitivity_summary,
    build_spearman_trends,
    build_stage_summary,
    build_wilcoxon_results,
)


_DERIVED_MANIFEST = "phase6_bul_analysis_manifest.json"
_SENSITIVITY_MANIFEST = "phase6_sensitivity_manifest.json"
_FORMAL_MANIFEST = "formal_run_manifest.json"
_EXTERNAL_MANIFEST = "process_quality_manifest.json"

_CODE_FILES = (
    "src/bul_project/statistics/__init__.py",
    "src/bul_project/statistics/statistical_tests.py",
    "src/bul_project/statistics/plotting.py",
    "src/bul_project/statistics/analysis_runner.py",
    "scripts/run_statistical_analysis.py",
)


class Phase7AnalysisError(RuntimeError):
    """Raised when source authentication or Phase 7 execution fails."""


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest of a file.

    Inputs:
        path: Path to the input or output artifact.
    Outputs:
        A value of type `str` produced by the operation.
    Paper role:
        Statistical inference and publication-quality figure generation.
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
        Statistical inference and publication-quality figure generation.
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


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize tabular content to deterministic UTF-8 CSV bytes.

    Inputs:
        frame: Tabular data processed by the operation.
    Outputs:
        A value of type `bytes` produced by the operation.
    Paper role:
        Statistical inference and publication-quality figure generation.
    """
    stream = io.StringIO(newline="")
    frame.to_csv(stream, index=False, na_rep="NA", lineterminator="\n")
    return stream.getvalue().encode("utf-8")


def _read_json(path: Path, role: str) -> dict[str, Any]:
    """Read and parse a JSON document.

    Inputs:
        path: Path to the input or output artifact.
        role: Input value used by this operation.
    Outputs:
        A value of type `dict[str, Any]` produced by the operation.
    Paper role:
        Statistical inference and publication-quality figure generation.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase7AnalysisError(f"Cannot read {role}: {path}") from exc
    if not isinstance(value, dict):
        raise Phase7AnalysisError(f"{role} must be a JSON object.")
    return value


def _write_immutable(path: Path, payload: bytes) -> Path:
    """Write an artifact without overwriting conflicting content.

    Inputs:
        path: Path to the input or output artifact.
        payload: Structured content to validate, serialize, or write.
    Outputs:
        A value of type `Path` produced by the operation.
    Paper role:
        Statistical inference and publication-quality figure generation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise Phase7AnalysisError(
                f"Immutable Phase 7 artifact already differs: {path}"
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
                raise Phase7AnalysisError(
                    f"Concurrent Phase 7 artifact differs: {path}"
                )
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _verify_declared_artifact(
    directory: Path,
    manifest: dict[str, Any],
    filename: str,
    role: str,
) -> tuple[Path, str]:
    """Verify declared artifact.

    Inputs:
        directory: Input value used by this operation.
        manifest: Input value used by this operation.
        filename: Input value used by this operation.
        role: Input value used by this operation.
    Outputs:
        A value of type `tuple[Path, str]` produced by the operation.
    Paper role:
        Statistical inference and publication-quality figure generation.
    """
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or filename not in hashes:
        raise Phase7AnalysisError(f"{role} manifest does not declare {filename}.")
    expected = hashes[filename]
    if not isinstance(expected, str) or len(expected) != 64:
        raise Phase7AnalysisError(f"{role} declares an invalid SHA-256 for {filename}.")
    path = directory / filename
    if not path.is_file():
        raise Phase7AnalysisError(f"Missing {role} artifact: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise Phase7AnalysisError(f"{role} artifact SHA-256 changed: {path}")
    return path, actual


def _load_authenticated_sources(
    *,
    derived_dir: Path,
    reference_dir: Path,
    sensitivity_dir: Path,
    external_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load authenticated sources.

    Inputs:
        derived_dir: Input value used by this operation.
        reference_dir: Input value used by this operation.
        sensitivity_dir: Input value used by this operation.
        external_dir: Input value used by this operation.
    Outputs:
        A value of type `tuple[dict[str, pd.DataFrame], dict[str, Any]]` produced by the operation.
    Paper role:
        Statistical inference and publication-quality figure generation.
    """
    derived_manifest_path = derived_dir / _DERIVED_MANIFEST
    formal_manifest_path = reference_dir / _FORMAL_MANIFEST
    sensitivity_manifest_path = sensitivity_dir / _SENSITIVITY_MANIFEST
    external_manifest_path = external_dir / _EXTERNAL_MANIFEST

    derived_manifest = _read_json(derived_manifest_path, "Phase 6 derived manifest")
    formal_manifest = _read_json(formal_manifest_path, "Phase 6 formal manifest")
    sensitivity_manifest = _read_json(
        sensitivity_manifest_path, "Phase 6 sensitivity manifest"
    )
    # PM4Py validation is a representative, optional feasibility check.  It is
    # deliberately not a prerequisite for the BUL core statistics or figures.
    external_manifest = (
        _read_json(external_manifest_path, "Phase 6 external-validity manifest")
        if external_manifest_path.is_file()
        else None
    )
    for role, manifest in (
        ("derived", derived_manifest),
        ("formal", formal_manifest),
        ("sensitivity", sensitivity_manifest),
    ):
        if manifest.get("status") != "complete":
            raise Phase7AnalysisError(f"Phase 6 {role} source is not complete.")
    if external_manifest is not None and external_manifest.get("status") != "complete":
        raise Phase7AnalysisError("Phase 6 external source is not complete.")

    source_specs = {
        "enriched": (derived_dir, derived_manifest, "bul_enriched_result.csv", "derived"),
        "ablation": (derived_dir, derived_manifest, "dimension_ablation.csv", "derived"),
        "stage": (reference_dir, formal_manifest, "experiment_stage_result.csv", "formal"),
        "sensitivity": (
            sensitivity_dir,
            sensitivity_manifest,
            "sensitivity_result.csv",
            "sensitivity",
        ),
    }
    if external_manifest is not None:
        source_specs["external"] = (
            external_dir,
            external_manifest,
            "process_quality_result.csv",
            "external-validity",
        )
    frames: dict[str, pd.DataFrame] = {}
    source_hashes: dict[str, Any] = {}
    for name, (directory, manifest, filename, role) in source_specs.items():
        path, digest = _verify_declared_artifact(directory, manifest, filename, role)
        try:
            frames[name] = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            raise Phase7AnalysisError(f"Cannot read authenticated source: {path}") from exc
        source_hashes[name] = {
            "path": str(path.resolve()),
            "sha256": digest,
            "rows": int(len(frames[name])),
        }
    source_hashes["manifests"] = {
        "derived": _sha256_file(derived_manifest_path),
        "formal": _sha256_file(formal_manifest_path),
        "sensitivity": _sha256_file(sensitivity_manifest_path),
        "external": (
            _sha256_file(external_manifest_path)
            if external_manifest is not None
            else None
        ),
    }
    source_hashes["external_validation"] = {
        "status": "complete" if external_manifest is not None else "not_run_optional",
        "manifest_path": str(external_manifest_path.resolve()),
    }
    return frames, source_hashes


def assess_external_validity(frame: pd.DataFrame | None) -> dict[str, Any]:
    """Determine whether the reduced representative PM4Py sample supports rho.

    Input: authenticated three-condition external-validity table. Output:
    machine-readable eligibility decision. Paper step: Phase 7 scope guard; no
    correlation is fabricated when a process-quality variable has zero variance.
    """

    if frame is None:
        return {
            "role": "representative_external_validity_check_only",
            "row_count": 0,
            "quality_unique_counts": {},
            "spearman_correlation_eligible": False,
            "status": "not_run_optional",
            "reason": (
                "Representative PM4Py external validation was not run; it is "
                "outside the reduced BUL core scope."
            ),
            "process_quality_in_BUL": False,
            "core_inferential_statistics_included": False,
        }

    required = {"BUL_equal", "fitness", "precision", "generalization"}
    missing = required - set(frame.columns)
    if missing:
        raise Phase7AnalysisError(
            f"External-validity result is missing: {sorted(missing)!r}"
        )
    quality_unique = {
        metric: int(pd.to_numeric(frame[metric], errors="coerce").nunique(dropna=True))
        for metric in ("fitness", "precision", "generalization")
    }
    zero_variance = [metric for metric, count in quality_unique.items() if count < 2]
    eligible = len(frame) >= 5 and not zero_variance
    if zero_variance:
        status = "not_estimable_zero_variance"
        reason = (
            "The representative three-condition check produced constant process-"
            "quality values; Spearman correlation is undefined."
        )
    elif len(frame) < 5:
        status = "not_estimable_representative_scope"
        reason = "The reduced representative scope is too small for formal correlation."
    else:
        status = "eligible_but_outside_reduced_core_scope"
        reason = "Correlation was not requested in the reduced Phase 6/7 core scope."
    return {
        "role": "representative_external_validity_check_only",
        "row_count": int(len(frame)),
        "quality_unique_counts": quality_unique,
        "spearman_correlation_eligible": bool(eligible),
        "status": status,
        "reason": reason,
        "process_quality_in_BUL": False,
        "core_inferential_statistics_included": False,
    }


def _analysis_scope(
    *, bootstrap_resamples: int, bootstrap_seed: int, external: dict[str, Any]
) -> dict[str, Any]:
    """Return analysis scope.

    Inputs:
        bootstrap_resamples: Input value used by this operation.
        bootstrap_seed: Input value used by this operation.
        external: Input value used by this operation.
    Outputs:
        A value of type `dict[str, Any]` produced by the operation.
    Paper role:
        Statistical inference and publication-quality figure generation.
    """
    return {
        "original_phase": 7,
        "primary_endpoint": "BUL_equal",
        "main_metrics": list(MAIN_METRICS),
        "exploratory_endpoint": "BUL_entropy",
        "BUL_entropy_role": "exploratory_robustness_only",
        "BUL_entropy_in_inferential_tests": False,
        "pairing_block": "seed",
        "seed_scope": list(range(30)),
        "alpha": 0.05,
        "confidence_level": 0.95,
        "bootstrap": {
            "method": "percentile",
            "resamples": int(bootstrap_resamples),
            "seed": int(bootstrap_seed),
            "trend_CI_resampling_unit": "seed_cluster",
        },
        "wilcoxon": {
            "alternative": "two-sided",
            "zero_method": "wilcox",
            "effect_size": "matched-pairs rank-biserial correlation",
            "effect_direction": "positive means first condition has larger loss",
        },
        "friedman": "repeated seed blocks",
        "holm": "within each declared post-hoc family",
        "spearman_epsilon_trend": {
            "point_estimate": "five epsilon-level means",
            "p_value_unit": "five epsilon levels",
            "p_value_method": "exact two-sided exhaustive permutation (5! = 120)",
            "CI": "seed-cluster bootstrap",
            "strict_monotonicity_forced": False,
        },
        "dp_edge_only": {
            "analyzed_metric": "L_structure",
            "L_temporal": "NA",
            "L_information": "NA",
            "BUL": "NA",
        },
        "stage_losses_are_additive": False,
        "cross_dataset_absolute_BUL_ranking_allowed": False,
        "external_validity": external,
        "raw_log_access": False,
        "dp_noise_regenerated": False,
    }


def run_phase7_analysis(
    *,
    derived_dir: Path = PROJECT_ROOT / "results" / "phase6" / "bul_core_derived_v2",
    reference_dir: Path = PROJECT_ROOT / "results" / "phase8a" / "reference_k100_tau10_validator_v2",
    sensitivity_dir: Path = PROJECT_ROOT / "results" / "phase6" / "bul_sensitivity_bpic2017_v2",
    external_dir: Path = PROJECT_ROOT / "results" / "phase6" / "external_validity_bpic2017_representative3_sampled_final",
    output_dir: Path = PROJECT_ROOT / "results" / "phase7" / "statistical_analysis_v3",
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Execute authenticated Phase 7 statistics and paper-figure generation.

    Input: four frozen Phase 6 artifact directories and bootstrap configuration.
    Output: immutable statistics tables, six PNG and six PDF figures, scope
    JSON, and final manifest. Paper step: complete original Phase 7 statistical
    analysis/plotting.
    """

    if int(bootstrap_resamples) < 1:
        raise Phase7AnalysisError("bootstrap_resamples must be positive.")
    if int(bootstrap_seed) not in range(30):
        raise Phase7AnalysisError("bootstrap_seed must be in 0-29.")
    frames, source_hashes = _load_authenticated_sources(
        derived_dir=Path(derived_dir),
        reference_dir=Path(reference_dir),
        sensitivity_dir=Path(sensitivity_dir),
        external_dir=Path(external_dir),
    )
    try:
        tables = {
            "descriptive_summary.csv": build_descriptive_summary(
                frames["enriched"],
                n_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "friedman_results.csv": build_friedman_results(frames["enriched"]),
            "wilcoxon_holm_results.csv": build_wilcoxon_results(frames["enriched"]),
            "spearman_epsilon_trends.csv": build_spearman_trends(
                frames["enriched"],
                n_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "stage_loss_summary.csv": build_stage_summary(
                frames["stage"],
                n_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "dimension_ablation_summary.csv": build_ablation_summary(
                frames["ablation"],
                n_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "ablation_ranking_consistency.csv": build_ablation_ranking_consistency(
                frames["ablation"],
                n_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "parameter_sensitivity_summary.csv": build_sensitivity_summary(
                frames["sensitivity"],
                n_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
        }
        figure_payloads = build_figure_payloads(
            descriptive_summary=tables["descriptive_summary.csv"],
            stage_summary=tables["stage_loss_summary.csv"],
            ablation_summary=tables["dimension_ablation_summary.csv"],
            sensitivity_summary=tables["parameter_sensitivity_summary.csv"],
        )
    except (Phase7StatisticsError, ValueError, KeyError) as exc:
        raise Phase7AnalysisError(f"Phase 7 analysis contract failed: {exc}") from exc

    external_assessment = assess_external_validity(frames.get("external"))
    scope = _analysis_scope(
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        external=external_assessment,
    )
    payloads: dict[str, bytes] = {
        name: _csv_bytes(table) for name, table in tables.items()
    }
    payloads.update(figure_payloads)
    payloads["analysis_scope.json"] = _json_bytes(scope)
    payloads["figure_contract.json"] = _json_bytes(figure_contract())
    payloads["external_validity_assessment.json"] = _json_bytes(external_assessment)
    output_dir = Path(output_dir)
    for name, payload in payloads.items():
        _write_immutable(output_dir / name, payload)

    code_hashes: dict[str, str] = {}
    for relative in _CODE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise Phase7AnalysisError(f"Missing Phase 7 code file: {path}")
        code_hashes[relative] = _sha256_file(path)
    artifact_hashes = {
        name: _sha256_file(output_dir / name) for name in sorted(payloads)
    }
    manifest = {
        "name": "BUL_original_phase7_statistical_analysis",
        "original_phase": 7,
        "status": "complete",
        "source": source_hashes,
        "configuration": {
            "bootstrap_resamples": int(bootstrap_resamples),
            "bootstrap_seed": int(bootstrap_seed),
            "alpha": 0.05,
            "confidence_level": 0.95,
        },
        "counts": {
            **{name: int(len(table)) for name, table in tables.items()},
            "figure_files": len(figure_payloads),
        },
        "artifacts": {name: name for name in sorted(payloads)},
        "artifact_sha256": artifact_hashes,
        "code_sha256": code_hashes,
        "method_guards": {
            "primary_endpoint": "BUL_equal",
            "BUL_entropy_role": "exploratory_robustness_only",
            "dp_edge_only_structure_only": True,
            "strict_epsilon_monotonicity_forced": False,
            "stage_additivity_claimed": False,
            "cross_dataset_absolute_BUL_ranking_allowed": False,
            "spearman_p_value_method": "exact_two_sided_permutation",
            "PM4Py_correlation_status": external_assessment["status"],
            "raw_log_access": False,
            "dp_noise_regenerated": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "supersedes_outputs": [
            "results/phase7/statistical_analysis_v1",
            "results/phase7/statistical_analysis_v2",
            "results/statistical_analysis",
        ],
        "supersession_reason": (
            "The SCI revision replaces asymptotic Spearman p-values for five "
            "epsilon levels with exact exhaustive permutation p-values and "
            "upgrades all figures for accessible journal publication."
        ),
    }
    _write_immutable(output_dir / "phase7_analysis_manifest.json", _json_bytes(manifest))
    return manifest


# Natural release API names. Historical names remain import-compatible because
# they also occur in frozen artifact provenance.
StatisticalAnalysisError = Phase7AnalysisError
run_statistical_analysis = run_phase7_analysis

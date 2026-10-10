"""Phase 8A frozen-config formal execution with fail-closed checkpoints.

This module is an execution layer only. It validates the immutable Phase 7C
bundle, delegates every numerical condition to the already tested Phase 7A
runner, and never changes a privacy mechanism, loss metric, or frozen domain.
"""

from __future__ import annotations

import gc
import csv
import hashlib
import json
import math
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from bul_project.behavior import Edge
from bul_project.config import (
    DP_EDGE_COMPONENT_SHARE,
    PRIVACY_COMPONENTS,
    PROJECT_ROOT,
    TEMPORAL_BUDGET_RATIOS,
)
from bul_project.preprocessing.models import Trace
from bul_project.provenance import (
    ProvenanceError,
    build_code_hashes as build_release_code_hashes,
)

from .batch_run import (
    BatchCondition,
    BatchExperimentConfig,
    BatchExperimentRunner,
    DatasetExperimentInput,
)
from .parameter_freeze import (
    Phase7CDatasetInput,
    build_evaluation_log_metadata,
)
from .single_run import (
    SingleExperimentResult,
    run_single_experiment,
    write_single_experiment_artifacts,
)


_CHECKPOINT_MANIFEST = "checkpoint_manifest.json"
_PLAN_FILENAME = "formal_execution_plan.json"
_PROGRESS_FILENAME = "formal_run_progress.json"
_FINAL_MANIFEST = "formal_run_manifest.json"
_STAGE_ORDER = ("clipping", "dp_noise", "total")
_LOSS_FIELDS = {"L_structure", "L_temporal", "L_information", "BUL"}
_QUALITY_FIELDS = {"fitness", "precision", "generalization"}
_EXPECTED_EDGE_ONLY_MISSING = {
    "L_temporal",
    "L_information",
    "BUL",
    "fitness",
    "precision",
    "generalization",
}


class FormalRunIntegrityError(RuntimeError):
    """Raised when a frozen input, plan, or committed checkpoint changed."""


class FormalConditionExecutionError(RuntimeError):
    """Raised after one failed condition has been recorded atomically."""


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


def current_runtime_metadata() -> dict[str, str | None]:
    """Return the runtime fields frozen by Phase 7C.

    Output
    ------
    dict
        Python and dependency versions used for strict formal-run validation.

    Paper experiment step
    ---------------------
    Rerun environment lock: a formal run must use the exact runtime recorded
    by the local frozen parameter configuration.
    """

    return {
        "python": platform.python_version(),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "scipy": _package_version("scipy"),
        "pm4py": _package_version("pm4py"),
        "lxml": _package_version("lxml"),
    }


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
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_write_dataframe(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a tabular artifact atomically.

    Inputs:
        path: Path to the input or output artifact.
        rows: Input value used by this operation.
    Outputs:
        None.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    pd.DataFrame(rows).to_csv(
        temporary,
        index=False,
        encoding="utf-8",
        na_rep="NA",
    )
    temporary.replace(path)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    """Read and parse a JSON document.

    Inputs:
        path: Path to the input or output artifact.
        label: Input value used by this operation.
    Outputs:
        A value of type `dict[str, Any]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalRunIntegrityError(f"Cannot read valid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise FormalRunIntegrityError(f"{label} must contain one JSON object.")
    return value


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    """Validate and return a mapping value.

    Inputs:
        value: Candidate value to validate or transform.
        label: Input value used by this operation.
    Outputs:
        A value of type `dict[str, Any]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    if not isinstance(value, dict):
        raise FormalRunIntegrityError(f"{label} must be a JSON object.")
    return value


def _require_sequence(value: Any, label: str) -> list[Any]:
    """Validate and return a sequence value.

    Inputs:
        value: Candidate value to validate or transform.
        label: Input value used by this operation.
    Outputs:
        A value of type `list[Any]` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    if not isinstance(value, list):
        raise FormalRunIntegrityError(f"{label} must be a JSON array.")
    return value


def _validate_sha256(value: Any, label: str) -> str:
    """Validate sha256.

    Inputs:
        value: Candidate value to validate or transform.
        label: Input value used by this operation.
    Outputs:
        A value of type `str` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise FormalRunIntegrityError(f"{label} must be a SHA-256 hex digest.")
    return value.lower()


def build_code_manifest(
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, str]:
    """Hash all Python sources that can affect the approved pipeline.

    Input
    -----
    project_root:
        BUL project source root.

    Output
    ------
    dict
        Stable relative-path to SHA-256 mapping, excluding tests and caches.

    Paper experiment step
    ---------------------
    Phase 8A provenance lock.  The plan hash changes if implementation code
    changes between interrupted formal-run invocations.
    """

    try:
        return build_release_code_hashes(project_root)
    except ProvenanceError as exc:
        raise FormalRunIntegrityError(str(exc)) from exc


@dataclass(frozen=True)
class FrozenEvaluationLogSpec:
    evaluation_log_id: str
    sha256: str
    event_count: int
    case_count: int


@dataclass(frozen=True)
class FrozenDatasetExecutionSpec:
    dataset: str
    source_sha256: str
    source_path: str | None
    l_max: int
    t_max: float
    top_k_variants: int
    min_edge_support: int
    normalization_scales: Mapping[Edge, float]
    evaluation_log: FrozenEvaluationLogSpec | None


@dataclass(frozen=True)
class FrozenExecutionSpec:
    """Validated Phase 7C bundle used by the Phase 8A reference scenario."""

    config_path: Path
    config_sha256: str
    phase7c_manifest_sha256: str
    runtime: Mapping[str, str | None]
    batch_config: BatchExperimentConfig
    datasets: tuple[FrozenDatasetExecutionSpec, ...]
    reference_top_k: int
    reference_edge_tau: int
    main_dp_edge_mode: str
    appendix_dp_edge_mode: str | None
    process_quality: Mapping[str, Any] | None
    artifact_sha256: Mapping[str, str]

    @property
    def scenario_id(self) -> str:
        """Return scenario id.

        Inputs:
            None.
        Outputs:
            A value of type `str` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return (
            f"reference_k{self.reference_top_k}_tau{self.reference_edge_tau}"
        )

    @property
    def dataset_names(self) -> tuple[str, ...]:
        """Return dataset names.

        Inputs:
            None.
        Outputs:
            A value of type `tuple[str, ...]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return tuple(item.dataset for item in self.datasets)

    def verify_source_files(self) -> None:
        """Verify every available frozen source path before reading XES data."""

        for item in self.datasets:
            if item.source_path is None:
                continue
            path = Path(item.source_path).resolve()
            if not path.is_file():
                raise FormalRunIntegrityError(
                    f"Frozen source file is unavailable for {item.dataset!r}: {path}"
                )
            actual = _sha256_file(path)
            if actual != item.source_sha256:
                raise FormalRunIntegrityError(
                    f"Source SHA-256 changed for dataset {item.dataset!r}."
                )

    def build_dataset_inputs(
        self,
        phase7c_inputs: Sequence[Phase7CDatasetInput],
        *,
        include_process_quality_events: bool = False,
    ) -> tuple[DatasetExperimentInput, ...]:
        """Bridge verified Phase 1 data into unchanged Phase 7B contracts.

        Input
        -----
        phase7c_inputs:
            One explicit Phase 1 trace/input contract per frozen dataset.
        include_process_quality_events:
            Whether verified evaluation rows are retained in returned inputs.
            Phase 8A keeps this false because the formal PM4Py condition scope
            is not yet frozen.

        Output
        ------
        tuple[DatasetExperimentInput, ...]
            Reference-scenario inputs with only Phase 7C-frozen parameters.

        Paper experiment step
        ---------------------
        Phase 8A input lock: source and evaluation hashes are checked before any
        epsilon/seed condition can execute. No bound or domain is recomputed.
        """

        supplied = tuple(phase7c_inputs)
        if any(not isinstance(value, Phase7CDatasetInput) for value in supplied):
            raise TypeError(
                "phase7c_inputs must contain Phase7CDatasetInput objects."
            )
        by_name = {value.dataset: value for value in supplied}
        if len(by_name) != len(supplied):
            raise ValueError("phase7c_inputs contains duplicate datasets.")
        if set(by_name) != set(self.dataset_names):
            raise FormalRunIntegrityError(
                "Phase 1 inputs must exactly match frozen Phase 7C datasets."
            )

        bridged: list[DatasetExperimentInput] = []
        for spec in self.datasets:
            source = by_name[spec.dataset]
            if source.source_sha256 != spec.source_sha256:
                raise FormalRunIntegrityError(
                    f"Phase 1 source hash changed for dataset {spec.dataset!r}."
                )
            if spec.source_path is not None:
                if source.source_path is None:
                    raise FormalRunIntegrityError(
                        f"Phase 1 source path is missing for {spec.dataset!r}."
                    )
                if Path(source.source_path).resolve() != Path(spec.source_path).resolve():
                    raise FormalRunIntegrityError(
                        f"Phase 1 source path changed for dataset {spec.dataset!r}."
                    )

            evaluation_events = None
            evaluation_log_id = None
            if spec.evaluation_log is not None:
                if source.evaluation_events is None or source.evaluation_log_id is None:
                    raise FormalRunIntegrityError(
                        f"Frozen evaluation log is missing for {spec.dataset!r}."
                    )
                current = build_evaluation_log_metadata(
                    source.evaluation_events,
                    source.evaluation_log_id,
                )
                expected = spec.evaluation_log
                if (
                    current.evaluation_log_id != expected.evaluation_log_id
                    or current.sha256 != expected.sha256
                    or current.event_count != expected.event_count
                    or current.case_count != expected.case_count
                ):
                    raise FormalRunIntegrityError(
                        f"Evaluation log changed for dataset {spec.dataset!r}."
                    )
                if include_process_quality_events:
                    evaluation_events = source.evaluation_events
                    evaluation_log_id = source.evaluation_log_id
            elif source.evaluation_events is not None:
                raise FormalRunIntegrityError(
                    f"Unfrozen evaluation log supplied for {spec.dataset!r}."
                )

            bridged.append(
                DatasetExperimentInput(
                    dataset=spec.dataset,
                    traces=source.traces,
                    l_max=spec.l_max,
                    t_max=spec.t_max,
                    top_k_variants=spec.top_k_variants,
                    min_edge_support=spec.min_edge_support,
                    normalization_scales=spec.normalization_scales,
                    evaluation_events=evaluation_events,
                    evaluation_log_id=evaluation_log_id,
                )
            )
        return tuple(bridged)


def _parse_evaluation_log(value: Any, dataset: str) -> FrozenEvaluationLogSpec | None:
    """Parse evaluation log.

    Inputs:
        value: Candidate value to validate or transform.
        dataset: Dataset identifier or dataset-specific input.
    Outputs:
        A value of type `FrozenEvaluationLogSpec | None` produced by the operation.
    Paper role:
        Frozen experiment orchestration and result derivation.
    """
    if value is None:
        return None
    payload = _require_mapping(value, f"datasets[{dataset}].evaluation_log")
    log_id = payload.get("evaluation_log_id")
    if not isinstance(log_id, str) or not log_id:
        raise FormalRunIntegrityError("Frozen evaluation_log_id is invalid.")
    event_count = payload.get("event_count")
    case_count = payload.get("case_count")
    if (
        isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count <= 0
        or isinstance(case_count, bool)
        or not isinstance(case_count, int)
        or case_count <= 0
    ):
        raise FormalRunIntegrityError("Frozen evaluation-log counts are invalid.")
    return FrozenEvaluationLogSpec(
        evaluation_log_id=log_id,
        sha256=_validate_sha256(
            payload.get("evaluation_log_sha256"),
            "evaluation_log_sha256",
        ),
        event_count=event_count,
        case_count=case_count,
    )


def load_frozen_execution_spec(
    final_config_path: str | Path,
) -> FrozenExecutionSpec:
    """Load and fail-closed validate one immutable Phase 7C bundle.

    Input
    -----
    final_config_path:
        Path to the locally bound ``experiment_config_final.json``.

    Output
    ------
    FrozenExecutionSpec
        Typed reference-scenario parameters and the complete numerical matrix.

    Paper experiment step
    ---------------------
    Formal execution is forbidden unless the locally generated final config,
    manifest, diagnostics, normalization factors, runtime, and Phase 7C smoke
    evidence remain byte/hash consistent.
    """

    config_path = Path(final_config_path).resolve()
    if not config_path.is_file():
        raise FormalRunIntegrityError(f"Final config does not exist: {config_path}")
    config_bytes = config_path.read_bytes()
    config_sha256 = _sha256_bytes(config_bytes)
    payload = _read_json(config_path, "Phase 7C final config")
    if (
        payload.get("name") != "BUL_experiment_config_final"
        or payload.get("phase") != "7C"
        or payload.get("status") != "frozen"
    ):
        raise FormalRunIntegrityError(
            "Formal execution requires the frozen Phase 7C final config."
        )
    smoke = _require_mapping(payload.get("smoke_evidence"), "smoke_evidence")
    if smoke.get("passed") is not True:
        raise FormalRunIntegrityError("Phase 7C smoke evidence is not passing.")
    phase7c_dir = config_path.parent
    manifest_path = phase7c_dir / "phase7c_manifest.json"
    manifest = _read_json(manifest_path, "Phase 7C manifest")
    if (
        manifest.get("name") != "BUL_Phase7C_manifest"
        or manifest.get("phase") != "7C"
        or manifest.get("status") != "frozen"
    ):
        raise FormalRunIntegrityError("Phase 7C manifest is not frozen.")
    manifest_hashes = _require_mapping(
        manifest.get("artifact_sha256"),
        "phase7c_manifest.artifact_sha256",
    )
    expected_config_hash = _validate_sha256(
        manifest_hashes.get(config_path.name),
        "Phase 7C final-config manifest hash",
    )
    if expected_config_hash != config_sha256:
        raise FormalRunIntegrityError(
            "Phase 7C manifest does not match the final config bytes."
        )

    artifact_hashes_raw = _require_mapping(
        payload.get("artifact_sha256"),
        "artifact_sha256",
    )
    artifact_hashes: dict[str, str] = {}
    for filename, expected_value in artifact_hashes_raw.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise FormalRunIntegrityError("Frozen artifact filename is invalid.")
        expected = _validate_sha256(
            expected_value,
            f"artifact_sha256[{filename}]",
        )
        path = phase7c_dir / filename
        if not path.is_file() or _sha256_file(path) != expected:
            raise FormalRunIntegrityError(
                f"Frozen Phase 7C artifact changed: {filename}"
            )
        artifact_hashes[filename] = expected
        manifest_expected = _validate_sha256(
            manifest_hashes.get(filename),
            f"Phase 7C manifest hash for {filename}",
        )
        if manifest_expected != expected:
            raise FormalRunIntegrityError(
                f"Phase 7C manifest and final config disagree for {filename}."
            )

    smoke_hashes_raw = _require_mapping(
        smoke.get("artifact_sha256"),
        "smoke_evidence.artifact_sha256",
    )
    smoke_filenames = {
        "batch_manifest": "batch_manifest.json",
        "experiment_result": "experiment_result.csv",
        "experiment_run_metadata": "experiment_run_metadata.jsonl",
        "experiment_stage_result": "experiment_stage_result.csv",
    }
    if set(smoke_hashes_raw) != set(smoke_filenames):
        raise FormalRunIntegrityError("Phase 7C smoke artifact set changed.")
    for logical_name, filename in smoke_filenames.items():
        expected = _validate_sha256(
            smoke_hashes_raw[logical_name],
            f"smoke_evidence.artifact_sha256[{logical_name}]",
        )
        smoke_path = phase7c_dir / "smoke" / filename
        if not smoke_path.is_file() or _sha256_file(smoke_path) != expected:
            raise FormalRunIntegrityError(
                f"Frozen Phase 7C smoke artifact changed: {filename}"
            )

    frozen_runtime = _require_mapping(payload.get("runtime"), "runtime")
    current_runtime = current_runtime_metadata()
    if frozen_runtime != current_runtime:
        raise FormalRunIntegrityError(
            "Current runtime does not match the Phase 7C frozen runtime."
        )
    if manifest.get("runtime") != frozen_runtime:
        raise FormalRunIntegrityError(
            "Phase 7C manifest runtime differs from the final config."
        )
    privacy_boundary = _require_mapping(
        payload.get("privacy_boundary"),
        "privacy_boundary",
    )
    if privacy_boundary.get("conditional_dp_claim") is not True:
        raise FormalRunIntegrityError(
            "The required public-benchmark conditional-DP boundary is absent."
        )

    policy = _require_mapping(payload.get("policy"), "policy")
    reference_top_k = policy.get("reference_top_k")
    reference_edge_tau = policy.get("reference_edge_tau")
    if (
        isinstance(reference_top_k, bool)
        or not isinstance(reference_top_k, int)
        or reference_top_k <= 0
        or isinstance(reference_edge_tau, bool)
        or not isinstance(reference_edge_tau, int)
        or reference_edge_tau <= 0
    ):
        raise FormalRunIntegrityError("Frozen reference k/tau values are invalid.")
    if policy.get("variant_coverage_role") != "diagnostic_only":
        raise FormalRunIntegrityError("Variant coverage role changed after freeze.")
    if policy.get("synthetic_consistency_scope") != "process_quality_datasets_only":
        raise FormalRunIntegrityError("Synthetic consistency scope changed.")

    robustness = _require_mapping(
        payload.get("parameter_robustness"),
        "parameter_robustness",
    )
    if (
        robustness.get("reference_top_k") != reference_top_k
        or robustness.get("reference_edge_tau") != reference_edge_tau
        or robustness.get("full_cross_product_required") is not True
    ):
        raise FormalRunIntegrityError("Parameter-robustness metadata is inconsistent.")

    matrix = _require_mapping(payload.get("experiment_matrix"), "experiment_matrix")
    batch_config = BatchExperimentConfig(
        epsilons=tuple(_require_sequence(matrix.get("epsilon_values"), "epsilon_values")),
        seeds=tuple(_require_sequence(matrix.get("seeds"), "seeds")),
        dp_full_strategies=tuple(
            _require_sequence(matrix.get("budget_strategies"), "budget_strategies")
        ),
        dp_edge_modes=tuple(
            _require_sequence(matrix.get("dp_edge_modes"), "dp_edge_modes")
        ),
        process_quality_selection=None,
    )
    split = _require_mapping(
        matrix.get("time_budget_internal_split"),
        "time_budget_internal_split",
    )
    if set(split) != set(TEMPORAL_BUDGET_RATIOS) or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in split.values()
    ) or any(
        not math.isclose(
            float(split[name]),
            float(TEMPORAL_BUDGET_RATIOS[name]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for name in TEMPORAL_BUDGET_RATIOS
    ):
        raise FormalRunIntegrityError("Frozen temporal budget split is invalid.")
    main_dp_edge_mode = matrix.get("main_dp_edge_mode")
    appendix_dp_edge_mode = matrix.get("appendix_dp_edge_mode")
    if (
        main_dp_edge_mode != "component"
        or main_dp_edge_mode not in batch_config.dp_edge_modes
        or (
            appendix_dp_edge_mode is not None
            and (
                appendix_dp_edge_mode != "mechanism"
                or appendix_dp_edge_mode not in batch_config.dp_edge_modes
            )
        )
    ):
        raise FormalRunIntegrityError(
            "DP-edge-only main/appendix roles differ from the approved design."
        )
    if set(_require_sequence(
        payload.get("dp_edge_only_missing_dimensions"),
        "dp_edge_only_missing_dimensions",
    )) != _EXPECTED_EDGE_ONLY_MISSING:
        raise FormalRunIntegrityError("DP-edge-only NA dimensions changed.")

    normalization_path = phase7c_dir / "edge_normalization.json"
    if normalization_path.name not in artifact_hashes:
        raise FormalRunIntegrityError("Frozen normalization artifact is unreferenced.")
    normalization = _read_json(normalization_path, "edge normalization")
    if normalization.get("phase") != "7C" or normalization.get("status") != "frozen":
        raise FormalRunIntegrityError("Edge normalization artifact is not frozen.")
    normalization_policy = _require_mapping(
        normalization.get("policy"),
        "edge normalization policy",
    )
    if (
        normalization_policy.get("dp_release") is not False
        or normalization_policy.get("raw_reference_used") is not True
        or normalization_policy.get("quantile_method") != "linear"
        or not math.isclose(
            float(normalization_policy.get("low_quantile", math.nan)),
            0.05,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(normalization_policy.get("high_quantile", math.nan)),
            0.95,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or normalization_policy.get("zero_scale_policy")
        != "replace exact zero with 1 only"
        or
        normalization_policy.get("unit") != "seconds"
        or normalization_policy.get("normalization_only") is not True
        or normalization_policy.get("synthetic_generation_input") is not False
        or normalization_policy.get("epsilon_independent") is not True
    ):
        raise FormalRunIntegrityError("Edge normalization policy changed.")
    normalization_datasets = _require_mapping(
        normalization.get("datasets"),
        "edge normalization datasets",
    )

    dataset_rows = _require_sequence(payload.get("datasets"), "datasets")
    specs: list[FrozenDatasetExecutionSpec] = []
    names: list[str] = []
    for raw_row in dataset_rows:
        row = _require_mapping(raw_row, "dataset row")
        dataset = row.get("dataset")
        if not isinstance(dataset, str) or not dataset:
            raise FormalRunIntegrityError("Frozen dataset name is invalid.")
        names.append(dataset)
        source_sha256 = _validate_sha256(
            row.get("source_sha256"),
            f"{dataset}.source_sha256",
        )
        source_path = row.get("source_path")
        if source_path is not None and (
            not isinstance(source_path, str) or not source_path
        ):
            raise FormalRunIntegrityError("Frozen source_path is invalid.")
        l_max = row.get("l_max")
        t_max = row.get("t_max_seconds")
        top_k = row.get("top_k_variants")
        tau = row.get("min_edge_support")
        if (
            isinstance(l_max, bool)
            or not isinstance(l_max, int)
            or l_max <= 0
            or isinstance(t_max, bool)
            or not isinstance(t_max, (int, float))
            or not math.isfinite(float(t_max))
            or float(t_max) <= 0
            or top_k != reference_top_k
            or tau != reference_edge_tau
        ):
            raise FormalRunIntegrityError(
                f"Frozen parameters are inconsistent for {dataset!r}."
            )
        if row.get("normalization_reference") != normalization_path.name:
            raise FormalRunIntegrityError("Normalization reference changed.")
        factor_rows = _require_sequence(
            normalization_datasets.get(dataset),
            f"normalization[{dataset}]",
        )
        scales: dict[Edge, float] = {}
        for raw_factor in factor_rows:
            factor = _require_mapping(raw_factor, "normalization factor")
            if factor.get("selected_for_experiment") is not True:
                continue
            source = factor.get("source")
            target = factor.get("target")
            value = factor.get("normalization_factor_seconds")
            if (
                not isinstance(source, str)
                or not source
                or not isinstance(target, str)
                or not target
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise FormalRunIntegrityError("Frozen normalization factor is invalid.")
            edge = (source, target)
            if edge in scales:
                raise FormalRunIntegrityError("Duplicate frozen normalization edge.")
            scales[edge] = float(value)
        expected_scale_count = row.get("normalization_edge_count")
        if expected_scale_count != len(scales) or not scales:
            raise FormalRunIntegrityError(
                f"Normalization edge count changed for {dataset!r}."
            )
        specs.append(
            FrozenDatasetExecutionSpec(
                dataset=dataset,
                source_sha256=source_sha256,
                source_path=source_path,
                l_max=l_max,
                t_max=float(t_max),
                top_k_variants=top_k,
                min_edge_support=tau,
                normalization_scales=scales,
                evaluation_log=_parse_evaluation_log(
                    row.get("evaluation_log"),
                    dataset,
                ),
            )
        )
    if not specs or len(set(names)) != len(names):
        raise FormalRunIntegrityError("Frozen datasets must be non-empty and unique.")
    if set(normalization_datasets) != set(names):
        raise FormalRunIntegrityError("Normalization dataset set changed.")
    manifest_sources = _require_mapping(
        manifest.get("source_sha256"),
        "phase7c_manifest.source_sha256",
    )
    expected_sources = {
        item.dataset: item.source_sha256 for item in specs
    }
    if manifest_sources != expected_sources:
        raise FormalRunIntegrityError(
            "Phase 7C manifest source hashes differ from the final config."
        )

    process_quality_raw = payload.get("process_quality")
    process_quality = None
    if process_quality_raw is not None:
        process_quality = _require_mapping(process_quality_raw, "process_quality")
        quality_datasets = set(
            _require_sequence(process_quality.get("datasets"), "process_quality.datasets")
        )
        if not quality_datasets or not quality_datasets.issubset(set(names)):
            raise FormalRunIntegrityError("Process-quality dataset scope is invalid.")
        for spec in specs:
            if (spec.dataset in quality_datasets) != (spec.evaluation_log is not None):
                raise FormalRunIntegrityError(
                    "Evaluation-log scope differs from process-quality datasets."
                )
        if process_quality.get("conformance_variant") not in (
            "token_based",
            "alignments",
        ):
            raise FormalRunIntegrityError("Frozen conformance variant is invalid.")
        noise = process_quality.get("inductive_miner_noise_threshold")
        if (
            isinstance(noise, bool)
            or not isinstance(noise, (int, float))
            or not math.isfinite(float(noise))
            or float(noise) < 0
            or float(noise) > 1
        ):
            raise FormalRunIntegrityError("Frozen process-model threshold is invalid.")
    elif any(spec.evaluation_log is not None for spec in specs):
        raise FormalRunIntegrityError("Unscoped evaluation log exists in final config.")

    return FrozenExecutionSpec(
        config_path=config_path,
        config_sha256=config_sha256,
        phase7c_manifest_sha256=_sha256_file(manifest_path),
        runtime=frozen_runtime,
        batch_config=batch_config,
        datasets=tuple(specs),
        reference_top_k=reference_top_k,
        reference_edge_tau=reference_edge_tau,
        main_dp_edge_mode=main_dp_edge_mode,
        appendix_dp_edge_mode=appendix_dp_edge_mode,
        process_quality=process_quality,
        artifact_sha256=artifact_hashes,
    )


@dataclass(frozen=True)
class FormalCondition:
    scenario_id: str
    condition: BatchCondition

    @property
    def formal_condition_id(self) -> str:
        """Return formal condition id.

        Inputs:
            None.
        Outputs:
            A value of type `str` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return f"{self.scenario_id}|{self.condition.condition_id}"

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
        return _sha256_bytes(self.formal_condition_id.encode("utf-8"))

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
            "scenario_id": self.scenario_id,
            "formal_condition_id": self.formal_condition_id,
            "checkpoint_key": self.checkpoint_key,
            "condition": self.condition.to_dict(),
        }


@dataclass(frozen=True)
class FormalExecutionPlan:
    spec: FrozenExecutionSpec
    conditions: tuple[FormalCondition, ...]
    code_sha256: Mapping[str, str]
    plan_sha256: str

    def _core_dict(self) -> dict[str, Any]:
        """Return core dict.

        Inputs:
            None.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return {
            "name": "BUL_formal_execution_plan",
            "phase": "8A",
            "status": "planned",
            "scope": "reference_numeric",
            "scenario_id": self.spec.scenario_id,
            "phase7c_final_config": self.spec.config_path.name,
            "phase7c_final_config_sha256": self.spec.config_sha256,
            "phase7c_manifest_sha256": self.spec.phase7c_manifest_sha256,
            "batch_config": self.spec.batch_config.to_dict(),
            "counts": {
                "datasets": len(self.spec.datasets),
                "dp_conditions": len(self.conditions),
                "baseline_rows": 2 * len(self.spec.datasets),
                "experiment_rows": len(self.conditions) + 2 * len(self.spec.datasets),
                "stage_rows": 3 * len(self.conditions),
                "metadata_rows": len(self.conditions),
            },
            "reference_parameters": {
                "top_k": self.spec.reference_top_k,
                "edge_tau": self.spec.reference_edge_tau,
                "main_dp_edge_mode": self.spec.main_dp_edge_mode,
                "appendix_dp_edge_mode": self.spec.appendix_dp_edge_mode,
            },
            "source_sha256": {
                item.dataset: item.source_sha256 for item in self.spec.datasets
            },
            "runtime": dict(self.spec.runtime),
            "code_sha256": dict(self.code_sha256),
            "condition_plan": [condition.to_dict() for condition in self.conditions],
            "process_quality_scope_status": "deferred_not_frozen",
            "process_quality_conditions": 0,
            "sensitivity_pending": True,
            "artifact_role": "public_benchmark_evaluation_not_dp_release",
            "conditional_dp_claim": True,
            "raw_data_rows_serialized": False,
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
        return {**self._core_dict(), "plan_sha256": self.plan_sha256}


def build_formal_execution_plan(
    spec: FrozenExecutionSpec,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> FormalExecutionPlan:
    """Build the deterministic reference scenario without reading XES files."""

    if not isinstance(spec, FrozenExecutionSpec):
        raise TypeError("spec must be a FrozenExecutionSpec.")
    conditions: list[FormalCondition] = []
    config = spec.batch_config
    for dataset in spec.dataset_names:
        for epsilon in config.epsilons:
            for seed in config.seeds:
                for strategy in config.dp_full_strategies:
                    conditions.append(
                        FormalCondition(
                            spec.scenario_id,
                            BatchCondition(
                                dataset=dataset,
                                epsilon=epsilon,
                                seed=seed,
                                method="DP-full",
                                budget_strategy=strategy,
                                dp_edge_mode=None,
                                evaluate_process_quality=False,
                                conformance_variant=None,
                                process_model_noise_threshold=None,
                            ),
                        )
                    )
                for mode in config.dp_edge_modes:
                    conditions.append(
                        FormalCondition(
                            spec.scenario_id,
                            BatchCondition(
                                dataset=dataset,
                                epsilon=epsilon,
                                seed=seed,
                                method="DP-edge-only",
                                budget_strategy=None,
                                dp_edge_mode=mode,
                                evaluate_process_quality=False,
                                conformance_variant=None,
                                process_model_noise_threshold=None,
                            ),
                        )
                    )
    expected = config.expected_dp_conditions(len(spec.datasets))
    if len(conditions) != expected:
        raise RuntimeError("Formal plan size differs from the frozen matrix.")
    identifiers = [condition.formal_condition_id for condition in conditions]
    if len(set(identifiers)) != len(identifiers):
        raise FormalRunIntegrityError("Formal plan produced duplicate IDs.")
    code_hashes = build_code_manifest(project_root)
    provisional = FormalExecutionPlan(
        spec=spec,
        conditions=tuple(conditions),
        code_sha256=code_hashes,
        plan_sha256="",
    )
    plan_hash = _sha256_bytes(_json_bytes(provisional._core_dict()))
    return FormalExecutionPlan(
        spec=spec,
        conditions=tuple(conditions),
        code_sha256=code_hashes,
        plan_sha256=plan_hash,
    )


def write_formal_execution_plan(
    plan: FormalExecutionPlan,
    output_dir: str | Path,
) -> Path:
    """Write an immutable plan; an existing different plan is rejected."""

    if not isinstance(plan, FormalExecutionPlan):
        raise TypeError("plan must be a FormalExecutionPlan.")
    path = Path(output_dir) / _PLAN_FILENAME
    payload = _json_bytes(plan.to_dict())
    if path.exists():
        if path.read_bytes() != payload:
            raise FormalRunIntegrityError(
                "Existing formal execution plan differs from the current plan."
            )
        return path
    _atomic_write_bytes(path, payload)
    return path


ConditionExecutor = Callable[
    [DatasetExperimentInput, BatchCondition],
    SingleExperimentResult,
]


@dataclass(frozen=True)
class FormalExecutionSummary:
    status: str
    expected_conditions: int
    completed_conditions: int
    executed_this_invocation: int
    skipped_this_invocation: int
    failed_conditions: int
    output_dir: str
    artifacts: Mapping[str, str]

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
            "phase": "8A",
            "status": self.status,
            "expected_conditions": self.expected_conditions,
            "completed_conditions": self.completed_conditions,
            "executed_this_invocation": self.executed_this_invocation,
            "skipped_this_invocation": self.skipped_this_invocation,
            "failed_conditions": self.failed_conditions,
            "output_dir": self.output_dir,
            "artifacts": dict(self.artifacts),
        }


class FormalExperimentRunner:
    """Execute and resume the frozen reference matrix one condition at a time."""

    def __init__(
        self,
        spec: FrozenExecutionSpec,
        phase7c_inputs: Sequence[Phase7CDatasetInput],
        output_dir: str | Path,
        *,
        executor: ConditionExecutor | None = None,
        project_root: str | Path = PROJECT_ROOT,
    ) -> None:
        """Initialize the instance and validate its construction inputs.

        Inputs:
            spec: Input value used by this operation.
            phase7c_inputs: Input value used by this operation.
            output_dir: Input value used by this operation.
            executor: Input value used by this operation.
            project_root: Input value used by this operation.
        Outputs:
            None.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        if not isinstance(spec, FrozenExecutionSpec):
            raise TypeError("spec must be a FrozenExecutionSpec.")
        self.spec = spec
        self.spec.verify_source_files()
        self.dataset_inputs = spec.build_dataset_inputs(
            phase7c_inputs,
            include_process_quality_events=False,
        )
        self.input_by_name = {
            item.dataset: item for item in self.dataset_inputs
        }
        self.plan = build_formal_execution_plan(spec, project_root=project_root)
        phase7b_plan = BatchExperimentRunner(spec.batch_config).build_plan(
            self.dataset_inputs
        )
        if [value.condition.to_dict() for value in self.plan.conditions] != [
            value.to_dict() for value in phase7b_plan
        ]:
            raise FormalRunIntegrityError(
                "Phase 8A plan differs from the unchanged Phase 7B plan."
            )
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plan_path = write_formal_execution_plan(self.plan, self.output_dir)
        self.checkpoint_root = self.output_dir / "checkpoints"
        self.executor = executor or self._default_executor

    @staticmethod
    def _default_executor(
        dataset_input: DatasetExperimentInput,
        condition: BatchCondition,
    ) -> SingleExperimentResult:
        """Return default executor.

        Inputs:
            dataset_input: Input value used by this operation.
            condition: Input value used by this operation.
        Outputs:
            A value of type `SingleExperimentResult` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return run_single_experiment(
            dataset_input.traces,
            condition.single_config(dataset_input),
            normalization_scales=(
                dataset_input.normalization_scales
                if condition.method == "DP-full"
                else None
            ),
            evaluation_events=(
                dataset_input.evaluation_events
                if condition.evaluate_process_quality
                else None
            ),
            evaluation_log_id=(
                dataset_input.evaluation_log_id
                if condition.evaluate_process_quality
                else None
            ),
        )

    def _checkpoint_dir(self, formal: FormalCondition) -> Path:
        """Return checkpoint dir.

        Inputs:
            formal: Input value used by this operation.
        Outputs:
            A value of type `Path` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return self.checkpoint_root / formal.checkpoint_key

    def _checkpoint_manifest_path(self, formal: FormalCondition) -> Path:
        """Return checkpoint manifest path.

        Inputs:
            formal: Input value used by this operation.
        Outputs:
            A value of type `Path` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        return self._checkpoint_dir(formal) / _CHECKPOINT_MANIFEST

    def _expected_single_config(self, formal: FormalCondition) -> dict[str, Any]:
        """Return expected single config.

        Inputs:
            formal: Input value used by this operation.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        item = self.input_by_name[formal.condition.dataset]
        return formal.condition.single_config(item).to_dict()

    @staticmethod
    def _unit_interval(value: Any, label: str) -> float:
        """Validate and return a finite value in the unit interval.

        Inputs:
            value: Candidate value to validate or transform.
            label: Input value used by this operation.
        Outputs:
            A value of type `float` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or float(value) > 1.0
        ):
            raise FormalRunIntegrityError(
                f"{label} must be a finite value in [0, 1]."
            )
        return float(value)

    def _validate_losses(
        self,
        losses_raw: Any,
        *,
        edge_only: bool,
        label: str,
    ) -> dict[str, Any]:
        """Validate losses.

        Inputs:
            losses_raw: Input value used by this operation.
            edge_only: Input value used by this operation.
            label: Input value used by this operation.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        losses = _require_mapping(losses_raw, label)
        if set(losses) != _LOSS_FIELDS:
            raise FormalRunIntegrityError(f"{label} loss fields changed.")
        structure = self._unit_interval(
            losses["L_structure"],
            f"{label}.L_structure",
        )
        if edge_only:
            if any(
                losses[field] is not None
                for field in ("L_temporal", "L_information", "BUL")
            ):
                raise FormalRunIntegrityError(
                    "DP-edge-only unavailable loss dimensions must remain NA."
                )
            return dict(losses)

        temporal = self._unit_interval(
            losses["L_temporal"],
            f"{label}.L_temporal",
        )
        information = self._unit_interval(
            losses["L_information"],
            f"{label}.L_information",
        )
        bul = self._unit_interval(losses["BUL"], f"{label}.BUL")
        if not math.isclose(
            bul,
            (structure + temporal + information) / 3.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise FormalRunIntegrityError(
                f"{label}.BUL differs from the approved three-part mean."
            )
        return dict(losses)

    def _validate_condition_metadata(
        self,
        formal: FormalCondition,
        metadata_raw: Any,
    ) -> dict[str, Any]:
        """Validate condition metadata.

        Inputs:
            formal: Input value used by this operation.
            metadata_raw: Input value used by this operation.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        metadata = _require_mapping(metadata_raw, "condition metadata")
        expected_config = self._expected_single_config(formal)
        if (
            metadata.get("name") != "BUL_single_experiment"
            or metadata.get("phase") != "7A"
            or metadata.get("config") != expected_config
        ):
            raise FormalRunIntegrityError(
                f"Checkpoint SingleExperimentConfig changed: "
                f"{formal.formal_condition_id}"
            )
        edge_only = formal.condition.method == "DP-edge-only"
        stages = _require_mapping(
            metadata.get("stage_losses"),
            "checkpoint stage_losses",
        )
        if set(stages) != set(_STAGE_ORDER):
            raise FormalRunIntegrityError("Checkpoint stage set changed.")
        checked_stages = {
            stage: self._validate_losses(
                stages[stage],
                edge_only=edge_only,
                label=f"stage_losses.{stage}",
            )
            for stage in _STAGE_ORDER
        }

        record = _require_mapping(
            metadata.get("result_record"),
            "checkpoint result_record",
        )
        expected_record_values = {
            "dataset": formal.condition.dataset,
            "epsilon": formal.condition.epsilon,
            "method": formal.condition.method,
            "seed": formal.condition.seed,
            "budget_strategy": formal.condition.budget_strategy,
            "dp_edge_mode": formal.condition.dp_edge_mode,
            "process_quality_evaluated": False,
            "conformance_variant": None,
        }
        for field, expected in expected_record_values.items():
            if record.get(field) != expected:
                raise FormalRunIntegrityError(
                    f"Checkpoint result field changed: {field}."
                )
        for field in _LOSS_FIELDS:
            if record.get(field) != checked_stages["total"][field]:
                raise FormalRunIntegrityError(
                    f"Checkpoint total loss and result row differ: {field}."
                )
        if any(record.get(field) is not None for field in _QUALITY_FIELDS):
            raise FormalRunIntegrityError(
                "Numerical-only Phase 8A checkpoints cannot contain PM4Py metrics."
            )
        runtime_seconds = record.get("runtime")
        if (
            isinstance(runtime_seconds, bool)
            or not isinstance(runtime_seconds, (int, float))
            or not math.isfinite(float(runtime_seconds))
            or float(runtime_seconds) < 0.0
        ):
            raise FormalRunIntegrityError("Checkpoint runtime is invalid.")
        if metadata.get("process_quality") is not None:
            raise FormalRunIntegrityError(
                "Formal PM4Py conditions are deferred and must remain absent."
            )
        if metadata.get("synthetic_log_generation") is not None:
            raise FormalRunIntegrityError(
                "Numerical-only conditions must not generate a synthetic log."
            )
        if edge_only:
            if metadata.get("temporal_generation") is not None:
                raise FormalRunIntegrityError(
                    "DP-edge-only cannot contain temporal reconstruction."
                )
        elif metadata.get("temporal_generation") is None:
            raise FormalRunIntegrityError(
                "DP-full checkpoint is missing temporal reconstruction."
            )

        release = _require_mapping(metadata.get("release"), "checkpoint release")
        if (
            release.get("name") != "G_DP"
            or release.get("method") != formal.condition.method
            or release.get("epsilon_total") != formal.condition.epsilon
            or release.get("seed") != formal.condition.seed
            or release.get("budget_strategy")
            != formal.condition.budget_strategy
            or release.get("dp_edge_mode") != formal.condition.dp_edge_mode
            or release.get("composition") != "sequential"
        ):
            raise FormalRunIntegrityError(
                "Checkpoint release metadata differs from its formal condition."
            )
        component_epsilons = _require_mapping(
            release.get("component_epsilons"),
            "release.component_epsilons",
        )
        if set(component_epsilons) != set(PRIVACY_COMPONENTS):
            raise FormalRunIntegrityError("Release component budget set changed.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in component_epsilons.values()
        ):
            raise FormalRunIntegrityError("Release component budget is invalid.")
        expected_spent = formal.condition.epsilon
        if edge_only and formal.condition.dp_edge_mode == "component":
            expected_spent *= DP_EDGE_COMPONENT_SHARE
        if not math.isclose(
            float(release.get("epsilon_spent", math.nan)),
            expected_spent,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            sum(float(value) for value in component_epsilons.values()),
            expected_spent,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise FormalRunIntegrityError(
                "Checkpoint sequential epsilon spending changed."
            )
        sensitivities = _require_mapping(
            release.get("sensitivities"),
            "release.sensitivities",
        )
        l_max = expected_config["l_max"]
        t_max = expected_config["t_max"]
        edge_bound = float(l_max - 1)
        expected_sensitivities = {
            "activity": float(l_max),
            "edge": edge_bound,
            "variant": 1.0,
            "time": edge_bound * t_max,
            "time_sum": edge_bound * t_max,
            "time_squared_sum": edge_bound * t_max**2,
            "l_max": l_max,
            "t_max": t_max,
        }
        if set(sensitivities) != set(expected_sensitivities) or any(
            not math.isclose(
                float(sensitivities[name]),
                float(expected),
                rel_tol=1e-15,
                abs_tol=1e-12,
            )
            for name, expected in expected_sensitivities.items()
        ):
            raise FormalRunIntegrityError(
                "Checkpoint sensitivities differ from the approved formulas."
            )
        release_domain = _require_mapping(
            release.get("release_domain"),
            "release.release_domain",
        )
        if (
            release_domain.get("top_k_variants")
            != expected_config["top_k_variants"]
            or release_domain.get("min_edge_support")
            != expected_config["min_edge_support"]
        ):
            raise FormalRunIntegrityError("Checkpoint release domain changed.")

        if edge_only:
            if (
                release.get("temporal_epsilons") is not None
                or release.get("activity") is not None
                or release.get("variants") is not None
                or release.get("variant_other") is not None
                or release.get("temporal") is not None
            ):
                raise FormalRunIntegrityError(
                    "DP-edge-only release contains an unavailable component."
                )
        else:
            temporal_epsilons = _require_mapping(
                release.get("temporal_epsilons"),
                "release.temporal_epsilons",
            )
            time_budget = float(component_epsilons["time"])
            for name, ratio in TEMPORAL_BUDGET_RATIOS.items():
                if not math.isclose(
                    float(temporal_epsilons.get(name, math.nan)),
                    time_budget * float(ratio),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise FormalRunIntegrityError(
                        "Checkpoint temporal budget split changed."
                    )
        analysis = _require_mapping(
            metadata.get("analysis_metadata"),
            "analysis_metadata",
        )
        if analysis.get("statistical_spaces_frozen_before_noise") is not True:
            raise FormalRunIntegrityError(
                "Checkpoint does not attest pre-noise statistical domains."
            )
        if not isinstance(analysis.get("edge_domain"), list):
            raise FormalRunIntegrityError("Checkpoint edge domain is missing.")
        return metadata

    @staticmethod
    def _validate_checkpoint_csv(
        path: Path,
        record: Mapping[str, Any],
    ) -> None:
        """Validate checkpoint csv.

        Inputs:
            path: Path to the input or output artifact.
            record: Single record processed by the operation.
        Outputs:
            None.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except OSError as exc:
            raise FormalRunIntegrityError(
                f"Cannot read checkpoint CSV: {path}"
            ) from exc
        if len(rows) != 1:
            raise FormalRunIntegrityError(
                "Checkpoint CSV must contain exactly one result row."
            )
        row = rows[0]
        if set(row) != set(record):
            raise FormalRunIntegrityError("Checkpoint CSV fields changed.")
        for field, expected in record.items():
            actual = row[field]
            if expected is None:
                matches = actual == "NA"
            elif isinstance(expected, bool):
                matches = actual == str(expected)
            elif isinstance(expected, (int, float)) and not isinstance(
                expected,
                bool,
            ):
                try:
                    matches = math.isclose(
                        float(actual),
                        float(expected),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                except ValueError:
                    matches = False
            else:
                matches = actual == str(expected)
            if not matches:
                raise FormalRunIntegrityError(
                    f"Checkpoint CSV and JSON differ for field {field!r}."
                )

    def _load_checkpoint(
        self,
        formal: FormalCondition,
    ) -> dict[str, Any] | None:
        """Load checkpoint.

        Inputs:
            formal: Input value used by this operation.
        Outputs:
            A value of type `dict[str, Any] | None` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        manifest_path = self._checkpoint_manifest_path(formal)
        if not manifest_path.exists():
            return None
        manifest = _read_json(manifest_path, "condition checkpoint manifest")
        if (
            manifest.get("name") != "BUL_formal_condition_checkpoint"
            or manifest.get("phase") != "8A"
            or manifest.get("status") != "complete"
            or manifest.get("formal_condition_id") != formal.formal_condition_id
            or manifest.get("scenario_id") != formal.scenario_id
            or manifest.get("phase7c_final_config_sha256")
            != self.spec.config_sha256
            or manifest.get("execution_plan_sha256") != self.plan.plan_sha256
            or manifest.get("condition") != formal.condition.to_dict()
        ):
            raise FormalRunIntegrityError(
                f"Committed checkpoint metadata changed: {formal.formal_condition_id}"
            )
        artifacts = _require_mapping(
            manifest.get("artifacts"),
            "checkpoint artifacts",
        )
        hashes = _require_mapping(
            manifest.get("artifact_sha256"),
            "checkpoint artifact hashes",
        )
        expected_artifacts = {
            "experiment_result": "experiment_result.csv",
            "experiment_metadata": "experiment_metadata.json",
        }
        if artifacts != expected_artifacts or set(hashes) != set(
            expected_artifacts.values()
        ):
            raise FormalRunIntegrityError("Checkpoint artifact set changed.")
        dataset_spec = next(
            value
            for value in self.spec.datasets
            if value.dataset == formal.condition.dataset
        )
        if (
            manifest.get("dataset_source_sha256")
            != dataset_spec.source_sha256
            or manifest.get("evaluation_log_sha256")
            != (
                None
                if dataset_spec.evaluation_log is None
                else dataset_spec.evaluation_log.sha256
            )
        ):
            raise FormalRunIntegrityError(
                "Committed checkpoint data fingerprint changed."
            )
        for logical_name in ("experiment_result", "experiment_metadata"):
            filename = artifacts.get(logical_name)
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise FormalRunIntegrityError("Checkpoint artifact name is invalid.")
            expected = _validate_sha256(
                hashes.get(filename),
                f"checkpoint hash {filename}",
            )
            path = manifest_path.parent / filename
            if not path.is_file() or _sha256_file(path) != expected:
                raise FormalRunIntegrityError(
                    f"Committed checkpoint artifact changed: {path}"
                )
        metadata_path = manifest_path.parent / artifacts["experiment_metadata"]
        metadata = _read_json(metadata_path, "condition metadata")
        checked = self._validate_condition_metadata(formal, metadata)
        self._validate_checkpoint_csv(
            manifest_path.parent / artifacts["experiment_result"],
            _require_mapping(checked["result_record"], "result_record"),
        )
        return checked

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
            Frozen experiment orchestration and result derivation.
        """
        payload = {
            "name": "BUL_formal_run_progress",
            "phase": "8A",
            "status": status,
            "scenario_id": self.spec.scenario_id,
            "phase7c_final_config_sha256": self.spec.config_sha256,
            "execution_plan_sha256": self.plan.plan_sha256,
            "expected_conditions": len(self.plan.conditions),
            "completed_conditions": completed,
            "remaining_conditions": len(self.plan.conditions) - completed,
            "executed_this_invocation": executed,
            "skipped_this_invocation": skipped,
            "failed_this_invocation": failed,
            "authoritative_completion_marker": _FINAL_MANIFEST,
        }
        _atomic_write_bytes(
            self.output_dir / _PROGRESS_FILENAME,
            _json_bytes(payload),
        )

    def _execute_condition(self, formal: FormalCondition) -> None:
        """Execute condition.

        Inputs:
            formal: Input value used by this operation.
        Outputs:
            None.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        checkpoint_dir = self._checkpoint_dir(formal)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        item = self.input_by_name[formal.condition.dataset]
        result = self.executor(item, formal.condition)
        if not isinstance(result, SingleExperimentResult):
            raise TypeError("condition executor must return SingleExperimentResult.")
        if result.config.to_dict() != self._expected_single_config(formal):
            raise FormalRunIntegrityError(
                "Condition executor returned a result for a different config."
            )
        self._validate_condition_metadata(formal, result.to_dict())
        artifacts = write_single_experiment_artifacts(result, checkpoint_dir)
        artifact_hashes = {
            filename: _sha256_file(checkpoint_dir / filename)
            for filename in artifacts.values()
        }
        dataset_spec = next(
            value
            for value in self.spec.datasets
            if value.dataset == formal.condition.dataset
        )
        manifest = {
            "name": "BUL_formal_condition_checkpoint",
            "phase": "8A",
            "status": "complete",
            "scenario_id": formal.scenario_id,
            "formal_condition_id": formal.formal_condition_id,
            "condition": formal.condition.to_dict(),
            "phase7c_final_config_sha256": self.spec.config_sha256,
            "execution_plan_sha256": self.plan.plan_sha256,
            "dataset_source_sha256": dataset_spec.source_sha256,
            "evaluation_log_sha256": (
                None
                if dataset_spec.evaluation_log is None
                else dataset_spec.evaluation_log.sha256
            ),
            "artifacts": artifacts,
            "artifact_sha256": artifact_hashes,
            "raw_data_rows_serialized": False,
        }
        failure_path = checkpoint_dir / "failure.json"
        if failure_path.exists():
            failure_path.unlink()
        _atomic_write_bytes(
            checkpoint_dir / _CHECKPOINT_MANIFEST,
            _json_bytes(manifest),
        )

    def _write_failure(self, formal: FormalCondition, exc: Exception) -> None:
        """Write failure.

        Inputs:
            formal: Input value used by this operation.
            exc: Input value used by this operation.
        Outputs:
            None.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        checkpoint_dir = self._checkpoint_dir(formal)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": "BUL_formal_condition_failure",
            "phase": "8A",
            "status": "failed",
            "scenario_id": formal.scenario_id,
            "formal_condition_id": formal.formal_condition_id,
            "condition": formal.condition.to_dict(),
            "phase7c_final_config_sha256": self.spec.config_sha256,
            "execution_plan_sha256": self.plan.plan_sha256,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "completion_marker_written": False,
        }
        _atomic_write_bytes(
            checkpoint_dir / "failure.json",
            _json_bytes(payload),
        )

    @staticmethod
    def _baseline_record(
        metadata: Mapping[str, Any],
        scenario_id: str,
        method: str,
        losses: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return baseline record.

        Inputs:
            metadata: Metadata associated with the artifact or experiment condition.
            scenario_id: Input value used by this operation.
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
            **losses,
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
            "scenario_id": scenario_id,
            "batch_condition_id": f"{metadata['dataset']}|{method}",
            "condition_id": (
                f"{scenario_id}|{metadata['dataset']}|{method}"
            ),
            "status": "success",
            "comparison": (
                "G_raw versus G_raw"
                if method == "Raw"
                else "G_raw versus G_clip"
            ),
        }

    def _aggregate(
        self,
        metadata_by_id: Mapping[str, Mapping[str, Any]],
        *,
        executed_this_invocation: int,
        skipped_this_invocation: int,
    ) -> dict[str, str]:
        """Aggregate.

        Inputs:
            metadata_by_id: Input value used by this operation.
            executed_this_invocation: Input value used by this operation.
            skipped_this_invocation: Input value used by this operation.
        Outputs:
            A value of type `dict[str, str]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        expected_ids = {
            formal.formal_condition_id for formal in self.plan.conditions
        }
        if set(metadata_by_id) != expected_ids:
            raise FormalRunIntegrityError(
                "Formal results cannot be aggregated before every checkpoint commits."
            )
        existing_manifest = self.output_dir / _FINAL_MANIFEST
        if existing_manifest.exists():
            self._validate_final_manifest(existing_manifest)
            return {
                "experiment_result": "experiment_result.csv",
                "experiment_stage_result": "experiment_stage_result.csv",
                "experiment_run_metadata": "experiment_run_metadata.jsonl",
                "formal_run_manifest": _FINAL_MANIFEST,
            }

        clipping_by_dataset: dict[str, dict[str, Any]] = {}
        experiment_records: list[dict[str, Any]] = []
        stage_records: list[dict[str, Any]] = []
        metadata_lines: list[str] = []
        checkpoint_hashes: dict[str, str] = {}
        process_quality_conditions = 0
        edge_domain_hashes: dict[str, str] = {}
        full_domain_hashes: dict[str, str] = {}

        for formal in self.plan.conditions:
            metadata = dict(metadata_by_id[formal.formal_condition_id])
            result_record = _require_mapping(
                metadata.get("result_record"),
                "checkpoint result_record",
            )
            stages = _require_mapping(
                metadata.get("stage_losses"),
                "checkpoint stage_losses",
            )
            if formal.condition.method == "DP-full":
                clipping = _require_mapping(
                    stages.get("clipping"),
                    "clipping loss",
                )
                previous = clipping_by_dataset.setdefault(
                    formal.condition.dataset,
                    dict(clipping),
                )
                if dict(clipping) != previous:
                    raise FormalRunIntegrityError(
                        "ClippedRaw losses changed across formal conditions for "
                        f"{formal.condition.dataset!r}."
                    )
            record = dict(result_record)
            record.update(
                {
                    "scenario_id": formal.scenario_id,
                    "batch_condition_id": formal.condition.condition_id,
                    "condition_id": formal.formal_condition_id,
                    "status": "success",
                    "comparison": "G_raw versus G_DP",
                }
            )
            experiment_records.append(record)
            if record.get("process_quality_evaluated") is True:
                process_quality_conditions += 1
            analysis = _require_mapping(
                metadata.get("analysis_metadata"),
                "checkpoint analysis_metadata",
            )
            edge_domain_hash = _sha256_bytes(
                _json_bytes(analysis.get("edge_domain"))
            )
            previous_edge = edge_domain_hashes.setdefault(
                formal.condition.dataset,
                edge_domain_hash,
            )
            if previous_edge != edge_domain_hash:
                raise FormalRunIntegrityError(
                    "Frozen edge domain changed across conditions for "
                    f"{formal.condition.dataset!r}."
                )
            if formal.condition.method == "DP-full":
                full_domain_hash = _sha256_bytes(
                    _json_bytes(
                        {
                            "activity_domain": analysis.get("activity_domain"),
                            "edge_domain": analysis.get("edge_domain"),
                            "variant_domain": analysis.get("variant_domain"),
                            "temporal_domain": analysis.get("temporal_domain"),
                        }
                    )
                )
                previous_full = full_domain_hashes.setdefault(
                    formal.condition.dataset,
                    full_domain_hash,
                )
                if previous_full != full_domain_hash:
                    raise FormalRunIntegrityError(
                        "Frozen DP-full statistical domain changed across "
                        f"conditions for {formal.condition.dataset!r}."
                    )
            for stage in _STAGE_ORDER:
                losses = _require_mapping(stages.get(stage), f"{stage} loss")
                stage_records.append(
                    {
                        "scenario_id": formal.scenario_id,
                        "condition_id": formal.formal_condition_id,
                        "batch_condition_id": formal.condition.condition_id,
                        "dataset": formal.condition.dataset,
                        "epsilon": formal.condition.epsilon,
                        "method": formal.condition.method,
                        "seed": formal.condition.seed,
                        "budget_strategy": formal.condition.budget_strategy,
                        "dp_edge_mode": formal.condition.dp_edge_mode,
                        "stage": stage,
                        **losses,
                    }
                )
            metadata_lines.append(
                json.dumps(
                    {
                        "scenario_id": formal.scenario_id,
                        "formal_condition_id": formal.formal_condition_id,
                        "condition": formal.condition.to_dict(),
                        "result": metadata,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            checkpoint_manifest = self._checkpoint_manifest_path(formal)
            checkpoint_hashes[formal.formal_condition_id] = _sha256_file(
                checkpoint_manifest
            )

        if (
            process_quality_conditions != 0
            or set(clipping_by_dataset) != set(self.spec.dataset_names)
            or set(edge_domain_hashes) != set(self.spec.dataset_names)
            or set(full_domain_hashes) != set(self.spec.dataset_names)
        ):
            raise FormalRunIntegrityError(
                "Formal numerical scope or per-dataset domains are incomplete."
            )
        baseline_records: list[dict[str, Any]] = []
        for item in self.dataset_inputs:
            metadata = item.to_dict()
            baseline_records.append(
                self._baseline_record(
                    metadata,
                    self.spec.scenario_id,
                    "Raw",
                    {
                        "L_structure": 0.0,
                        "L_temporal": 0.0,
                        "L_information": 0.0,
                        "BUL": 0.0,
                    },
                )
            )
            baseline_records.append(
                self._baseline_record(
                    metadata,
                    self.spec.scenario_id,
                    "ClippedRaw",
                    clipping_by_dataset[item.dataset],
                )
            )
        all_records = baseline_records + experiment_records
        if len(all_records) != len(self.plan.conditions) + 2 * len(self.spec.datasets):
            raise RuntimeError("Formal experiment row count is inconsistent.")
        if len(stage_records) != 3 * len(self.plan.conditions):
            raise RuntimeError("Formal stage row count is inconsistent.")

        experiment_path = self.output_dir / "experiment_result.csv"
        stage_path = self.output_dir / "experiment_stage_result.csv"
        metadata_path = self.output_dir / "experiment_run_metadata.jsonl"
        _atomic_write_dataframe(experiment_path, all_records)
        _atomic_write_dataframe(stage_path, stage_records)
        metadata_payload = ("\n".join(metadata_lines) + "\n").encode("utf-8")
        _atomic_write_bytes(metadata_path, metadata_payload)

        artifacts = {
            "experiment_result": experiment_path.name,
            "experiment_stage_result": stage_path.name,
            "experiment_run_metadata": metadata_path.name,
            "formal_run_manifest": _FINAL_MANIFEST,
        }
        scientific_hashes = {
            experiment_path.name: _sha256_file(experiment_path),
            stage_path.name: _sha256_file(stage_path),
            metadata_path.name: _sha256_file(metadata_path),
            self.plan_path.name: _sha256_file(self.plan_path),
        }
        manifest = {
            "name": "BUL_formal_run",
            "phase": "8A",
            "status": "complete",
            "scope": "reference_numeric",
            "scenario_id": self.spec.scenario_id,
            "phase7c_final_config_sha256": self.spec.config_sha256,
            "phase7c_manifest_sha256": self.spec.phase7c_manifest_sha256,
            "execution_plan_sha256": self.plan.plan_sha256,
            "counts": {
                "datasets": len(self.spec.datasets),
                "dp_conditions": len(self.plan.conditions),
                "completed_conditions": len(metadata_by_id),
                "failed_conditions": 0,
                "baseline_rows": len(baseline_records),
                "experiment_rows": len(all_records),
                "stage_rows": len(stage_records),
                "metadata_rows": len(metadata_lines),
                "process_quality_conditions": process_quality_conditions,
                "executed_finalizing_invocation": executed_this_invocation,
                "skipped_finalizing_invocation": skipped_this_invocation,
            },
            "artifacts": artifacts,
            "artifact_sha256": scientific_hashes,
            "checkpoint_manifest_sha256": checkpoint_hashes,
            "source_sha256": {
                item.dataset: item.source_sha256 for item in self.spec.datasets
            },
            "statistical_domain_sha256": {
                dataset: {
                    "edge_domain": edge_domain_hashes[dataset],
                    "dp_full_domain": full_domain_hashes[dataset],
                }
                for dataset in self.spec.dataset_names
            },
            "code_sha256": dict(self.plan.code_sha256),
            "runtime": dict(self.spec.runtime),
            "process_quality_scope_status": "deferred_not_frozen",
            "sensitivity_pending": True,
            "reference_parameters": {
                "top_k": self.spec.reference_top_k,
                "edge_tau": self.spec.reference_edge_tau,
                "main_dp_edge_mode": self.spec.main_dp_edge_mode,
                "appendix_dp_edge_mode": self.spec.appendix_dp_edge_mode,
            },
            "artifact_role": "public_benchmark_evaluation_not_dp_release",
            "conditional_dp_claim": True,
            "raw_data_rows_serialized": False,
        }
        _atomic_write_bytes(existing_manifest, _json_bytes(manifest))
        return artifacts

    def _validate_final_manifest(self, path: Path) -> dict[str, Any]:
        """Validate final manifest.

        Inputs:
            path: Path to the input or output artifact.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Frozen experiment orchestration and result derivation.
        """
        manifest = _read_json(path, "formal run manifest")
        if (
            manifest.get("name") != "BUL_formal_run"
            or manifest.get("phase") != "8A"
            or manifest.get("status") != "complete"
            or manifest.get("scope") != "reference_numeric"
            or manifest.get("scenario_id") != self.spec.scenario_id
            or manifest.get("phase7c_final_config_sha256")
            != self.spec.config_sha256
            or manifest.get("phase7c_manifest_sha256")
            != self.spec.phase7c_manifest_sha256
            or manifest.get("execution_plan_sha256") != self.plan.plan_sha256
            or manifest.get("source_sha256")
            != {
                item.dataset: item.source_sha256
                for item in self.spec.datasets
            }
            or manifest.get("code_sha256") != dict(self.plan.code_sha256)
            or manifest.get("runtime") != dict(self.spec.runtime)
            or manifest.get("process_quality_scope_status")
            != "deferred_not_frozen"
            or manifest.get("sensitivity_pending") is not True
            or manifest.get("artifact_role")
            != "public_benchmark_evaluation_not_dp_release"
            or manifest.get("conditional_dp_claim") is not True
            or manifest.get("raw_data_rows_serialized") is not False
        ):
            raise FormalRunIntegrityError("Formal completion manifest changed.")
        expected_reference = {
            "top_k": self.spec.reference_top_k,
            "edge_tau": self.spec.reference_edge_tau,
            "main_dp_edge_mode": self.spec.main_dp_edge_mode,
            "appendix_dp_edge_mode": self.spec.appendix_dp_edge_mode,
        }
        if manifest.get("reference_parameters") != expected_reference:
            raise FormalRunIntegrityError("Formal reference parameters changed.")
        counts = _require_mapping(manifest.get("counts"), "formal counts")
        expected_fixed_counts = {
            "datasets": len(self.spec.datasets),
            "dp_conditions": len(self.plan.conditions),
            "completed_conditions": len(self.plan.conditions),
            "failed_conditions": 0,
            "baseline_rows": 2 * len(self.spec.datasets),
            "experiment_rows": len(self.plan.conditions)
            + 2 * len(self.spec.datasets),
            "stage_rows": 3 * len(self.plan.conditions),
            "metadata_rows": len(self.plan.conditions),
            "process_quality_conditions": 0,
        }
        if set(counts) != set(expected_fixed_counts) | {
            "executed_finalizing_invocation",
            "skipped_finalizing_invocation",
        } or any(
            counts.get(name) != expected
            for name, expected in expected_fixed_counts.items()
        ):
            raise FormalRunIntegrityError("Formal completion counts changed.")
        executed = counts.get("executed_finalizing_invocation")
        skipped = counts.get("skipped_finalizing_invocation")
        if (
            isinstance(executed, bool)
            or not isinstance(executed, int)
            or executed < 0
            or isinstance(skipped, bool)
            or not isinstance(skipped, int)
            or skipped < 0
            or executed + skipped != len(self.plan.conditions)
        ):
            raise FormalRunIntegrityError(
                "Formal finalizing-invocation counts changed."
            )
        expected_artifacts = {
            "experiment_result": "experiment_result.csv",
            "experiment_stage_result": "experiment_stage_result.csv",
            "experiment_run_metadata": "experiment_run_metadata.jsonl",
            "formal_run_manifest": _FINAL_MANIFEST,
        }
        if manifest.get("artifacts") != expected_artifacts:
            raise FormalRunIntegrityError("Formal artifact mapping changed.")
        hashes = _require_mapping(
            manifest.get("artifact_sha256"),
            "formal artifact hashes",
        )
        expected_hashed_files = {
            "experiment_result.csv",
            "experiment_stage_result.csv",
            "experiment_run_metadata.jsonl",
            _PLAN_FILENAME,
        }
        if set(hashes) != expected_hashed_files:
            raise FormalRunIntegrityError("Formal artifact hash set changed.")
        for filename, expected_raw in hashes.items():
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise FormalRunIntegrityError("Formal artifact name is invalid.")
            expected = _validate_sha256(expected_raw, f"formal hash {filename}")
            artifact = self.output_dir / filename
            if not artifact.is_file() or _sha256_file(artifact) != expected:
                raise FormalRunIntegrityError(
                    f"Completed formal artifact changed: {filename}"
                )
        checkpoint_hashes = _require_mapping(
            manifest.get("checkpoint_manifest_sha256"),
            "checkpoint manifest hashes",
        )
        if set(checkpoint_hashes) != {
            condition.formal_condition_id for condition in self.plan.conditions
        }:
            raise FormalRunIntegrityError("Completed checkpoint set changed.")
        for formal in self.plan.conditions:
            expected = _validate_sha256(
                checkpoint_hashes[formal.formal_condition_id],
                "checkpoint manifest hash",
            )
            checkpoint = self._checkpoint_manifest_path(formal)
            if not checkpoint.is_file() or _sha256_file(checkpoint) != expected:
                raise FormalRunIntegrityError(
                    f"Checkpoint manifest changed: {formal.formal_condition_id}"
                )
        domains = _require_mapping(
            manifest.get("statistical_domain_sha256"),
            "formal statistical domain hashes",
        )
        if set(domains) != set(self.spec.dataset_names):
            raise FormalRunIntegrityError("Formal statistical domain set changed.")
        for dataset, raw_values in domains.items():
            values = _require_mapping(
                raw_values,
                f"statistical_domain_sha256[{dataset}]",
            )
            if set(values) != {"edge_domain", "dp_full_domain"}:
                raise FormalRunIntegrityError(
                    "Formal statistical domain hash fields changed."
                )
            for label, digest in values.items():
                _validate_sha256(digest, f"{dataset}.{label}")
        return manifest

    def run(
        self,
        *,
        max_new_conditions: int | None = None,
    ) -> FormalExecutionSummary:
        """Execute unfinished conditions and finalize only when all commit.

        Input
        -----
        max_new_conditions:
            Optional positive chunk size. It limits work in this invocation but
            never changes or truncates the frozen condition plan.

        Output
        ------
        FormalExecutionSummary
            Partial or complete progress. A complete formal manifest is written
            only after every condition checkpoint validates.

        Paper experiment step
        ---------------------
        Phase 8A reference execution. Missing/uncommitted checkpoints are
        rerun; committed but changed checkpoints fail closed; failures stop the
        current phase rather than dropping a seed or changing a parameter.
        """

        if max_new_conditions is not None and (
            isinstance(max_new_conditions, bool)
            or not isinstance(max_new_conditions, int)
            or max_new_conditions <= 0
        ):
            raise ValueError("max_new_conditions must be a positive integer.")

        metadata_by_id: dict[str, dict[str, Any]] = {}
        for formal in self.plan.conditions:
            metadata = self._load_checkpoint(formal)
            if metadata is not None:
                metadata_by_id[formal.formal_condition_id] = metadata
        skipped = len(metadata_by_id)
        executed = 0
        failed = 0

        for formal in self.plan.conditions:
            if formal.formal_condition_id in metadata_by_id:
                continue
            if max_new_conditions is not None and executed >= max_new_conditions:
                break
            try:
                self._execute_condition(formal)
                metadata = self._load_checkpoint(formal)
                if metadata is None:
                    raise RuntimeError("Condition completed without a commit manifest.")
                metadata_by_id[formal.formal_condition_id] = metadata
                executed += 1
                self._write_progress(
                    status="running",
                    completed=len(metadata_by_id),
                    executed=executed,
                    skipped=skipped,
                    failed=failed,
                )
            except Exception as exc:
                failed += 1
                self._write_failure(formal, exc)
                self._write_progress(
                    status="failed",
                    completed=len(metadata_by_id),
                    executed=executed,
                    skipped=skipped,
                    failed=failed,
                )
                raise FormalConditionExecutionError(
                    f"Formal condition failed: {formal.formal_condition_id}"
                ) from exc
            finally:
                gc.collect()

        complete = len(metadata_by_id) == len(self.plan.conditions)
        artifacts: dict[str, str] = {
            "formal_execution_plan": self.plan_path.name,
            "formal_run_progress": _PROGRESS_FILENAME,
        }
        if complete:
            self._write_progress(
                status="complete",
                completed=len(metadata_by_id),
                executed=executed,
                skipped=skipped,
                failed=failed,
            )
            artifacts.update(
                self._aggregate(
                    metadata_by_id,
                    executed_this_invocation=executed,
                    skipped_this_invocation=skipped,
                )
            )
            status = "complete"
        else:
            self._write_progress(
                status="partial",
                completed=len(metadata_by_id),
                executed=executed,
                skipped=skipped,
                failed=failed,
            )
            status = "partial"
        return FormalExecutionSummary(
            status=status,
            expected_conditions=len(self.plan.conditions),
            completed_conditions=len(metadata_by_id),
            executed_this_invocation=executed,
            skipped_this_invocation=skipped,
            failed_conditions=failed,
            output_dir=str(self.output_dir),
            artifacts=artifacts,
        )


__all__ = [
    "FormalCondition",
    "FormalConditionExecutionError",
    "FormalExecutionPlan",
    "FormalExecutionSummary",
    "FormalExperimentRunner",
    "FormalRunIntegrityError",
    "FrozenDatasetExecutionSpec",
    "FrozenEvaluationLogSpec",
    "FrozenExecutionSpec",
    "build_code_manifest",
    "build_formal_execution_plan",
    "current_runtime_metadata",
    "load_frozen_execution_spec",
    "write_formal_execution_plan",
]

"""Deterministic provenance manifests for the BUL research artifact.

Function:
    Hash the complete scientific implementation and describe local data,
    configuration, result, environment, and compact paper artifacts.
Input:
    A repository root containing ``main.py`` and ``src/bul_project``.
Output:
    JSON-compatible deterministic manifests and optional atomically written
    manifest files.
Paper role:
    Reproducibility and artifact traceability only. This module performs no
    behavioral statistic, privacy release, loss metric, or BUL calculation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from bul_project.config import PROJECT_ROOT


PROVENANCE_SCHEMA_VERSION = "1.0"
DEFAULT_CODE_MANIFEST = Path("paper_artifacts/configs/code_manifest.json")
DEFAULT_ARTIFACT_MANIFEST = Path("paper_artifacts/manifest.json")
_EXCLUDED_PARTS = {
    ".git",
    ".idea",
    ".pytest_cache",
    "__pycache__",
    "data",
    "results",
    "tests",
}
_REQUIRED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "pm4py",
    "lxml",
    "pytest",
)
_RELEASE_METADATA = (
    ".gitignore",
    "CITATION.cff",
    "LICENSE",
    "LICENSE_NOTICE.md",
    "README.md",
    "environment.yml",
    "environment_snapshot.md",
    "pyproject.toml",
    "pytest.ini",
    "requirements-lock.txt",
)


class ProvenanceError(RuntimeError):
    """Raised when a provenance source or stored manifest is inconsistent."""


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one existing regular file."""

    source = Path(path)
    if not source.is_file():
        raise ProvenanceError(f"Provenance source is not a file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize a mapping to stable UTF-8 JSON bytes."""

    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _aggregate_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    """Hash a stable sequence of path, size, and file-digest records."""

    normalized = [
        {
            "path": str(record["path"]),
            "size_bytes": int(record["size_bytes"]),
            "sha256": str(record["sha256"]),
        }
        for record in records
    ]
    return hashlib.sha256(_json_bytes({"files": normalized})).hexdigest()


def _inside(root: Path, path: Path) -> Path:
    """Resolve a path and fail unless it remains inside the repository root."""

    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ProvenanceError(
            f"Provenance path escapes the repository root: {resolved}"
        ) from exc
    return resolved


def _record_files(root: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Return sorted relative-path, size, and SHA-256 records."""

    unique: dict[str, Path] = {}
    for candidate in paths:
        resolved = _inside(root, candidate)
        if not resolved.is_file():
            continue
        relative = resolved.relative_to(root).as_posix()
        unique[relative] = resolved
    return [
        {
            "path": relative,
            "size_bytes": unique[relative].stat().st_size,
            "sha256": sha256_file(unique[relative]),
        }
        for relative in sorted(unique)
    ]


def _collection(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return one deterministic file-collection description."""

    return {
        "status": "present" if records else "absent",
        "file_count": len(records),
        "aggregate_sha256": _aggregate_sha256(records),
        "files": records,
    }


def iter_code_files(
    project_root: str | Path = PROJECT_ROOT,
) -> tuple[Path, ...]:
    """Return every Python file included in the release code boundary.

    The boundary contains root ``main.py``, all package modules under
    ``src/bul_project``, and all Python command wrappers under ``scripts``.
    Tests, data, results, caches, IDE files, and generated outputs are excluded.
    """

    root = Path(project_root).resolve()
    main_path = root / "main.py"
    package_root = root / "src" / "bul_project"
    if not main_path.is_file():
        raise ProvenanceError(f"Missing release entry point: {main_path}")
    if not package_root.is_dir():
        raise ProvenanceError(f"Missing release package: {package_root}")

    candidates = [main_path]
    candidates.extend(package_root.rglob("*.py"))
    scripts_root = root / "scripts"
    if scripts_root.is_dir():
        candidates.extend(scripts_root.rglob("*.py"))

    accepted: list[Path] = []
    for candidate in candidates:
        relative = _inside(root, candidate).relative_to(root)
        if any(part in _EXCLUDED_PARTS for part in relative.parts):
            continue
        accepted.append(candidate.resolve())
    return tuple(sorted(set(accepted), key=lambda path: path.relative_to(root).as_posix()))


def build_code_hashes(
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, str]:
    """Return the complete stable release-code path-to-SHA-256 mapping."""

    root = Path(project_root).resolve()
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in iter_code_files(root)
    }


def build_code_manifest(
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Build the deterministic complete-code manifest."""

    root = Path(project_root).resolve()
    records = _record_files(root, iter_code_files(root))
    package_count = sum(
        record["path"].startswith("src/bul_project/") for record in records
    )
    workflow_count = sum(
        record["path"].startswith("scripts/") for record in records
    )
    payload: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "manifest_role": "complete_release_code",
        "scope": {
            "entry_point": "main.py",
            "package_glob": "src/bul_project/**/*.py",
            "workflow_glob": "scripts/**/*.py",
            "excluded": sorted(_EXCLUDED_PARTS),
        },
        "counts": {
            "total": len(records),
            "entry_points": int(any(row["path"] == "main.py" for row in records)),
            "package_modules": package_count,
            "workflow_modules": workflow_count,
        },
        "files": records,
        "aggregate_sha256": _aggregate_sha256(records),
    }
    return payload


def _git_commit(root: Path) -> str | None:
    """Return the repository commit when Git metadata is available."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip().lower()
    if completed.returncode != 0 or len(value) != 40:
        return None
    try:
        int(value, 16)
    except ValueError:
        return None
    return value


def _environment() -> dict[str, Any]:
    """Return the interpreter and required-package versions."""

    packages: dict[str, str | None] = {}
    for name in _REQUIRED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable_name": Path(sys.executable).name,
        "packages": packages,
    }


def _tree_files(
    root: Path,
    relative_root: str,
    *,
    exclude_names: frozenset[str] = frozenset(),
) -> list[Path]:
    """Return regular files below one optional repository-relative directory."""

    directory = root / relative_root
    if not directory.is_dir():
        return []
    return [
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.name not in exclude_names
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and ".tmp-" not in path.name
    ]


def build_release_manifest(
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Build the deterministic local release-artifact manifest.

    The manifest distinguishes a clean source release from a completed local
    reproduction. Absence of licensed XES files or generated results is
    reported explicitly and is never represented as completed experimentation.
    """

    root = Path(project_root).resolve()
    code = build_code_manifest(root)
    data_records = _record_files(
        root,
        [
            path
            for path in _tree_files(
                root,
                "data/raw",
                exclude_names=frozenset({"README.md", ".gitkeep"}),
            )
            if path.name.lower().endswith((".xes", ".xes.gz"))
        ],
    )
    config_records = _record_files(
        root,
        _tree_files(root, "configs/generated"),
    )
    result_records = _record_files(
        root,
        _tree_files(
            root,
            "results",
            exclude_names=frozenset({"README.md", ".gitkeep"}),
        ),
    )
    paper_records = _record_files(
        root,
        _tree_files(
            root,
            "paper_artifacts",
            exclude_names=frozenset({"README.md", ".gitkeep", "manifest.json"}),
        ),
    )
    documentation_records = _record_files(
        root,
        _tree_files(root, "docs"),
    )
    release_metadata_records = _record_files(
        root,
        [root / name for name in _RELEASE_METADATA],
    )

    is_clean = not data_records and not config_records and not result_records
    payload: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "manifest_role": "paper_artifact_traceability",
        "release_state": (
            "clean_source_release" if is_clean else "local_reproduction"
        ),
        "git_commit": _git_commit(root),
        "code": code,
        "environment": _environment(),
        "release_metadata": _collection(release_metadata_records),
        "documentation": _collection(documentation_records),
        "input_data": _collection(data_records),
        "generated_configuration": _collection(config_records),
        "generated_results": _collection(result_records),
        "paper_artifacts": _collection(paper_records),
        "scientific_guards": {
            "raw_xes_committed": False,
            "historical_results_committed": False,
            "top_k_variants": 100,
            "other_bucket": True,
            "edge_threshold_tau": 10,
            "experiment_seeds": "0-29",
            "main_condition_count": 3150,
            "conditional_dp": True,
            "mathematical_logic_modified_by_provenance": False,
        },
    }
    payload["manifest_sha256"] = hashlib.sha256(_json_bytes(payload)).hexdigest()
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace one UTF-8 JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_bytes(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_provenance_artifacts(
    project_root: str | Path = PROJECT_ROOT,
    *,
    code_output: str | Path = DEFAULT_CODE_MANIFEST,
    manifest_output: str | Path = DEFAULT_ARTIFACT_MANIFEST,
) -> dict[str, Any]:
    """Write the code and paper-artifact manifests in dependency order."""

    root = Path(project_root).resolve()
    code_path = _inside(root, root / Path(code_output))
    manifest_path = _inside(root, root / Path(manifest_output))
    if code_path == manifest_path:
        raise ProvenanceError("Code and artifact manifests require different paths.")

    code_payload = build_code_manifest(root)
    _write_json_atomic(code_path, code_payload)
    manifest_payload = build_release_manifest(root)
    _write_json_atomic(manifest_path, manifest_payload)
    return {
        "status": "complete",
        "release_state": manifest_payload["release_state"],
        "code_manifest": code_path.relative_to(root).as_posix(),
        "code_manifest_sha256": sha256_file(code_path),
        "artifact_manifest": manifest_path.relative_to(root).as_posix(),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "tracked_code_files": code_payload["counts"]["total"],
        "input_data_files": manifest_payload["input_data"]["file_count"],
        "generated_result_files": manifest_payload["generated_results"]["file_count"],
    }


def verify_provenance_artifacts(
    project_root: str | Path = PROJECT_ROOT,
    *,
    code_manifest: str | Path = DEFAULT_CODE_MANIFEST,
    artifact_manifest: str | Path = DEFAULT_ARTIFACT_MANIFEST,
) -> dict[str, Any]:
    """Fail unless stored provenance manifests match the current repository."""

    root = Path(project_root).resolve()
    code_path = _inside(root, root / Path(code_manifest))
    artifact_path = _inside(root, root / Path(artifact_manifest))
    try:
        stored_code = json.loads(code_path.read_text(encoding="utf-8"))
        stored_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"Cannot read stored provenance manifests: {exc}") from exc

    expected_code = build_code_manifest(root)
    expected_artifact = build_release_manifest(root)
    if stored_code != expected_code:
        raise ProvenanceError("Stored code manifest differs from current code.")
    if stored_artifact != expected_artifact:
        raise ProvenanceError(
            "Stored artifact manifest differs from current repository state."
        )
    return {
        "status": "verified",
        "code_manifest_sha256": sha256_file(code_path),
        "artifact_manifest_sha256": sha256_file(artifact_path),
        "tracked_code_files": expected_code["counts"]["total"],
    }


__all__ = [
    "DEFAULT_ARTIFACT_MANIFEST",
    "DEFAULT_CODE_MANIFEST",
    "PROVENANCE_SCHEMA_VERSION",
    "ProvenanceError",
    "build_code_hashes",
    "build_code_manifest",
    "build_release_manifest",
    "iter_code_files",
    "sha256_file",
    "verify_provenance_artifacts",
    "write_provenance_artifacts",
]

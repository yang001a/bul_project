"""Atomic, round-trip-verified artifacts for Phase 6 process quality.

This module is deliberately limited to post-processing objects that already
exist in memory: a :class:`SyntheticLogResult`, a discovered Petri net, and a
``ProcessQualityResult``.  It has no raw-log reader and no interface accepting
raw traces or raw temporal samples.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pm4py
from pm4py.objects.petri_net.obj import Marking, PetriNet

from bul_project.validation.process_model import PetriNetArtifact, ProcessQualityResult
from bul_project.validation.synthetic_log import SyntheticLogResult


CORE_COLUMNS = ("case_id", "activity", "timestamp")
XES_COLUMNS = ("case:concept:name", "concept:name", "time:timestamp")
_TEMPORARY_TOKEN = ".tmp-"
_FORBIDDEN_SAMPLE_KEYS = frozenset({"samples", "samples_seconds"})
TimestampPrecision = Literal["nanosecond", "microsecond"]


class ProcessQualityArtifactError(RuntimeError):
    """Raised when an exported artifact fails its round-trip audit."""


@dataclass(frozen=True)
class FileDigest:
    """Raw digest for one atomically committed auxiliary artifact."""

    filename: str
    byte_size: int
    sha256: str

    def to_dict(self) -> dict[str, str | int]:
        """Return a deterministic JSON-compatible record."""

        return {
            "filename": self.filename,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class VerifiedArtifact:
    """Raw and semantic digests for one verified XES or PNML artifact."""

    filename: str
    format: str
    byte_size: int
    sha256: str
    semantic_sha256: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible verification record."""

        return {
            "filename": self.filename,
            "format": self.format,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "semantic_sha256": self.semantic_sha256,
            "metadata": self.metadata,
        }


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one existing file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_temporary_artifact_path(path: str | Path) -> bool:
    """Return whether a path is an uncommitted temporary artifact path."""

    return _TEMPORARY_TOKEN in Path(path).name


def is_committed_artifact(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> bool:
    """Return whether a non-temporary artifact exists and matches its digest.

    A temporary sibling is never treated as a completed artifact, even when it
    is a valid XES, PNML, CSV, or JSON file.
    """

    candidate = Path(path)
    if is_temporary_artifact_path(candidate) or not candidate.is_file():
        return False
    if expected_sha256 is None:
        return True
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest.")
    return sha256_file(candidate) == expected_sha256


def _sha256_json(payload: Any) -> str:
    """Compute a deterministic SHA-256 digest for JSON content.

    Inputs:
        payload: Structured content to validate, serialize, or write.
    Outputs:
        A value of type `str` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_events(
    events: pd.DataFrame,
    *,
    timestamp_precision: TimestampPrecision,
) -> pd.DataFrame:
    """Return events in the canonical case, activity, and timestamp form.

    Inputs:
        events: Input value used by this operation.
        timestamp_precision: Input value used by this operation.
    Outputs:
        A value of type `pd.DataFrame` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a pandas DataFrame.")
    missing = set(CORE_COLUMNS) - set(events.columns)
    if missing:
        raise ValueError(f"events is missing canonical columns: {missing!r}")
    if events.empty:
        raise ValueError("events must contain at least one event.")
    if timestamp_precision not in ("nanosecond", "microsecond"):
        raise ValueError(
            "timestamp_precision must be 'nanosecond' or 'microsecond'."
        )

    canonical = events.loc[:, CORE_COLUMNS].copy()
    if canonical[["case_id", "activity"]].isna().any().any():
        raise ValueError("events contains missing case_id or activity values.")
    if any(
        not isinstance(value, str) or not value
        for value in canonical["case_id"]
    ):
        raise ValueError("events contains an invalid case_id.")
    if any(
        not isinstance(value, str) or not value
        for value in canonical["activity"]
    ):
        raise ValueError("events contains an invalid activity.")

    timestamps = pd.to_datetime(
        canonical["timestamp"], errors="coerce", utc=True
    )
    if timestamps.isna().any():
        raise ValueError("events contains an invalid timestamp.")
    if timestamp_precision == "microsecond":
        # PM4Py's XES exporter writes six fractional-second digits.  Flooring
        # before export makes that format limitation explicit and gives the
        # source and round-trip log the same semantic representation.
        timestamps = timestamps.dt.floor("us")
    canonical["timestamp"] = timestamps
    canonical["_event_order"] = range(len(canonical))
    canonical = canonical.sort_values(
        ["case_id", "timestamp", "_event_order"], kind="mergesort"
    ).drop(columns="_event_order")
    return canonical.reset_index(drop=True)


def synthetic_event_semantic_sha256(
    events: pd.DataFrame,
    *,
    timestamp_precision: TimestampPrecision = "microsecond",
) -> str:
    """Hash canonical synthetic-event semantics independently of XES bytes.

    Input
    -----
    events:
        A synthetic ``case_id/activity/timestamp`` event table.
    timestamp_precision:
        ``microsecond`` is the XES-compatible default; ``nanosecond`` records
        the full in-memory pandas timestamp precision.

    Output
    ------
    str
        SHA-256 over stable case, activity, ordered-event, and UTC timestamp
        values.

    Paper experiment step
    ---------------------
    Phase 6 artifact audit: identify a generated synthetic log by its event
    semantics rather than by exporter-specific XML byte ordering.
    """

    canonical = _canonical_events(
        events, timestamp_precision=timestamp_precision
    )
    divisor = 1 if timestamp_precision == "nanosecond" else 1_000
    rows = [
        {
            "case_id": case_id,
            "activity": activity,
            "timestamp_utc": int(timestamp.value) // divisor,
        }
        for case_id, activity, timestamp in canonical.itertuples(
            index=False, name=None
        )
    ]
    return _sha256_json(
        {
            "timestamp_unit": (
                "nanoseconds_since_unix_epoch"
                if timestamp_precision == "nanosecond"
                else "microseconds_since_unix_epoch"
            ),
            "events": rows,
        }
    )


def _temporary_sibling(destination: Path, format_suffix: str) -> Path:
    """Create a temporary path next to the destination artifact.

    Inputs:
        destination: Destination path for the generated artifact.
        format_suffix: Input value used by this operation.
    Outputs:
        A value of type `Path` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    return destination.with_name(
        f".{destination.name}{_TEMPORARY_TOKEN}{uuid.uuid4().hex}{format_suffix}"
    )


def _validate_destination(destination: Path) -> None:
    """Validate destination.

    Inputs:
        destination: Destination path for the generated artifact.
    Outputs:
        None.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if is_temporary_artifact_path(destination):
        raise ValueError("The committed destination cannot be a temporary path.")


def _commit_temp(
    temporary: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    """Atomically commit a temporary artifact to its destination.

    Inputs:
        temporary: Input value used by this operation.
        destination: Destination path for the generated artifact.
        overwrite: Input value used by this operation.
    Outputs:
        None.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Artifact already exists: {destination}")
    os.replace(temporary, destination)


def write_synthetic_xes_atomic(
    synthetic_log: SyntheticLogResult,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> VerifiedArtifact:
    """Atomically write and round-trip verify one canonical synthetic XES log.

    Input
    -----
    synthetic_log:
        A DP-only ``SyntheticLogResult``.  No raw log or temporal sample object
        is accepted by this interface.
    destination:
        Final ``.xes.gz`` (preferred) or ``.xes`` path.

    Output
    ------
    VerifiedArtifact
        Committed byte digest, XES-compatible semantic digest, event counts,
        and the full-precision source semantic digest.

    Paper experiment step
    ---------------------
    Phase 6 external validation artifact: persist the DP-consistent synthetic
    discovery log without exposing raw events or ``T'_e`` sample pools.
    """

    if not isinstance(synthetic_log, SyntheticLogResult):
        raise TypeError("synthetic_log must be a SyntheticLogResult.")
    target = Path(destination)
    _validate_destination(target)
    lower_name = target.name.lower()
    if not (lower_name.endswith(".xes") or lower_name.endswith(".xes.gz")):
        raise ValueError("Synthetic log destination must end in .xes or .xes.gz.")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Artifact already exists: {target}")

    canonical = _canonical_events(
        synthetic_log.events, timestamp_precision="microsecond"
    )
    case_count = int(canonical["case_id"].nunique())
    if case_count != synthetic_log.case_count:
        raise ValueError(
            "SyntheticLogResult.case_count differs from the event table."
        )
    semantic_sha256 = synthetic_event_semantic_sha256(
        canonical, timestamp_precision="microsecond"
    )
    full_precision_sha256 = synthetic_event_semantic_sha256(
        synthetic_log.events, timestamp_precision="nanosecond"
    )
    xes_frame = canonical.rename(
        columns={
            "case_id": "case:concept:name",
            "activity": "concept:name",
            "timestamp": "time:timestamp",
        }
    ).loc[:, XES_COLUMNS]

    target.parent.mkdir(parents=True, exist_ok=True)
    format_suffix = ".xes.gz" if lower_name.endswith(".xes.gz") else ".xes"
    temporary = _temporary_sibling(target, format_suffix)
    try:
        pm4py.write_xes(
            xes_frame,
            str(temporary),
            case_id_key="case:concept:name",
            encoding="utf-8",
            show_progress_bar=False,
        )
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise ProcessQualityArtifactError("PM4Py did not write a valid XES file.")
        if format_suffix == ".xes.gz":
            with temporary.open("rb") as handle:
                if handle.read(2) != b"\x1f\x8b":
                    raise ProcessQualityArtifactError(
                        "The .xes.gz artifact is not gzip-compressed."
                    )

        imported = pm4py.read_xes(
            str(temporary),
            return_legacy_log_object=False,
            encoding="utf-8",
            show_progress_bar=False,
        )
        if not isinstance(imported, pd.DataFrame):
            raise ProcessQualityArtifactError(
                "PM4Py XES round-trip did not return a DataFrame."
            )
        missing = set(XES_COLUMNS) - set(imported.columns)
        if missing:
            raise ProcessQualityArtifactError(
                f"Round-trip XES is missing standard columns: {missing!r}"
            )
        roundtrip = imported.loc[:, XES_COLUMNS].rename(
            columns={
                "case:concept:name": "case_id",
                "concept:name": "activity",
                "time:timestamp": "timestamp",
            }
        )
        roundtrip_semantic = synthetic_event_semantic_sha256(
            roundtrip, timestamp_precision="microsecond"
        )
        if roundtrip_semantic != semantic_sha256:
            raise ProcessQualityArtifactError(
                "Synthetic XES semantic digest changed during round-trip."
            )
        if len(roundtrip) != len(canonical):
            raise ProcessQualityArtifactError(
                "Synthetic XES event count changed during round-trip."
            )
        if int(roundtrip["case_id"].nunique()) != case_count:
            raise ProcessQualityArtifactError(
                "Synthetic XES case count changed during round-trip."
            )

        raw_sha256 = sha256_file(temporary)
        size = temporary.stat().st_size
        _commit_temp(temporary, target, overwrite=overwrite)
    finally:
        if temporary.exists():
            temporary.unlink()

    return VerifiedArtifact(
        filename=target.name,
        format="XES.GZ" if format_suffix == ".xes.gz" else "XES",
        byte_size=size,
        sha256=raw_sha256,
        semantic_sha256=semantic_sha256,
        metadata={
            "status": "success",
            "atomic_commit": True,
            "roundtrip_verified": True,
            "compression": "gzip" if format_suffix == ".xes.gz" else "none",
            "event_count": len(canonical),
            "case_count": case_count,
            "seed": synthetic_log.seed,
            "timestamp_timezone": "UTC",
            "xes_timestamp_precision": "microsecond_floor",
            "source_full_precision_semantic_sha256": full_precision_sha256,
            "raw_log_access": False,
            "temporal_samples_serialized": False,
        },
    )


def _node_record(node: PetriNet.Place | PetriNet.Transition) -> dict[str, Any]:
    """Return node record.

    Inputs:
        node: Input value used by this operation.
    Outputs:
        A value of type `dict[str, Any]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if isinstance(node, PetriNet.Place):
        return {"kind": "place", "name": str(node.name)}
    if isinstance(node, PetriNet.Transition):
        return {"kind": "transition", "name": str(node.name)}
    raise TypeError("Petri-net arc endpoint has an unsupported node type.")


def _marking_records(marking: Marking) -> list[dict[str, Any]]:
    """Create records.

    Inputs:
        marking: Input value used by this operation.
    Outputs:
        A value of type `list[dict[str, Any]]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    return sorted(
        (
            {"place": str(place.name), "tokens": int(tokens)}
            for place, tokens in marking.items()
        ),
        key=lambda item: (item["place"], item["tokens"]),
    )


def _petri_semantic_payload(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
) -> dict[str, Any]:
    """Return petri semantic payload.

    Inputs:
        net: Input value used by this operation.
        initial_marking: Input value used by this operation.
        final_marking: Input value used by this operation.
    Outputs:
        A value of type `dict[str, Any]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if not isinstance(net, PetriNet):
        raise TypeError("net must be a PM4Py PetriNet.")
    if not isinstance(initial_marking, Marking) or not isinstance(
        final_marking, Marking
    ):
        raise TypeError("initial_marking and final_marking must be Marking objects.")

    places = sorted(str(place.name) for place in net.places)
    transitions = sorted(
        (
            {
                "name": str(transition.name),
                "label": (
                    None
                    if transition.label is None
                    else str(transition.label)
                ),
            }
            for transition in net.transitions
        ),
        key=lambda item: (item["name"], "" if item["label"] is None else item["label"]),
    )
    arcs = [
        {
            "source": _node_record(arc.source),
            "target": _node_record(arc.target),
            "weight": float(arc.weight),
        }
        for arc in net.arcs
    ]
    arcs.sort(key=lambda item: json.dumps(item, sort_keys=True))
    return {
        "places": places,
        "transitions": transitions,
        "arcs": arcs,
        "initial_marking": _marking_records(initial_marking),
        "final_marking": _marking_records(final_marking),
    }


def petri_net_semantic_sha256(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
) -> str:
    """Hash Petri-net nodes, arcs, labels, weights, and both markings."""

    return _sha256_json(
        _petri_semantic_payload(net, initial_marking, final_marking)
    )


def write_petri_pnml_atomic(
    model: PetriNetArtifact,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> VerifiedArtifact:
    """Atomically write and round-trip verify one discovered Petri net.

    Input
    -----
    model:
        Petri net returned by ``ProcessModelEvaluator`` after discovery from a
        DP synthetic log.
    destination:
        Final ``.pnml`` path.

    Output
    ------
    VerifiedArtifact
        Committed byte and structural semantic digests with model counts.

    Paper experiment step
    ---------------------
    Phase 6 external validation artifact: persist the Inductive Miner model
    used to calculate Fitness, Precision, and Generalization.
    """

    if not isinstance(model, PetriNetArtifact):
        raise TypeError("model must be a PetriNetArtifact.")
    target = Path(destination)
    _validate_destination(target)
    if target.suffix.lower() != ".pnml":
        raise ValueError("Petri-net destination must end in .pnml.")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Artifact already exists: {target}")

    semantic_sha256 = petri_net_semantic_sha256(
        model.net, model.initial_marking, model.final_marking
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(target, ".pnml")
    try:
        pm4py.write_pnml(
            model.net,
            model.initial_marking,
            model.final_marking,
            str(temporary),
            encoding="utf-8",
        )
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise ProcessQualityArtifactError("PM4Py did not write a valid PNML file.")
        roundtrip_net, roundtrip_initial, roundtrip_final = pm4py.read_pnml(
            str(temporary), encoding="utf-8"
        )
        roundtrip_semantic = petri_net_semantic_sha256(
            roundtrip_net, roundtrip_initial, roundtrip_final
        )
        if roundtrip_semantic != semantic_sha256:
            raise ProcessQualityArtifactError(
                "Petri-net semantic digest changed during PNML round-trip."
            )

        raw_sha256 = sha256_file(temporary)
        size = temporary.stat().st_size
        _commit_temp(temporary, target, overwrite=overwrite)
    finally:
        if temporary.exists():
            temporary.unlink()

    return VerifiedArtifact(
        filename=target.name,
        format="PNML",
        byte_size=size,
        sha256=raw_sha256,
        semantic_sha256=semantic_sha256,
        metadata={
            "status": "success",
            "atomic_commit": True,
            "roundtrip_verified": True,
            "places": len(model.net.places),
            "transitions": len(model.net.transitions),
            "arcs": len(model.net.arcs),
            "raw_log_access": False,
            "temporal_samples_serialized": False,
        },
    )


def _condition_scalars(condition: Mapping[str, Any]) -> dict[str, Any]:
    """Create scalars.

    Inputs:
        condition: Input value used by this operation.
    Outputs:
        A value of type `dict[str, Any]` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if not isinstance(condition, Mapping) or not condition:
        raise ValueError("condition must be a non-empty mapping.")
    checked: dict[str, Any] = {}
    for key, value in condition.items():
        if not isinstance(key, str) or not key:
            raise ValueError("condition keys must be non-empty strings.")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError("condition values must be JSON scalar values.")
        if isinstance(value, Real) and not isinstance(value, bool):
            numeric = float(value)
            if not pd.notna(numeric) or numeric in (float("inf"), float("-inf")):
                raise ValueError("condition numeric values must be finite.")
        checked[key] = value
    return dict(sorted(checked.items()))


def _reject_temporal_samples(payload: Any) -> None:
    """Reject temporal samples.

    Inputs:
        payload: Structured content to validate, serialize, or write.
    Outputs:
        None.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key) in _FORBIDDEN_SAMPLE_KEYS:
                raise ValueError("T'_e sample arrays must not be serialized.")
            _reject_temporal_samples(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            _reject_temporal_samples(value)


def _write_bytes_atomic(
    destination: Path,
    data: bytes,
    *,
    overwrite: bool,
) -> FileDigest:
    """Write a byte sequence atomically.

    Inputs:
        destination: Destination path for the generated artifact.
        data: Input value used by this operation.
        overwrite: Input value used by this operation.
    Outputs:
        A value of type `FileDigest` produced by the operation.
    Paper role:
        Artifact provenance and representative external feasibility validation.
    """
    _validate_destination(destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destination, destination.suffix or ".bin")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        digest = sha256_file(temporary)
        size = temporary.stat().st_size
        _commit_temp(temporary, destination, overwrite=overwrite)
    finally:
        if temporary.exists():
            temporary.unlink()
    return FileDigest(destination.name, size, digest)


def write_process_quality_result_csv_atomic(
    condition: Mapping[str, Any],
    quality: ProcessQualityResult,
    synthetic_xes: VerifiedArtifact,
    petri_pnml: VerifiedArtifact,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> FileDigest:
    """Atomically write one scalar process-quality result row."""

    if not isinstance(quality, ProcessQualityResult):
        raise TypeError("quality must be a ProcessQualityResult.")
    checked = _condition_scalars(condition)
    row = {
        **checked,
        "fitness": quality.fitness,
        "precision": quality.precision,
        "generalization": quality.generalization,
        "synthetic_xes_sha256": synthetic_xes.sha256,
        "synthetic_xes_semantic_sha256": synthetic_xes.semantic_sha256,
        "petri_pnml_sha256": petri_pnml.sha256,
        "petri_pnml_semantic_sha256": petri_pnml.semantic_sha256,
    }
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    return _write_bytes_atomic(
        Path(destination),
        buffer.getvalue().encode("utf-8"),
        overwrite=overwrite,
    )


def write_process_quality_metadata_json_atomic(
    condition: Mapping[str, Any],
    synthetic_log: SyntheticLogResult,
    quality: ProcessQualityResult,
    synthetic_xes: VerifiedArtifact,
    petri_pnml: VerifiedArtifact,
    destination: str | Path,
    *,
    temporal_metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> FileDigest:
    """Atomically write one JSON audit record without events or ``T'_e`` arrays."""

    if not isinstance(synthetic_log, SyntheticLogResult):
        raise TypeError("synthetic_log must be a SyntheticLogResult.")
    if not isinstance(quality, ProcessQualityResult):
        raise TypeError("quality must be a ProcessQualityResult.")
    temporal = None if temporal_metadata is None else dict(temporal_metadata)
    payload = {
        "name": "BUL_process_quality_condition",
        "condition": _condition_scalars(condition),
        "result": quality.to_dict(),
        "synthetic_log_generation": synthetic_log.metadata,
        "temporal_generation": temporal,
        "artifacts": {
            "synthetic_xes": synthetic_xes.to_dict(),
            "petri_pnml": petri_pnml.to_dict(),
        },
        "artifact_writer_raw_log_access": False,
        "event_rows_serialized_in_metadata": False,
        "temporal_samples_serialized": False,
    }
    _reject_temporal_samples(payload)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return _write_bytes_atomic(
        Path(destination), encoded, overwrite=overwrite
    )


__all__ = [
    "FileDigest",
    "ProcessQualityArtifactError",
    "VerifiedArtifact",
    "is_committed_artifact",
    "is_temporary_artifact_path",
    "petri_net_semantic_sha256",
    "sha256_file",
    "synthetic_event_semantic_sha256",
    "write_petri_pnml_atomic",
    "write_process_quality_metadata_json_atomic",
    "write_process_quality_result_csv_atomic",
    "write_synthetic_xes_atomic",
]

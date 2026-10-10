"""Typed data contracts shared by the Phase 1 preprocessing modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from bul_project.config import DatasetSpec, PreprocessingConfig

if TYPE_CHECKING:
    from .dataset_report import DatasetReport


@dataclass(frozen=True)
class CleaningReport:
    """Audit trail for all row-level decisions made during cleaning."""

    dataset: str
    input_events: int
    output_events: int
    missing_case_id_events: int
    missing_activity_events: int
    dropped_core_events: int
    missing_timestamp_events: int
    invalid_timestamp_events: int
    unavailable_timestamp_events: int
    exact_duplicate_events: int
    dropped_exact_duplicates: int
    negative_intervals_before_sort: int
    timestamp_tie_events: int
    retained_extension_columns: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Inputs:
            None.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Event-log ingestion, cleaning, and trace construction.
        """
        data = asdict(self)
        data["retained_extension_columns"] = list(self.retained_extension_columns)
        return data


@dataclass(frozen=True)
class Trace:
    """One case-level activity trace in stable timestamp order.

    ``transition_seconds`` contains one item per adjacent event pair. ``None``
    means that at least one timestamp in the pair was unavailable. Negative
    values, if present in caller-supplied unsorted data, are preserved rather
    than converted to absolute values.
    """

    case_id: str
    activities: tuple[str, ...]
    timestamps: tuple[pd.Timestamp | None, ...]
    transition_seconds: tuple[float | None, ...]

    @property
    def length(self) -> int:
        """Return length.

        Inputs:
            None.
        Outputs:
            A value of type `int` produced by the operation.
        Paper role:
            Event-log ingestion, cleaning, and trace construction.
        """
        return len(self.activities)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Inputs:
            None.
        Outputs:
            A value of type `dict[str, Any]` produced by the operation.
        Paper role:
            Event-log ingestion, cleaning, and trace construction.
        """
        return {
            "case_id": self.case_id,
            "activities": list(self.activities),
            "timestamps": [
                value.isoformat() if value is not None else None
                for value in self.timestamps
            ],
            "transition_seconds": list(self.transition_seconds),
        }


@dataclass
class Phase1Result:
    """Complete in-memory output of one Phase 1 preprocessing run."""

    dataset: str
    dataset_spec: DatasetSpec
    preprocessing_config: PreprocessingConfig
    reader: str
    source_path: str
    source_sha256: str
    events: pd.DataFrame
    traces: list[Trace]
    cleaning_report: CleaningReport
    dataset_report: "DatasetReport"

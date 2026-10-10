"""Canonical field mapping, validation, cleaning, and stable event ordering."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from bul_project.config import CORE_COLUMNS, DatasetSpec, PreprocessingConfig
from .models import CleaningReport


class SchemaError(ValueError):
    """Raised when an input log does not provide the configured core fields."""


def _blank_or_missing(series: pd.Series) -> pd.Series:
    """Return blank or missing.

    Inputs:
        series: Input value used by this operation.
    Outputs:
        A value of type `pd.Series` produced by the operation.
    Paper role:
        Event-log ingestion, cleaning, and trace construction.
    """
    text = series.astype("string")
    return series.isna() | text.str.strip().eq("").fillna(False)


def _validate_unique_columns(columns: Iterable[object]) -> None:
    """Validate unique columns.

    Inputs:
        columns: Input value used by this operation.
    Outputs:
        None.
    Paper role:
        Event-log ingestion, cleaning, and trace construction.
    """
    index = pd.Index(columns)
    duplicates = index[index.duplicated()].unique().tolist()
    if duplicates:
        raise SchemaError(f"Input contains duplicate column names: {duplicates}")


def _count_negative_intervals(events: pd.DataFrame) -> int:
    """Count negative intervals.

    Inputs:
        events: Input value used by this operation.
    Outputs:
        A value of type `int` produced by the operation.
    Paper role:
        Event-log ingestion, cleaning, and trace construction.
    """
    timed = events.loc[events["timestamp"].notna(), ["case_id", "timestamp"]]
    if timed.empty:
        return 0
    deltas = timed.groupby("case_id", sort=False, observed=True)["timestamp"].diff()
    return int((deltas.dt.total_seconds() < 0).sum())


def normalize_and_clean(
    raw_events: pd.DataFrame,
    dataset: DatasetSpec,
    policy: PreprocessingConfig,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Map an XES table to the canonical event schema and clean it.

    The output is stably ordered by ``case_id``, parsed UTC timestamp, and the
    original event position. Missing timestamps are retained and placed last
    within their case. Equal timestamps preserve source order. No time delta is
    converted to an absolute value.
    """

    if not isinstance(raw_events, pd.DataFrame):
        raise TypeError("raw_events must be a pandas.DataFrame")
    if not policy.timestamp_utc:
        raise ValueError("Phase 1 requires timestamp_utc=True.")
    _validate_unique_columns(raw_events.columns)

    source_columns = {
        "case_id": dataset.case_column,
        "activity": dataset.activity_column,
        "timestamp": dataset.timestamp_column,
    }
    missing = [column for column in source_columns.values() if column not in raw_events]
    if missing:
        raise SchemaError(
            f"Dataset {dataset.name} is missing required XES columns: {missing}"
        )

    work = pd.DataFrame(
        {
            canonical: raw_events[source].copy()
            for canonical, source in source_columns.items()
        }
    )
    work["_source_order"] = range(len(work))

    missing_case = _blank_or_missing(work["case_id"])
    missing_activity = _blank_or_missing(work["activity"])
    dropped_core = missing_case | missing_activity

    raw_timestamp_missing = _blank_or_missing(work["timestamp"])
    parsed_timestamp = pd.to_datetime(
        work["timestamp"], errors="coerce", utc=True, format="mixed"
    )
    invalid_timestamp = (~raw_timestamp_missing) & parsed_timestamp.isna()
    work["timestamp"] = parsed_timestamp

    # StringDtype preserves missing values instead of turning them into "nan".
    work["case_id"] = work["case_id"].astype("string")
    work["activity"] = work["activity"].astype("string")

    retained = work.loc[~dropped_core].copy()
    duplicate_mask = retained.duplicated(
        subset=list(CORE_COLUMNS), keep="first"
    )
    exact_duplicate_events = int(duplicate_mask.sum())
    dropped_exact_duplicates = 0
    if policy.drop_exact_duplicates:
        retained = retained.loc[~duplicate_mask].copy()
        dropped_exact_duplicates = exact_duplicate_events

    negative_before_sort = _count_negative_intervals(retained)
    tie_mask = retained["timestamp"].notna() & retained.duplicated(
        subset=["case_id", "timestamp"], keep=False
    )

    extension_columns: list[str] = []
    if policy.keep_extensions:
        required_sources = set(source_columns.values())
        extension_columns = [
            str(column)
            for column in raw_events.columns
            if column not in required_sources and str(column) not in CORE_COLUMNS
        ]
        if extension_columns:
            extension_values = raw_events.loc[retained.index, extension_columns].copy()
            retained = retained.join(extension_values)

    retained = retained.sort_values(
        ["case_id", "timestamp", "_source_order"],
        kind=policy.stable_sort_kind,
        na_position=policy.missing_timestamp_position,
        ignore_index=True,
    )
    retained = retained.drop(columns="_source_order")
    ordered_columns = list(CORE_COLUMNS) + extension_columns
    retained = retained.loc[:, ordered_columns]

    report = CleaningReport(
        dataset=dataset.name,
        input_events=int(len(raw_events)),
        output_events=int(len(retained)),
        missing_case_id_events=int(missing_case.sum()),
        missing_activity_events=int(missing_activity.sum()),
        dropped_core_events=int(dropped_core.sum()),
        missing_timestamp_events=int(raw_timestamp_missing.sum()),
        invalid_timestamp_events=int(invalid_timestamp.sum()),
        unavailable_timestamp_events=int(retained["timestamp"].isna().sum()),
        exact_duplicate_events=exact_duplicate_events,
        dropped_exact_duplicates=dropped_exact_duplicates,
        negative_intervals_before_sort=negative_before_sort,
        timestamp_tie_events=int(tie_mask.sum()),
        retained_extension_columns=tuple(extension_columns),
    )
    return retained, report

"""Phase 1: canonical XES ingestion and case-level trace construction."""

from .cleaning import SchemaError, normalize_and_clean
from .dataset_report import DatasetReport, build_dataset_report
from .models import CleaningReport, Phase1Result, Trace
from .pipeline import EmptyEventLogError, preprocess_dataset, write_phase1_artifacts
from .traces import build_traces
from .xes_reader import XESReadError, read_xes_dataframe

__all__ = [
    "CleaningReport",
    "DatasetReport",
    "EmptyEventLogError",
    "Phase1Result",
    "SchemaError",
    "Trace",
    "XESReadError",
    "build_dataset_report",
    "build_traces",
    "normalize_and_clean",
    "preprocess_dataset",
    "read_xes_dataframe",
    "write_phase1_artifacts",
]

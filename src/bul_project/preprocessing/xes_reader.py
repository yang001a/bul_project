"""Offset-safe streaming XES input helpers for Phase 1."""

from __future__ import annotations

import hashlib
import gzip
from pathlib import Path
from typing import BinaryIO

import pandas as pd
from lxml import etree

from bul_project.config import SUPPORTED_INPUT_SUFFIXES


class XESReadError(RuntimeError):
    """Raised when an XES file cannot be converted to a pandas DataFrame."""


def _has_supported_suffix(path: Path) -> bool:
    """Return has supported suffix.

    Inputs:
        path: Path to the input or output artifact.
    Outputs:
        A value of type `bool` produced by the operation.
    Paper role:
        Event-log ingestion, cleaning, and trace construction.
    """
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_INPUT_SUFFIXES)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of an input artifact without loading it whole."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _open_xes(path: Path) -> BinaryIO:
    """Open an XES or compressed XES event-log stream.

    Inputs:
        path: Path to the input or output artifact.
    Outputs:
        A value of type `BinaryIO` produced by the operation.
    Paper role:
        Event-log ingestion, cleaning, and trace construction.
    """
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def _local_name(element: etree._Element) -> str:
    """Return local name.

    Inputs:
        element: Input value used by this operation.
    Outputs:
        A value of type `str` produced by the operation.
    Paper role:
        Event-log ingestion, cleaning, and trace construction.
    """
    return etree.QName(element).localname


def _scalar_value(element: etree._Element) -> object:
    """Convert a top-level XES scalar while keeping dates as ISO strings."""

    value = element.get("value")
    if value is None:
        return None
    kind = _local_name(element)
    if kind == "int":
        try:
            return int(value)
        except ValueError:
            return value
    if kind == "float":
        try:
            return float(value)
        except ValueError:
            return value
    if kind == "boolean":
        return value.strip().lower() == "true"
    # Keeping XES dates as their original ISO-8601 text is intentional. Some
    # PM4Py date parsers replace +01/+02 with UTC without applying the offset.
    # normalize_and_clean performs the correct conversion exactly once.
    return value


def _direct_attributes(element: etree._Element) -> dict[str, object]:
    """Return direct attributes.

    Inputs:
        element: Input value used by this operation.
    Outputs:
        A value of type `dict[str, object]` produced by the operation.
    Paper role:
        Event-log ingestion, cleaning, and trace construction.
    """
    attributes: dict[str, object] = {}
    for child in element:
        if _local_name(child) == "event":
            continue
        key = child.get("key")
        if key is not None and child.get("value") is not None:
            attributes[key] = _scalar_value(child)
    return attributes


def read_xes_dataframe(
    path: str | Path, *, keep_extensions: bool = False
) -> pd.DataFrame:
    """Read ``.xes`` or ``.xes.gz`` with an offset-safe streaming parser.

    Parameters
    ----------
    path:
        Local XES file. Paths containing spaces and gzip-compressed XES files
        are supported.
    keep_extensions:
        Preserve top-level event and trace attributes. Trace attributes receive
        the standard ``case:`` prefix. The default reads only the three fields
        required by the BUL core and has a much smaller memory footprint.

    Returns
    -------
    pandas.DataFrame
        XES event rows in source order. Timestamp values remain original ISO
        strings until ``normalize_and_clean`` converts them correctly to UTC.
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"XES input does not exist: {source}")
    if not _has_supported_suffix(source):
        supported = ", ".join(SUPPORTED_INPUT_SUFFIXES)
        raise ValueError(f"Unsupported input {source.name!r}; expected {supported}.")

    case_ids: list[object] = []
    activities: list[object] = []
    timestamps: list[object] = []
    extension_rows: list[dict[str, object]] = []
    try:
        with _open_xes(source) as stream:
            context = etree.iterparse(
                stream,
                events=("end",),
                huge_tree=True,
                recover=False,
            )
            for _, trace in context:
                if _local_name(trace) != "trace":
                    continue
                trace_attributes = _direct_attributes(trace)
                case_id = trace_attributes.get("concept:name")
                case_attributes = {
                    f"case:{key}": value
                    for key, value in trace_attributes.items()
                }
                for child in trace:
                    if _local_name(child) != "event":
                        continue
                    event_attributes = _direct_attributes(child)
                    case_ids.append(case_id)
                    activities.append(event_attributes.get("concept:name"))
                    timestamps.append(event_attributes.get("time:timestamp"))
                    if keep_extensions:
                        row = dict(event_attributes)
                        row.update(case_attributes)
                        extension_rows.append(row)

                trace.clear()
                parent = trace.getparent()
                if parent is not None:
                    while trace.getprevious() is not None:
                        del parent[0]
    except (OSError, EOFError, etree.XMLSyntaxError) as exc:
        raise XESReadError(f"Failed to read XES input {source}: {exc}") from exc

    if keep_extensions:
        result = pd.DataFrame.from_records(extension_rows)
        # These assignments also guarantee the required columns for logs whose
        # optional attributes differ from trace to trace.
        result["case:concept:name"] = case_ids
        result["concept:name"] = activities
        result["time:timestamp"] = timestamps
        return result
    return pd.DataFrame(
        {
            "case:concept:name": case_ids,
            "concept:name": activities,
            "time:timestamp": timestamps,
        }
    )

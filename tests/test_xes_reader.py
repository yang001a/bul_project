from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import pytest

from bul_project.preprocessing.xes_reader import read_xes_dataframe, sha256_file


def test_read_plain_xes_preserves_trace_and_event_concept_names(
    demo_path: Path,
) -> None:
    frame = read_xes_dataframe(demo_path)

    assert len(frame) == 8
    assert "case:concept:name" in frame
    assert "concept:name" in frame
    assert "time:timestamp" in frame
    assert frame.loc[0, "case:concept:name"] == "case-1"
    assert frame.loc[0, "concept:name"] == "Start"
    register = frame.loc[frame["concept:name"] == "Register", "time:timestamp"].iloc[0]
    assert register == "2024-01-02T09:00:00.000+01:00"


def test_read_gzip_xes_with_spaces_in_path(demo_path: Path, tmp_path: Path) -> None:
    compressed = tmp_path / "demo event log.xes.gz"
    with demo_path.open("rb") as source, gzip.open(compressed, "wb") as target:
        shutil.copyfileobj(source, target)

    frame = read_xes_dataframe(compressed)

    assert len(frame) == 8
    assert set(frame["case:concept:name"]) == {"case-1", "case-2", "case-3"}


def test_read_namespaced_xes(demo_path: Path, tmp_path: Path) -> None:
    text = demo_path.read_text(encoding="utf-8")
    text = text.replace(
        '<log xes.version="1.0"',
        '<log xmlns="http://www.xes-standard.org/" xes.version="1.0"',
        1,
    )
    namespaced = tmp_path / "namespaced.xes"
    namespaced.write_text(text, encoding="utf-8")

    frame = read_xes_dataframe(namespaced)

    assert len(frame) == 8
    assert frame["case:concept:name"].nunique() == 3


def test_rejects_missing_and_unsupported_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_xes_dataframe(tmp_path / "missing.xes")

    unsupported = tmp_path / "events.csv"
    unsupported.write_text("case,activity\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported input"):
        read_xes_dataframe(unsupported)


def test_sha256_is_deterministic(demo_path: Path) -> None:
    first = sha256_file(demo_path)
    second = sha256_file(demo_path)

    assert first == second
    assert len(first) == 64

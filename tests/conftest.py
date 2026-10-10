from __future__ import annotations

import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bul_project.config import DatasetSpec
from bul_project.preprocessing.models import Trace
from bul_project.preprocessing.pipeline import preprocess_dataset


@pytest.fixture(scope="session")
def demo_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "demo" / "demo_log.xes"


@pytest.fixture(scope="session")
def demo_spec(demo_path: Path) -> DatasetSpec:
    return DatasetSpec(name="DEMO", path=demo_path, role="unit test")


@pytest.fixture(scope="session")
def demo_traces(demo_spec: DatasetSpec) -> list[Trace]:
    return preprocess_dataset(demo_spec).traces

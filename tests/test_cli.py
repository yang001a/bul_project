from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bul_project.cli import COMMAND_MODULES
from scripts.reproduce_paper import _workflow_commands
from scripts.run_demo import main as run_demo


def test_public_command_registry_contains_reproduction_and_demo() -> None:
    assert tuple(COMMAND_MODULES) == (
        "preprocess",
        "freeze-parameters",
        "run-main",
        "derive-bul",
        "run-sensitivity",
        "run-external-validation",
        "analyze",
        "demo",
        "build-manifest",
        "reproduce-paper",
    )


def test_reproduce_paper_sequence_preserves_frozen_workflow() -> None:
    commands = _workflow_commands(
        skip_tests=True,
        include_external_validation=False,
    )
    public = [command[2:] for command in commands]
    assert public == [
        ["freeze-parameters", "--output-dir", "configs/generated"],
        ["run-main", "--plan-only"],
        ["run-main", "--execute-numeric-reference"],
        ["derive-bul"],
        ["run-sensitivity", "--execute"],
        ["analyze"],
        ["build-manifest"],
    ]


def test_demo_is_scientifically_reproducible_for_fixed_seed(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    arguments = ["--epsilon", "8", "--seed", "0"]

    assert run_demo([*arguments, "--output-dir", str(first_dir)]) == 0
    assert run_demo([*arguments, "--output-dir", str(second_dir)]) == 0

    first = pd.read_csv(first_dir / "experiment_result.csv")
    second = pd.read_csv(second_dir / "experiment_result.csv")
    comparable = [column for column in first.columns if column != "runtime"]
    pd.testing.assert_frame_equal(first[comparable], second[comparable])

    metadata = json.loads(
        (first_dir / "experiment_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["config"]["dataset"] == "DEMO"
    assert metadata["config"]["top_k_variants"] == 100
    assert metadata["config"]["min_edge_support"] == 1
    assert metadata["config"]["budget_strategy"] == "uniform"
    assert (
        metadata["analysis_metadata"]["comparison_stages"]["additivity_claimed"]
        is False
    )

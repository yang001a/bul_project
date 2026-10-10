from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_release_functions_have_no_encoding_damaged_docstrings() -> None:
    """Reject the uniform question-mark template introduced by encoding damage."""

    roots = (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts")
    python_files = [PROJECT_ROOT / "main.py"]
    for root in roots:
        python_files.extend(root.rglob("*.py"))

    damaged: list[str] = []
    for path in python_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            docstring = ast.get_docstring(node, clean=False)
            if docstring and "???" in docstring:
                relative = path.relative_to(PROJECT_ROOT)
                damaged.append(f"{relative}:{node.lineno}:{node.name}")

    assert damaged == []


def test_public_artifact_documents_and_output_directory_are_consistent() -> None:
    """Keep public-artifact instructions aligned with its initially empty tree."""

    active_documents = (
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "reproducibility.md",
        PROJECT_ROOT / "configs" / "README.md",
        PROJECT_ROOT / "results" / "README.md",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in active_documents)

    assert "configs/generated" in combined
    assert "results/main_experiment" in combined
    assert "python main.py freeze-parameters" in combined
    assert list((PROJECT_ROOT / "results").rglob("*")) == [
        PROJECT_ROOT / "results" / "README.md"
    ]

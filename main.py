"""Thin command-line entry point for the BUL research artifact.

The implementation lives in :mod:`bul_project.cli`. This file only makes the
``src`` package importable when the repository is executed without installing
it and then delegates to the package-level CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bul_project.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

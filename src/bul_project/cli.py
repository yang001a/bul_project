"""Unified command-line interface for the BUL research artifact.

Function:
    Dispatch preprocessing, parameter freezing, main experiments, derived BUL
    analysis, sensitivity analysis, representative external validation,
    statistical analysis, the quick demo, and the complete paper workflow.
Input:
    A command name followed by arguments owned by that command.
Output:
    The selected command's process exit code.
Paper role:
    Reproducibility orchestration only; this module implements no mathematical
    formula and changes no experiment parameter.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from types import ModuleType
from typing import Sequence

from bul_project.logging_utils import configure_logging


COMMAND_MODULES = {
    "preprocess": "scripts.preprocess_data",
    "freeze-parameters": "scripts.freeze_experiment_parameters",
    "run-main": "scripts.run_main_experiment",
    "derive-bul": "scripts.derive_bul_results",
    "run-sensitivity": "scripts.run_sensitivity_analysis",
    "run-external-validation": "scripts.run_external_validation",
    "analyze": "scripts.run_statistical_analysis",
    "demo": "scripts.run_demo",
    "build-manifest": "scripts.build_provenance_manifest",
    "reproduce-paper": "scripts.reproduce_paper",
}


def _build_parser() -> argparse.ArgumentParser:
    """Build the repository-level command parser."""

    parser = argparse.ArgumentParser(
        description="BUL SCI paper artifact: reproducible experiment commands."
    )
    parser.add_argument(
        "command",
        choices=tuple(COMMAND_MODULES),
        help="Artifact task to execute.",
    )
    parser.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded unchanged to the selected task.",
    )
    return parser


def _load_command(command: str) -> ModuleType:
    """Import one command module from the fixed public-command registry."""

    try:
        module_name = COMMAND_MODULES[command]
    except KeyError as exc:
        raise ValueError(f"Unknown artifact command: {command}") from exc
    module = importlib.import_module(module_name)
    if not callable(getattr(module, "main", None)):
        raise RuntimeError(f"Command module has no callable main(): {module_name}")
    return module


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one artifact command and return its exit status."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    if not any(value in {"-h", "--help"} for value in args.arguments):
        configure_logging("bul_artifact")
        logging.getLogger(__name__).info(
            "Dispatching artifact command: %s", args.command
        )
    module = _load_command(args.command)
    return int(module.main(list(args.arguments)))


__all__ = ["COMMAND_MODULES", "main"]

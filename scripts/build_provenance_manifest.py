"""Build or verify deterministic BUL paper-artifact provenance manifests."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from bul_project.config import PROJECT_ROOT
from bul_project.provenance import (
    DEFAULT_ARTIFACT_MANIFEST,
    DEFAULT_CODE_MANIFEST,
    ProvenanceError,
    verify_provenance_artifacts,
    write_provenance_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    """Build the provenance command parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build or verify deterministic code, environment, data, config, "
            "result, and compact paper-artifact manifests."
        )
    )
    parser.add_argument(
        "--code-output",
        type=Path,
        default=DEFAULT_CODE_MANIFEST,
        help="Repository-relative complete-code manifest path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_MANIFEST,
        help="Repository-relative paper-artifact manifest path.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing manifests without rewriting them.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build or verify the release provenance artifacts."""

    args = _parser().parse_args(argv)
    try:
        if args.verify:
            summary = verify_provenance_artifacts(
                PROJECT_ROOT,
                code_manifest=args.code_output,
                artifact_manifest=args.output,
            )
        else:
            summary = write_provenance_artifacts(
                PROJECT_ROOT,
                code_output=args.code_output,
                manifest_output=args.output,
            )
    except ProvenanceError as exc:
        logging.getLogger(__name__).error("Provenance operation failed: %s", exc)
        return 2
    logging.getLogger(__name__).info(
        "Provenance summary: %s",
        json.dumps(summary, ensure_ascii=False, sort_keys=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

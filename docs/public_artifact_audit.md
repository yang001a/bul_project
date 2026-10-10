# Public artifact audit

## Initial-state policy

The repository contains source code, unit tests, a small non-sensitive Demo XES
file, environment specifications, and user-facing documentation only. It does
not contain benchmark XES files, data-derived configuration artifacts, prior
experiment outputs, checkpoints, plots, run logs, or provenance snapshots of
an earlier completed experiment.

## Local-generation policy

Users generate `configs/generated/` from their own licensed XES data by
running `freeze-parameters`. All later outputs are local and Git-ignored.

## Verification at packaging time

- Non-integration test suite: 310 passed, 10 skipped, 3 deselected.
- `data/raw/`: only `.gitkeep` and user-data instructions.
- `configs/`: only its generation instructions.
- `results/`: only its generation instructions.
- `logs/`: only its generation instructions.
- The only bundled XES file is `data/demo/demo_log.xes`.

## Deliberately excluded material

- BPIC2012, BPIC2017, and Sepsis Cases logs;
- fixed source-path/hash bindings from a local installation;
- historical Phase 7C diagnostic tables and normalization files;
- completed main, derived, sensitivity, external-validation, statistical, and
  figure artifacts;
- prior check-point lineage and frozen-source snapshots.

## Maintainer action before public GitHub release

Select an approved license and add a `LICENSE` file. Add the accepted-paper
citation and author metadata when available.

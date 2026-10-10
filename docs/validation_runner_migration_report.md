# Validation Runner Migration Report

## Scope

Stage 3A established one semantically named entry module for representative
external feasibility validation. No statistical definition, DP mechanism,
synthetic-log rule, PM4Py metric, experimental condition, seed, or existing
result was changed.

## File migration

```text
src/bul_project/validation/formal_runner.py
→ src/bul_project/validation/runner.py

tests/test_process_quality_formal_runner.py
→ tests/test_external_validation_runner.py
```

The external-validation command now imports `bul_project.validation.runner`.
The runner's provenance file list was updated to hash the new path.

## Compatibility

- Legacy `validation.formal_runner` references: 0
- Canonical `validation.runner` exists: yes
- Old runner path exists: no
- External-validation CLI options: unchanged
- Frozen three-condition scope: unchanged
- Checkpoint and manifest behavior: unchanged

## Verification

- Package modules discovered: 57
- Module import failures: 0
- `python main.py run-external-validation --help`: passed
- External runner resume/idempotence tests: passed
- Full non-integration suite: 311 passed, 10 skipped, 3 deselected
- Test failures: 0

## Boundary

This stage did not merge `scope.py`, `release.py`, `artifacts.py`, or
`evaluation_sample.py`. Those modules contain DP-only input guards,
authenticated-checkpoint handling, artifact round-trip verification, and
evaluation-log selection logic. They require a separate function-level
consolidation audit before any deletion.

## Status

Stage 3A passed. The external validation workflow now has a natural public
runner name while preserving its existing implementation and tests.

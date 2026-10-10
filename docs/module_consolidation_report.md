# Module Consolidation Report

## Scope

Stage 2 consolidated the two duplicated external-validation implementations in
`BUL_Project_release_artifact_version`. No preprocessing, behavior-statistics,
privacy, metric, BUL, parameter, seed, experiment-matrix, or frozen-result
definition was changed. The 3,150-condition experiment was not executed.

## Canonical modules

The release now uses these single implementations:

```text
bul_project.validation.synthetic_log
bul_project.validation.process_model
```

The duplicate files formerly under `bul_project.experiments` were removed.
Imports in experiment orchestration, validation, package exports, and tests now
refer to the canonical validation modules.

## Static equivalence

After removing docstrings, the complete function and class ASTs of both
canonical modules were compared with their frozen experiment counterparts.

| Comparison | Definition names | Executable definitions |
|---|---|---|
| process-model evaluator | identical | identical |
| synthetic-log generator | identical | identical |

No legacy import of `experiments.process_model_evaluator` or
`experiments.synthetic_log_generator` remains.

## Numerical regression

With the same DP statistics and seed 5, the frozen and consolidated synthetic
logs were exactly identical. Their canonical serialized payload SHA-256 was:

```text
2bd886f467ed0afbaa19e7d2c2b5ed332fe20cffe1396be5b6a252e6492cac11
```

The representative token-based PM4Py evaluation also produced identical
scientific outputs:

```text
fitness        = 1.0
precision      = 1.0
generalization = 0.7300827181165915
places         = 4
transitions    = 4
```

Only wall-clock runtime metadata differed, as expected for separate process
executions.

## Protected scientific modules

Byte-level SHA-256 comparison against the frozen experiment version found zero
mismatches across all 36 Python files in:

```text
preprocessing/
behavior/
privacy/
metrics/
bul/
```

## Runtime verification

- Legacy duplicate-module references: 0
- Package modules discovered: 57
- Module import failures: 0
- `main.py --help`: passed
- Unit tests: 311 passed, 10 skipped, 3 deselected, 0 failed
- Existing frozen manifest hashes: unchanged

## Status

Stage 2 passed. Duplicate implementations were removed without changing their
executable definitions or numerical outputs. Phase-name cleanup, CLI
consolidation, and validation-wrapper simplification remain separate later
stages.

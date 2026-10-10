# Release Artifact Cleanup Report

## Scope

Stage 1 cleaned only `BUL_Project_release_artifact_version`. The original
`BUL_Project_open_source` and `BUL_Project_frozen_experiment_version` were not
modified. No experiment was executed and no Python source, configuration,
formula, default parameter, seed, or existing result was changed.

## Cleanup method

The release was rebuilt from an explicit source whitelist. The Stage 0 full
copy was retained non-destructively as
`BUL_Project_release_artifact_version_stage0_full_copy`.

The following local runtime content was excluded from the clean release:

| Category | Files | Bytes |
|---|---:|---:|
| IDE metadata | 7 | 4,877 |
| pytest cache | 5 | 32,219 |
| Python bytecode caches | 93 | 1,797,991 |
| runtime logs | 7 | 1,820,493 |
| licensed raw XES logs | 3 | 658,482,888 |
| generated configuration | 12 | 1,628,320 |
| generated results | 10,211 | 1,401,789,575 |

The clean release retains the demo XES, raw-data instructions, configuration
instructions, result instructions, source, tests, documentation, and locked
environment files.

## Source integrity

- Frozen Python files expected: 98
- Python files present in release: 98
- Missing Python files: 0
- SHA-256 mismatches: 0
- Forbidden raw-data/result/cache files after cleanup: 0

## Verification

The non-integration test suite completed with:

```text
311 passed
10 skipped
3 deselected
22 warnings
0 failed
```

The all-module import audit discovered 59 modules and one pre-existing import
blocker:

```text
bul_project.validation.process_model
ModuleNotFoundError:
No module named 'bul_project.validation.synthetic_log_generator'
```

This unused duplicate validation module is not exercised by the current test
suite. It is the identified Stage 2 consolidation target; Stage 1 intentionally
does not modify Python code to resolve it.

## Paper artifact skeleton

The release now contains empty tracked directories for compact paper tables,
figures, CSV files, and frozen configurations. No historical experimental
result was copied into them during Stage 1.

## Status

Repository cleanup and source-integrity verification passed. Full module-import
verification remains blocked by the pre-existing duplicate-module import issue
and must be resolved during Stage 2 before the repository can be declared
release-ready.

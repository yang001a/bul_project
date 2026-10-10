# Validation Support Module Audit

## Scope

This Stage 3B audit examines the four support modules surrounding the
representative PM4Py external-feasibility validation:

- `src/bul_project/validation/scope.py`
- `src/bul_project/validation/release.py`
- `src/bul_project/validation/artifacts.py`
- `src/bul_project/validation/evaluation_sample.py`

The audit is read-only with respect to Python source, configuration, frozen
results, and experiment parameters. It does not run the 3150-condition
experiment.

## Function-level findings

| Module | Size | Scientific/reproducibility responsibility | Active consumers | Dedicated tests | Decision |
|---|---:|---|---|---:|---|
| `scope.py` | 948 lines | Freezes the three representative conditions, authenticates reference sources, selects the preregistered seed independently of PM4Py outcomes, and prevents silent scope drift. | `validation/runner.py`, `scripts/run_external_validation.py` | 5 direct scope tests plus runner integration tests | Retain |
| `release.py` | 714 lines | Strictly parses authenticated DP-full checkpoints, verifies condition identity and temporal reconstruction metadata, excludes `OTHER`, and regenerates the synthetic log using DP outputs only. | `validation/runner.py` | 10 direct tests | Retain |
| `artifacts.py` | 825 lines | Atomically writes and round-trip verifies XES, PNML, CSV, and JSON artifacts; records byte and semantic hashes; rejects raw temporal sample arrays. | `validation/runner.py` | 6 direct tests | Retain |
| `evaluation_sample.py` | 233 lines | Creates and authenticates one deterministic, whole-case BPIC2017 evaluation sample that is reused across representative conditions and never enters DP generation. | `validation/scope.py`, `scripts/run_external_validation.py` | 3 direct tests plus scope tests | Retain |

All four modules have production callers. None is an unused wrapper or a
second implementation of `synthetic_log.py`, `process_model.py`, or
`runner.py`.

## Duplicate-code check

Normalized top-level AST comparison found only two exact internal helper
duplicates:

- `_json_bytes`: `runner.py` and `scope.py` (20 lines each)
- `_sha256_bytes`: `runner.py` and `scope.py` (11 lines each)

This is 31 duplicated source lines per module pair, not a duplicated scientific
component. Extracting these private helpers solely to reduce the line count
would add another shared utility boundary and create unnecessary migration
risk. They should remain unchanged during the scientific-equivalence refactor.

No top-level function or class in `release.py`, `artifacts.py`, or
`evaluation_sample.py` is an exact duplicate of another validation module.

## Why a three-file validation package is unsafe

Merging the four support modules into `runner.py` would create a module of
approximately 4,572 lines before imports and integration edits. It would mix
five separate concerns:

1. external-validation protocol freeze;
2. authenticated DP checkpoint parsing;
3. deterministic evaluation-log construction;
4. artifact serialization and integrity checks;
5. execution and resume orchestration.

That structure would be harder to review and test, would obscure the boundary
that prevents raw-log access during DP post-processing, and would not remove
scientific complexity. The original three-file target
(`synthetic_log.py`, `process_model.py`, `runner.py`) is therefore rejected as
an unsafe cosmetic consolidation.

## Recommended release structure

Keep the seven responsibility-focused modules:

```text
validation/
├── synthetic_log.py
├── process_model.py
├── runner.py
├── scope.py
├── release.py
├── artifacts.py
└── evaluation_sample.py
```

The current names are understandable and correspond to distinct artifacts or
boundaries. Optional future renaming for additional clarity may be considered
only as a separate import-only migration:

| Current name | Optional descriptive name |
|---|---|
| `scope.py` | `protocol.py` |
| `release.py` | `dp_release.py` |
| `artifacts.py` | `artifact_io.py` |
| `evaluation_sample.py` | `evaluation_log.py` |

Such renaming offers little functional benefit and is not required for an SCI
artifact. No rename is performed in Stage 3B.

## Verification

The focused validation-support test set completed successfully:

```text
40 passed, 0 failed, 1 optional PM4Py performance warning
```

The warning recommends the optional `r4pm` exporter accelerator and does not
affect correctness because PM4Py used its supported fallback.

## Final decision

- Safe whole-module deletions: **none**
- Safe whole-module merges: **none**
- Scientific formulas or parameters changed: **no**
- Python source changed: **no**
- Frozen results changed: **no**
- Full experiment rerun required: **no**

Stage 3B therefore closes with a retain decision. Further cleanup should move
to CLI and experiment-entry naming, where Phase terminology can be removed
without collapsing validated scientific boundaries.

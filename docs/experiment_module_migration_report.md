# Experiment Module Naming Migration

## Scope

Stage 5 replaces three development-oriented filenames with task-oriented
research artifact names:

| Previous file | Release file |
|---|---|
| `src/bul_project/experiments/formal_runner.py` | `src/bul_project/experiments/main_experiment.py` |
| `src/bul_project/experiments/derived_analysis.py` | `src/bul_project/experiments/bul_analysis.py` |
| `src/bul_project/experiments/sensitivity_analysis.py` | `src/bul_project/experiments/sensitivity.py` |
| `tests/test_formal_runner.py` | `tests/test_main_experiment.py` |

`single_run.py`, `batch_run.py`, and `parameter_freeze.py` already describe
their responsibilities clearly and were not renamed.

## Updated references

Imports were updated in:

- `scripts/run_main_experiment.py`
- `scripts/derive_bul_results.py`
- `scripts/run_sensitivity_analysis.py`
- `src/bul_project/experiments/__init__.py`
- `src/bul_project/experiments/sensitivity.py`
- the three corresponding test modules

Self-referential provenance path lists were updated in `bul_analysis.py` and
`sensitivity.py`. No old Python import or old source-path reference remains.

## Preserved compatibility boundary

The following are intentionally unchanged:

- all function and class definitions;
- the 3150-condition experiment matrix;
- Top-k=100 plus OTHER;
- edge threshold tau=10;
- epsilon values, allocation strategies, and seeds;
- Laplace release, sensitivity, temporal reconstruction, established metrics,
  and BUL aggregation;
- output directories such as `results/derived_analysis`;
- historical artifact names such as `phase6_bul_analysis_manifest.json`;
- serialized provenance keys including `original_phase` and
  `phase7c_final_config_sha256`.

Historical serialized identifiers are data contracts, not public Python module
names.

## Verification

Definition-level AST comparison against the frozen experiment version:

| Release module | Frozen source | Common definitions | Mismatches |
|---|---|---:|---:|
| `main_experiment.py` | `formal_runner.py` | 25 | 0 |
| `bul_analysis.py` | `derived_analysis.py` | 18 | 0 |
| `sensitivity.py` | `sensitivity_analysis.py` | 26 | 0 |

Additional verification:

- package modules discovered: 58;
- import failures: 0;
- focused tests: 22 passed, 1 skipped;
- full non-integration suite: 314 passed, 10 skipped, 3 deselected;
- full 3150-condition experiment executed: no;
- existing result files changed: no.

The PM4Py/NumPy messages in the full test run are upstream optional-performance
and pending-deprecation warnings; they do not indicate a numerical failure.

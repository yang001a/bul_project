# CLI and Public Naming Migration Report

## Purpose

This stage converts the development-style command dispatcher into a
package-owned SCI artifact interface. It changes orchestration names and
documentation only. No privacy mechanism, sensitivity, metric, aggregation,
parameter, seed, experiment matrix, or frozen result is changed.

## Entry-point changes

- Root `main.py` is now a thin bootstrap:
  `from bul_project.cli import main`.
- `src/bul_project/cli.py` owns the fixed command registry and dispatch.
- Existing commands remain available:
  `preprocess`, `freeze-parameters`, `run-main`, `derive-bul`,
  `run-sensitivity`, `run-external-validation`, and `analyze`.
- Two requested reproducibility commands were added:
  - `demo`: bundled XES to preprocessing, statistics, DP-full, metrics, and BUL.
  - `reproduce-paper`: cross-platform orchestration of the complete paper
    workflow.
- `scripts/run_full_paper_workflow.ps1` is now a compatibility wrapper around
  `python main.py reproduce-paper`, eliminating a duplicated workflow.

## Public naming changes

Natural public aliases were added without removing historical imports:

| Historical API | Public release API |
|---|---|
| `Phase7CDatasetInput` | `DatasetParameterInput` |
| `Phase7CPolicy` | `ParameterFreezePolicy` |
| `analyze_phase7c_dataset` | `analyze_dataset_parameters` |
| `select_phase7c_parameters` | `select_experiment_parameters` |
| `run_phase7c_smoke_test` | `run_parameter_smoke_test` |
| `finalize_phase7c_parameters` | `finalize_experiment_parameters` |
| `write_phase7c_artifacts` | `write_parameter_artifacts` |
| `Phase6DerivedAnalysisError` | `DerivedAnalysisError` |
| `run_phase6_derived_analysis` | `run_derived_analysis` |
| `Phase6SensitivityError` | `SensitivityAnalysisError` |
| `Phase7AnalysisError` | `StatisticalAnalysisError` |
| `run_phase7_analysis` | `run_statistical_analysis` |
| `Phase7StatisticsError` | `StatisticalTestError` |
| `Phase7PlotError` | `PlottingError` |

Historical names remain valid aliases. Serialized fields and filenames such as
`phase7c_manifest.json`, `phase6_scope_boundary`, and `original_phase` are
unchanged because they are part of frozen provenance and checkpoint
authentication.

## Test-file naming

- `test_phase6_derived_analysis.py` -> `test_derived_analysis.py`
- `test_phase6_sensitivity.py` -> `test_sensitivity_analysis.py`
- `test_phase7_plots.py` -> `test_plotting.py`
- `test_phase7_runner.py` -> `test_statistical_analysis_runner.py`
- `test_phase7_statistics.py` -> `test_statistical_tests.py`

The historical-to-public mapping is documented in
`docs/history/phase_mapping.md`.

## Demo boundary

The bundled three-case demo uses:

- Top-k=100;
- epsilon from the existing configured list;
- uniform sequential budget allocation;
- seed from 0-29;
- the existing parameter-diagnostic implementation to derive `Lmax`, `Tmax`,
  and edge normalization factors.

Only its minimum edge support is 1 because a three-case log cannot satisfy the
main experiment's tau=10. The output explicitly records this as a demo-only
setting. The main experiment remains fixed at Top-k=100 plus OTHER and tau=10.

## Verification

- Public CLI help: passed for all nine commands.
- Quick demo: completed successfully.
- Fixed-seed demo repeat: compared in the test suite excluding runtime only.
- Focused migration tests: 52 passed, 10 skipped.
- Full non-integration suite: 314 passed, 10 skipped, 3 deselected.
- Package import scan: 58 modules discovered, 0 failures.
- Common top-level definitions compared with the frozen version: 128.
- Definition-hash mismatches: 0.
- The optional PM4Py and NumPy warnings are upstream deprecation/performance
  notices, not correctness failures.

## Scientific-equivalence statement

No 3150-condition experiment was rerun. Existing results were neither copied
into the clean release nor modified. All common scientific functions/classes
in the touched experiment and statistics modules remain AST-identical to the
frozen experiment version; only aliases were appended.

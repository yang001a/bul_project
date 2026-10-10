# Reproducibility protocol

## Starting state

The public artifact starts without data-derived configuration artifacts or
experiment outputs. `configs/generated/`, `results/`, and `logs/` are created
only by the commands you run locally.

## Required inputs

Supply licensed BPIC2012, BPIC2017, and Sepsis XES files using the filenames
listed in the root README, or set `BUL_DATASET_ROOT` before running a command.

## Execution order

1. Create the locked environment and install the package.
2. Run the unit tests.
3. Generate the local parameter bundle and Phase 7C smoke calibration with
   `freeze-parameters`.
4. Inspect the main execution plan.
5. Run the 3,150-condition numerical reference matrix.
6. Run BUL derivation, sensitivity analysis, and statistical analysis.
7. Optionally run representative external feasibility validation.

The parameter-freezing step records the public data-dependent quantities used
by the conditional-DP boundary, including clipping domains and nEMD evaluation
scales. It also produces the fixed entropy-weight calibration matrix used by
the later derived BUL analysis. Do not copy a generated configuration from
another data installation without verifying the input-file hashes recorded in
its manifest.

## Outputs

- `configs/generated/`: local parameter diagnostics and execution configuration.
- `results/main_experiment/`: resumable main numerical matrix and checkpoints.
- `results/derived_analysis/`: derived BUL and ablation outputs.
- `results/sensitivity_analysis/`: Top-k/tau sensitivity outputs.
- `results/statistical_analysis/`: statistical tables and paper figures.

No command writes to a different project directory.

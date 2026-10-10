# Behavioral Utility Loss (BUL) under Differentially Private Event-Log Release

This repository implements the Behavioral Utility Loss (BUL) evaluation
framework for differentially private event-log release. BUL combines existing
structure, temporal, and information-loss measures; it does not propose a new
privacy mechanism or a new distance metric.

## Clean public-artifact policy

This repository intentionally contains no benchmark event logs, no
data-derived parameter configuration, no historical experiment result, no
checkpoint, no figure, and no run log. The user generates every local
configuration and output from their own licensed XES inputs.

The repository does not redistribute BPIC2012, BPIC2017, or Sepsis Cases.
Obtain each dataset from its lawful source and comply with its terms before
using it.

## Scientific scope

- Canonical event fields: `case_id`, `activity`, and `timestamp`.
- Case-level Laplace DP with the documented sensitivity and sequential budget
  composition.
- DP-full and DP-edge-only modes.
- Fixed main-reference settings after local parameter freezing: Top-k=100 plus
  OTHER and edge threshold tau=10.
- Existing JS divergence, cosine loss, normalized EMD, entropy deviation, and
  BUL aggregation.
- Conditional-DP boundary: public data-dependent clipping and evaluation
  domains are fixed before the DP release.

## Installation

```powershell
conda env create -f environment.yml
conda activate bul-paper-freeze
python -m pip install -e . --no-deps
python -m pip check
```

## Dataset placement

Place the three files below in `data/raw/`, or set `BUL_DATASET_ROOT` to a
directory containing them:

```text
BPI_Challenge_2012.xes      # BPIC2012
BPI_Challenge_2017.xes      # BPIC2017
Sepsis_Cases.xes            # Sepsis
```

Both `.xes` and `.xes.gz` are supported by the loader. The default filenames
above are used by the public configuration. See [data/raw/README.md](data/raw/README.md).

## Reproducible workflow

For a quick check that does not require the licensed benchmark datasets:

```powershell
python main.py demo
```

The demo runs the bundled XES log through preprocessing, behavioral
statistics, one DP-full release, the established loss metrics, and BUL. Its
support threshold is reduced to 1 solely because the demo contains three
cases; the frozen main-experiment threshold remains tau=10.

First generate a local parameter bundle from the XES logs. This creates local
diagnostics, normalization factors, a parameter smoke-calibration result, and
`configs/generated/` on your machine; none are versioned by Git. The smoke
calibration fixes the entropy-weight reference before the 3,150-condition main
matrix is executed.

```powershell
python main.py freeze-parameters --output-dir configs/generated
```

Then inspect and run the numerical reference matrix:

```powershell
python main.py run-main --plan-only
python main.py run-main --execute-numeric-reference
```

After the main experiment finishes:

```powershell
python main.py derive-bul
python main.py run-sensitivity --execute
python main.py analyze
```

External PM4Py validation is optional and is not a component of BUL:

```powershell
python main.py run-external-validation --execute
```

All generated output stays in `configs/generated/`, `results/`, and `logs/`.

The complete workflow can also be started with one cross-platform command:

```powershell
python main.py reproduce-paper --dataset-root data/raw
```

Use `--dry-run` to inspect the command sequence without generating any result.
Use `--include-external-validation` only when the optional representative
PM4Py feasibility check is required.

Generate or verify deterministic provenance manifests independently:

```powershell
python main.py build-manifest
python main.py build-manifest --verify
```

The manifest records the complete package and workflow code boundary,
environment versions, and the presence or absence of local data,
configuration, results, and compact paper artifacts.

## Tests

```powershell
$env:PYTHONPATH="$PWD\src"
python -m pytest -q -m "not integration"
```

After placing the real XES files, run:

```powershell
$env:RUN_REAL_XES_TESTS="1"
python -m pytest -q -m integration
```

## Citation and license

Add the accepted-paper citation before public release. A license choice is a
maintainer decision; see [LICENSE_NOTICE.md](LICENSE_NOTICE.md) before pushing
this repository to GitHub.

## Repository layout

```text
src/bul_project/     reusable scientific implementation
scripts/             thin command implementations
configs/             generated locally; not committed
data/raw/            user-supplied XES logs; not committed
data/demo/           small non-sensitive demonstration log
results/             generated experiment outputs; not committed
logs/                generated runtime logs; not committed
tests/               unit and integration tests
docs/                reproducibility documentation
```

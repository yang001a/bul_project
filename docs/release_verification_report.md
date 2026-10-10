# Release Verification Report

## Final status

**SCIENTIFIC EQUIVALENCE: PASS**

**SCI ARTIFACT CODE: READY**

The release version is scientifically equivalent to
`BUL_Project_frozen_experiment_version` for the protected algorithms,
interfaces, and fixed deterministic regression boundary. No 3,150-condition
experiment was rerun and no frozen benchmark result was changed.

Before public GitHub or archival publication, the maintainer must still choose
a formal license and replace the placeholder author entry in `CITATION.cff`.
These are publication-metadata tasks, not scientific or numerical blockers.

## Verification summary

| Verification | Result |
|---|---:|
| Protected scientific/environment hash rows | 38 passed, 0 failed |
| Public interface rows | 262 passed, 0 failed |
| Demo numeric fields | 274 passed, 0 failed |
| Demo non-numeric fields | 224 exact, 0 mismatches |
| Maximum absolute error | 0 (`< 1e-12`) |
| Demo canonical snapshots | byte-identical |
| Import discovery | 59 modules imported, 0 failures |
| CLI checks | 5 commands passed |
| Quick Demo | passed |
| Complete pytest suite | 318 passed, 13 skipped, 0 failed |
| Dependency consistency | no broken requirements |

The 13 skips are expected in a clean source release: three real-XES
integration cases require the user's licensed datasets and explicit
`RUN_REAL_XES_TESTS=1`; ten plotting/downstream-statistics checks require
locally regenerated main-result artifacts.

## Hash verification

All 36 Python files under the protected scientific directories are
byte-identical to the frozen version:

```text
preprocessing/
behavior/
privacy/
metrics/
bul/
```

The frozen `environment.yml` and `requirements-lock.txt` are also
byte-identical. The hash evidence is in `verification/hash_check.csv`.

Files outside that boundary changed only for the approved artifact
reorganization: consolidated validation implementations, natural experiment
module names, centralized CLI dispatch, provenance scanning, public
documentation, tests, and release metadata.

## Interface verification

The comparison includes public functions and classes in all protected modules,
plus the renamed experiment and validation modules. Every one of the 262
signatures matches its frozen counterpart. See
`verification/interface_check.csv`.

## Numerical regression and Demo

Both projects processed the same bundled `data/demo/demo_log.xes` with fixed
parameters and seed. The comparison covered:

```text
XES preprocessing
-> trace construction
-> Activity / DFG / Variant / Temporal statistics
-> case-level sensitivity
-> Laplace release
-> JS / Cosine loss / nEMD / Entropy
-> BUL
-> one DP-full experiment condition
```

Wall-clock runtime was the sole excluded field. The complete canonical
snapshots have the same SHA-256:

```text
1e9be334e7626a2c6949abe5c4509bc45f8e65171e6f619329082d2a7e8fd997
```

The CLI Demo also completed independently with three cases, seven retained
events, epsilon 8, seed 0, and the Demo-only edge threshold 1. Its generated
files were moved outside the repository after verification so the release
remains result-free.

## Test interpretation

The complete test suite produced:

```text
318 passed, 13 skipped, 0 failed, 22 warnings
```

The warnings come from an optional PM4Py performance notice and upstream NumPy
matrix pending-deprecation messages. They do not indicate changed numerical
behavior.

## Result protection

- The frozen project was not edited.
- The 3,150-condition matrix was not executed.
- Existing frozen experiment results were not regenerated or overwritten.
- The release `results/` directory contains only `README.md`.
- The release `data/raw/` directory contains only its instructions and
  placeholder file.
- Verification snapshots contain Demo data only.

## Submission conclusion

This code can be used to reproduce the experiments after the three licensed
XES files are installed and `freeze-parameters` generates the local
data-derived configuration. It is suitable as the paper's source artifact.
It must not be described as already containing newly reproduced benchmark
results.

Administrative actions before public deposit:

1. Add the chosen `LICENSE`.
2. Replace the placeholder entity in `CITATION.cff` with the real authors and,
   once available, the paper title, venue, year, and DOI.

# Provenance and Release Materials Report

## Completed work

Stage 6 adds one deterministic provenance implementation:

- `src/bul_project/provenance.py`
- `scripts/build_provenance_manifest.py`
- public command `python main.py build-manifest`

The implementation records:

- root `main.py`;
- all `src/bul_project/**/*.py` package modules;
- all `scripts/**/*.py` workflow modules;
- interpreter and required-package versions;
- release metadata and documentation;
- the presence or absence of user XES files;
- generated configuration files;
- generated result files;
- compact paper artifacts.

Tests, raw data, results, caches, IDE files, and generated outputs are excluded
from the code boundary. They are reported separately where appropriate.

## Main-experiment integration

`experiments/main_experiment.py::build_code_manifest` now delegates to the
central `build_code_hashes` implementation. This corrects the earlier
development-layout scanner that could hash only a subset of the release tree.

This change affects future provenance/plan hashes because they now cover the
complete release implementation. It does not alter any experiment condition,
random seed, privacy release, sensitivity, metric, BUL value, or frozen
historical result.

## Deterministic manifest behavior

The manifests omit wall-clock creation time and absolute workspace paths.
Repeated generation under identical code, documentation, environment, data,
configuration, and results produces identical JSON content.

Default outputs:

- `paper_artifacts/configs/code_manifest.json`
- `paper_artifacts/manifest.json`

Generation:

```text
python main.py build-manifest
```

Read-only verification:

```text
python main.py build-manifest --verify
```

The complete `reproduce-paper` workflow generates the manifest only after
statistical analysis, so local paper results are captured by the final
provenance state.

## Publication materials

Added:

- `CITATION.cff`
- `docs/paper_code_mapping.md`
- provenance instructions in `README.md`
- provenance instructions in `paper_artifacts/README.md`

`CITATION.cff` uses the repository's current project-author placeholder. The
maintainer must replace it with the final paper author list and accepted-paper
citation before journal release.

## License decision

A formal `LICENSE` file is not generated because the maintainer has not chosen
an open-source license. `LICENSE_NOTICE.md` remains an explicit warning.
Choosing MIT, BSD-3-Clause, Apache-2.0, or another license is a manual
publication decision and is not inferred by the refactor.

## Verification

- focused provenance/main-experiment/CLI tests: 20 passed;
- full non-integration suite: 318 passed, 10 skipped, 3 deselected;
- package and workflow code files tracked: 72;
- entry points tracked: 1;
- package modules tracked: 60;
- workflow modules tracked: 11;
- current input XES files in clean release: 0;
- current generated configuration files: 0;
- current generated result files: 0;
- current release state: `clean_source_release`;
- full 3150-condition experiment executed: no.

The 10 skipped tests require intentionally absent frozen/real-data artifacts.
The three deselected tests are marked integration tests. PM4Py/NumPy warnings
are upstream optional-performance or pending-deprecation notices.

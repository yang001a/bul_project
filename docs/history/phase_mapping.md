# Historical Phase Mapping

The development repository used numbered phases while the public SCI artifact
uses task-oriented names. Historical names remain only where changing them
would invalidate frozen manifests, checkpoint schemas, hashes, or existing
downstream notebooks.

| Historical development label | Public artifact name | Current entry point |
|---|---|---|
| Phase 1 | Preprocessing | `python main.py preprocess` |
| Phase 2 | Behavioral statistics | `bul_project.behavior` |
| Phase 3 | Differential privacy release | `bul_project.privacy` |
| Phase 4 | Established loss metrics | `bul_project.metrics` |
| Phase 5 | BUL aggregation | `bul_project.bul` |
| Phase 6 derived analysis | Derived BUL analysis | `python main.py derive-bul` |
| Phase 6 sensitivity | Sensitivity analysis | `python main.py run-sensitivity` |
| Phase 6 external validation | Representative external feasibility validation | `python main.py run-external-validation` |
| Phase 7A | Single-condition experiment | `bul_project.experiments.single_run` |
| Phase 7B | Batch experiment | `bul_project.experiments.batch_run` |
| Phase 7C | Parameter diagnostics and freeze | `python main.py freeze-parameters` |
| Phase 8A | Main numerical experiment | `python main.py run-main` |
| Original Phase 7 statistics | Statistical analysis | `python main.py analyze` |

Compatibility identifiers such as `phase7c_manifest.json`,
`phase6_scope_boundary`, and `original_phase` are serialized provenance
contracts. They are intentionally retained and must not be interpreted as
public workflow names.

## Experiment module migration

| Development module | Release module |
|---|---|
| `experiments/formal_runner.py` | `experiments/main_experiment.py` |
| `experiments/derived_analysis.py` | `experiments/bul_analysis.py` |
| `experiments/sensitivity_analysis.py` | `experiments/sensitivity.py` |

The move changes import paths only. Existing functions, classes, output
filenames, serialized fields, and historical manifest identifiers are retained.

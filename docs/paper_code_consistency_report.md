# Paper–Code Consistency Report

## Verdict

**PASS.** The release artifact implements the frozen BUL methodology and does
not introduce an alternative metric, privacy mechanism, parameter selection
rule, or experiment matrix.

This check concerns the source artifact. It does not claim that the clean
release already contains BPIC2012, BPIC2017, Sepsis, or regenerated paper
results; those files are intentionally excluded and must be produced locally.

## Frozen method checks

| Frozen item | Code evidence | Result |
|---|---|---|
| Top-k=100 | `src/bul_project/config.py` fixes `FROZEN_TOP_K_VARIANTS = 100`. | PASS |
| Top-k + OTHER | `behavior/variants.py::variant_statistics` retains the top variants and sets `other_count = total_cases - sum(top_counts)`. The release then perturbs the top counts and the OTHER count in `privacy/release.py::dp_full_release`. | PASS |
| Coverage is diagnostic only | `experiments/parameter_freeze.py` records `variant_coverage_role = "diagnostic_only"` and keeps the main reference predeclared. Coverage does not select or replace k. | PASS |
| Edge threshold tau=10 | `src/bul_project/config.py` fixes `FROZEN_EDGE_TAU = 10`; candidate values 5/10/20 are limited to sensitivity analysis. | PASS |
| Case-level sensitivity | `privacy/sensitivity.py::calculate_sensitivity` uses Delta activity = Lmax, Delta edge = Lmax-1, Delta variant = 1, Delta ST = (Lmax-1)Tmax, and Delta QT = (Lmax-1)Tmax squared. | PASS |
| Sequential epsilon composition | `privacy/budget.py::allocate_privacy_budget` enforces epsilon A + epsilon E + epsilon V + epsilon T = epsilon total. `split_temporal_budget` assigns epsilon T/2 to ST and epsilon T/2 to QT. | PASS |
| Q_T temporal DP | `privacy/release.py::dp_full_release` constructs sums of squared clipped intervals and releases them with Delta QT and epsilon QT. | PASS |
| Truncated-Gaussian reconstruction | `privacy/temporal_generator.py::TemporalDistributionGenerator.generate` uses only DP edge counts, DP ST, and DP QT; applies the specified mean, variance floor, sample-count rule, seed, and [0,Tmax] bounds. | PASS |
| Conditional-DP claim | README and experiment manifests explicitly use `conditional DP`. Lmax, Tmax, Top-k, edge domain, and nEMD factors are fixed before release; the repository does not claim end-to-end DP. | PASS |
| Z_e normalization factor | `experiments/parameter_freeze.py::_normalization_factors` computes P95-P5 and changes it to 1 only when the spread is exactly zero. The serialized policy marks it as normalization-only. | PASS |
| JS divergence | `metrics/js_metric.py::calculate_js` implements the stated base-2 Jensen–Shannon divergence. | PASS |
| Cosine loss direction | `metrics/cosine_metric.py::calculate_cosine_loss` returns `1 - cosine_similarity` and clips only for numerical range protection. | PASS |
| nEMD | `metrics/emd_metric.py` uses one-dimensional Wasserstein distance divided by the caller-fixed Z_e, with the specified zero-scale rule and [0,1] bound. | PASS |
| Entropy deviation | `metrics/entropy_metric.py::calculate_entropy_loss` retains the frozen reference-relative entropy-loss definition. | PASS |
| BUL aggregation | `bul/bul_score.py::calculate_BUL` combines the three bounded loss dimensions with explicit normalized weights; equal weighting remains the default. | PASS |
| DP-full structure loss | `bul/structure_loss.py` combines Edge JS, DFG Cosine loss, and Variant JS. | PASS |
| DP-edge-only structure loss | `experiments/single_run.py::_edge_only_structure_loss` combines only Edge JS and DFG Cosine loss. Temporal, information, and BUL fields are `None`/NA. | PASS |
| DP-edge-only fairness modes | `privacy/budget.py::edge_only_epsilon` uses epsilon total/4 for component mode and epsilon total for mechanism mode; modes are recorded separately. | PASS |
| Epsilon and seed grid | `src/bul_project/config.py` fixes epsilon to 0.5, 1, 2, 4, 8 and seeds to 0 through 29. | PASS |
| Main matrix | The frozen batch construction remains the 3-dataset, 5-epsilon, 7-method/budget-mode, 30-seed matrix: 3,150 conditions. | PASS |

## Numerical evidence

The fixed Demo regression compares the frozen experiment version with this
release across preprocessing, four behavior-statistics families, sensitivity,
Laplace release, the four established metrics, BUL, and a DP-full
single-condition pipeline.

- Numeric fields: 274
- Non-numeric fields: 224
- Maximum absolute numeric error: 0
- Required tolerance: less than `1e-12`
- Snapshot SHA-256, both versions:
  `1e9be334e7626a2c6949abe5c4509bc45f8e65171e6f619329082d2a7e8fd997`

See `verification/numerical_regression.csv` and
`verification/demo_end_to_end.json`.

## Claim boundary for the paper

The paper may describe this repository as a scientifically equivalent,
reorganized implementation of the frozen experiment. It should describe BUL
as the study's multi-dimensional evaluation framework built from established
metrics, not as an established external theory or a new DP mechanism. PM4Py
remains optional representative feasibility validation and is not part of the
BUL score.

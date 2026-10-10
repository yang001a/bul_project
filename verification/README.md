# Release verification evidence

This directory records the final, non-benchmark verification of the public
artifact against `BUL_Project_frozen_experiment_version`.

It contains no BPIC2012, BPIC2017, or Sepsis event log and no result from the
3,150-condition experiment. The JSON snapshots use only the bundled,
non-sensitive Demo XES log.

Files:

- `hash_check.csv`: byte-level SHA-256 comparison of the protected scientific
  modules and frozen dependency declarations.
- `interface_check.csv`: old/new public function and class signatures,
  including renamed orchestration modules.
- `numerical_regression.csv`: all numeric leaves of the fixed Demo reference
  and release snapshots, checked with an absolute tolerance of `1e-12`.
- `demo_end_to_end.json`: exact Demo comparison summary.
- `reference_demo_snapshot.json` and `release_demo_snapshot.json`: canonical
  inputs to the numerical comparison.
- `import_check.json`: import result for every `bul_project` package module.
- `cli_*.txt`: captured help text for the public commands.
- `pytest_output.txt`: complete test-suite result.
- `pip_check.txt`: dependency consistency result.
- `verification_summary.json`: machine-readable scientific-equivalence
  summary.

Wall-clock runtime is the only excluded old/new field because it is not a
scientific output and necessarily changes between executions.

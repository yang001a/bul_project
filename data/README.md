# Event-log inputs

`data/raw/` is intentionally empty in the public artifact. Add licensed
BPIC2012, BPIC2017, and Sepsis XES files locally, or set `BUL_DATASET_ROOT` to
their directory. These datasets must not be committed or redistributed with
the code.

The loader supports `.xes` and `.xes.gz`. `data/demo/demo_log.xes` is a small,
non-sensitive log for local demonstration and unit testing.

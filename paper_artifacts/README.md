# Paper artifacts

This directory is reserved for compact, traceable artifacts used by the paper.
Large checkpoints, raw event logs, and full runtime metadata are not stored in
the GitHub repository.

The `tables/`, `figures/`, and `csv/` directories remain empty in the clean
source release. Local reproduction may copy only paper-facing compact outputs
into them.

Generate or refresh traceability metadata with:

```text
python main.py build-manifest
```

This writes `configs/code_manifest.json` and `manifest.json`. Verify them
without rewriting:

```text
python main.py build-manifest --verify
```

The manifest explicitly distinguishes a clean source release from a completed
local reproduction. It never treats absent data or absent results as completed
experimentation.

# Environment snapshot

This repository uses the pinned environment specifications in
`environment.yml` and `requirements-lock.txt`.

| Component | Version |
|---|---:|
| Python | 3.13.9 |
| NumPy | 2.3.5 |
| Pandas | 2.3.3 |
| SciPy | 1.16.3 |
| Matplotlib | 3.10.6 |
| PM4Py | 2.7.23.1 |
| lxml | 5.3.0 |

Create the environment with:

```powershell
conda env create -f environment.yml
conda activate bul-paper-freeze
python -m pip install -e . --no-deps
python -m pip check
```

The environment snapshot intentionally records no dataset path, data-derived
configuration, result hash, or completed experiment.

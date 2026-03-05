# Changes

## 2026-03-02
- Added `scripts/experiment_config.py` — user-editable configuration for nerfstudio experiments (datasets, templates, models, paths)
- Added `scripts/run_experiments.py` — CLI runner script that imports config and runs experiments with `--dry-run`, `--config`, and `--filter` options
- Extracted from `notebooks/experiments.ipynb` to support headless/batch experiment runs

# fyp-utils

Utility scripts, notebooks, and environment configs for nerfstudio-based FYP experiments. Handles the workflow around training runs — from environment setup and batch execution to config comparison and output visualization.

## Structure

```
fyp-utils/
├── scripts/          # CLI tools for running and analyzing experiments
├── notebooks/        # Jupyter notebooks for post-processing
└── environments/     # Environment setup (conda, Docker)
    └── nerfstudio/
```

See each directory's README for detailed usage:

- **[scripts/README.md](scripts/README.md)** — experiment runner, config reader/differ, experiment logger, path fixer
- **[notebooks/README.md](notebooks/README.md)** — frames-to-video converter
- **[environments/nerfstudio/README.md](environments/nerfstudio/README.md)** — conda and Docker setup

## Quick Start

```bash
# 1. Set up the environment
./environments/nerfstudio/setup_env.sh --platform cpu

# 2. Activate
conda activate nerfstudio

# 3. Configure experiments
#    Edit scripts/experiment_config.py with your datasets and templates

# 4. Preview experiment commands
python scripts/run_experiments.py --dry-run

# 5. Run experiments
python scripts/run_experiments.py

# 6. Compare configs across runs
python scripts/config_reader.py diff <run-a> <run-b>

# 7. Generate experiment log
python scripts/experiment_log.py /path/to/method-dir -o log.txt
```

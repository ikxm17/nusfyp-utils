"""
Experiment configuration for nerfstudio training runs.

Thin orchestration layer: imports machine-specific settings from local_config.py,
applies defaults, and builds the EXPERIMENTS list as a cartesian product.

The runner script (run_experiments.py) imports EXPERIMENTS and LOG_DIR from here.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Local config — machine-specific settings
# ---------------------------------------------------------------------------
try:
    import local_config
except ImportError:
    sys.exit(
        "ERROR: local_config.py not found.\n"
        "Copy the template and edit it:\n"
        "    cp scripts/experiments/local_config.example.py scripts/experiments/local_config.py"
    )

# ---------------------------------------------------------------------------
# Required settings (must be defined in local_config.py)
# ---------------------------------------------------------------------------
WORKSPACE_DIR = local_config.WORKSPACE_DIR
DATASETS = local_config.DATASETS
EXPERIMENT_TEMPLATES = local_config.EXPERIMENT_TEMPLATES

# ---------------------------------------------------------------------------
# Optional settings (overridable in local_config.py)
# ---------------------------------------------------------------------------
OUTPUT_DIR = getattr(local_config, "OUTPUT_DIR", os.path.join(WORKSPACE_DIR, "outputs"))
LOG_DIR = getattr(local_config, "LOG_DIR", os.path.join(WORKSPACE_DIR, "logs"))
MODELS = getattr(local_config, "MODELS", ["sea-splatfacto"])
NUMBER_OF_REPEATS = getattr(local_config, "NUMBER_OF_REPEATS", 1)

# ---------------------------------------------------------------------------
# Build experiment list (datasets x templates x repeats x models)
# ---------------------------------------------------------------------------
EXPERIMENTS = []
for dataset_name, dataset_path in DATASETS.items():
    for template in EXPERIMENT_TEMPLATES:
        for _ in range(NUMBER_OF_REPEATS):
            for model in MODELS:
                EXPERIMENTS.append({
                    "name": f"{dataset_name}/{template['suffix']}",
                    "model": model,
                    "data": dataset_path,
                    "output_dir": os.path.join(OUTPUT_DIR, dataset_name),
                    "vis": "viewer+tensorboard",
                    "viewer": False,
                    "extra_args": {
                        "experiment-name": f"{dataset_name}-{template['suffix']}",
                        **template["extra_args"],
                    },
                })

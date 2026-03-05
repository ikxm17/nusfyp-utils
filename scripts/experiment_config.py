"""
Experiment configuration for nerfstudio training runs.

Edit this file to define which experiments to run.
The runner script (run_experiments.py) imports EXPERIMENTS and LOG_DIR from here.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE_DIR = "/home/islabella/workspaces/irwin_ws/fyp-playground"

DATASETS = {
    "torpedo_unprocessed": os.path.join(WORKSPACE_DIR, "datasets", "torpedo", "torpedo_unprocessed"),
    "saltpond_unprocessed": os.path.join(WORKSPACE_DIR, "datasets", "saltpond", "saltpond_unprocessed"),
}

OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "outputs")
LOG_DIR = os.path.join(WORKSPACE_DIR, "logs")

# ---------------------------------------------------------------------------
# Experiment templates — each has a suffix and optional extra CLI args
# ---------------------------------------------------------------------------
EXPERIMENT_TEMPLATES = [
    {
        "suffix": "repeat_baseline",
        "extra_args": {},
    },
    # {
    #     "suffix": "no-seathru",
    #     "extra_args": {
    #         "pipeline.model.do-seathru": False,
    #         "pipeline.model.learn-background": False,
    #         "pipeline.model.background-color": "random",
    #         "pipeline.model.output-depth-during-training": False,
    #         "pipeline.model.add-recon-depth-l1": False,
    #         "pipeline.model.use-depth-smooth-loss": False,
    #         "pipeline.model.use-alpha-smooth-loss": False,
    #         "pipeline.model.use-opacity-prior": False,
    #         "pipeline.model.use-dcp-loss": False,
    #         "pipeline.model.use-rgb-sat-loss": False,
    #         "pipeline.model.use-gw-loss": False,
    #         "pipeline.model.use-rgb-sv-loss": False,
    #         "pipeline.model.use-binf-loss": False,
    #         "pipeline.model.use-dsc-attenuation-loss": False,
    #         "pipeline.model.use-depth-l1-loss": False,
    #         "pipeline.model.use-depth-weighted-l1": False,
    #         "pipeline.model.use-depth-weighted-l2": False,
    #     },
    # },
]

# ---------------------------------------------------------------------------
# Models to train
# ---------------------------------------------------------------------------
# MODELS = ["splatfacto"]
MODELS = ["sea-splatfacto"]
# MODELS = ["splatfacto", "sea-splatfacto"]

# ---------------------------------------------------------------------------
# Repeats per (dataset, template, model) combination
# ---------------------------------------------------------------------------
NUMBER_OF_REPEATS = 3

# ---------------------------------------------------------------------------
# Build experiment list (datasets × templates × repeats × models)
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

# ---------------------------------------------------------------------------
# Filters — uncomment to run subsets
# ---------------------------------------------------------------------------
# EXPERIMENTS = [e for e in EXPERIMENTS if "torpedo_unprocessed" in e["name"]]
# EXPERIMENTS = [e for e in EXPERIMENTS if e["name"].endswith("no-seathru")]
# EXPERIMENTS = [e for e in EXPERIMENTS if e["name"] == "red-sea-wreck/no-priors"]
# KEEP = {"baseline", "no-seathru", "backscatter-only"}
# EXPERIMENTS = [e for e in EXPERIMENTS if e["name"].split("/")[1] in KEEP]

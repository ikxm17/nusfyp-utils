"""
Machine-specific settings for Vanda HPC cluster.

Copy to local_config.py and edit:
    cp cluster/local_config.example.py cluster/local_config.py
"""
import os

USERID = os.environ.get("USER", "e0908336")
WORKSPACE_DIR = f"/scratch/{USERID}/fyp-playground"

DATASETS = {
    "torpedo_unprocessed": WORKSPACE_DIR + "/datasets/torpedo/torpedo_unprocessed",
    "saltpond_unprocessed": WORKSPACE_DIR + "/datasets/saltpond/saltpond_unprocessed",
}

EXPERIMENT_TEMPLATES = [
    {"suffix": "baseline", "extra_args": {}},
]

# Headless cluster — tensorboard logging only, no viewer
VIS = "tensorboard"
VIEWER = False

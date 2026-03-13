"""
Machine-specific settings for experiment_config.py.

Copy to local_config.py and edit:
    cp config/local_config.example.py config/local_config.py
"""

WORKSPACE_DIR = "/path/to/fyp-playground"

DATASETS = {
    "torpedo_unprocessed": WORKSPACE_DIR + "/datasets/torpedo/torpedo_unprocessed",
    "saltpond_unprocessed": WORKSPACE_DIR + "/datasets/saltpond/saltpond_unprocessed",
}

EXPERIMENT_TEMPLATES = [
    {
        "suffix": "baseline",
        "extra_args": {},
    },
]

# Optional overrides:
# OUTPUT_DIR = WORKSPACE_DIR + "/outputs"
# LOG_DIR = WORKSPACE_DIR + "/logs"
# MODELS = ["sea-splatfacto"]
# NUMBER_OF_REPEATS = 1
# VIS = "viewer+tensorboard"   # or "tensorboard" for headless
# VIEWER = False

# Render settings:
# RENDER_OUTPUT_NAMES = [
#     "rgb",
#     "depth",
#     "accumulation",
#     "underwater_rgb",
#     "direct",
#     "backscatter",
#     "attenuation_map",
# ]
# RENDER_TYPE = "all"                    # "dataset", "camera-path", or "all"
# RENDER_SPLIT = "train+test+combined"   # "+"-separated; "combined" merges all splits

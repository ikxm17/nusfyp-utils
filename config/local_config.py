"""Idea 009v00: Dataset-Informed Medium Initialization — Saltpond.

Tests whether initializing β_D, β_B, and B_inf from dataset color statistics
produces healthy decomposition when combined with early medium + GW from step 0.

Hypothesis: the degenerate attractor from 008v02 occurs because the frozen medium
was near-identity (β_D≈[1.1,0.95,0.95], B_inf blue-biased). With physically-
informed initialization, Gaussians learn through a realistic medium from step 0,
placing the optimizer in the correct basin.

Submit:
  submit.sh --parallel --paid --render --analyze --batch-prefix 009v00 --dataset saltpond_unprocessed --walltime 02:00:00
"""
import os

USERID = os.environ.get("USER", "e0908336")
WORKSPACE_DIR = f"/scratch/{USERID}/fyp-playground"

DATASETS = {
    "saltpond_unprocessed": WORKSPACE_DIR + "/datasets/saltpond/saltpond_unprocessed",
}

# Shared base config
_BASE = {
    "max-num-iterations": "30000",
    "steps-per-save": "5000",
    "pipeline.model.seathru-from-iter": "10000",
    "pipeline.model.dcp-loss-lambda": "0.10",
    "pipeline.model.color-activation": "linear",
    # Early medium + GW from step 0 (validated in 008v02)
    "pipeline.model.use-early-medium": "True",
    "pipeline.model.early-medium-warmup-steps": "200",
    "pipeline.model.gw-loss-lambda": "0.50",
    "pipeline.model.gw-from-iter": "0",
    # SAT=1.5 (saltpond campaign best, reduces GW×SAT conflict)
    "pipeline.model.sat-loss-lambda": "1.5",
}

# Saltpond dataset-informed initialization (from analysis.md)
# R/G=0.078, B/G=0.849, DCP=0.0001, depth median=4.56m
_SALTPOND_INIT = {
    # β_D: red attenuates ~5x more than green (dampened from physical ratio)
    "pipeline.model.beta-d-init-r": "2.3",
    "pipeline.model.beta-d-init-g": "0.5",
    "pipeline.model.beta-d-init-b": "0.4",
    # β_B: near-zero (DCP≈0, clear water)
    "pipeline.model.beta-b-init-r": "0.01",
    "pipeline.model.beta-b-init-g": "0.05",
    "pipeline.model.beta-b-init-b": "0.04",
    # B_inf: green-dominant water (not blue-biased default)
    "pipeline.model.bg-init-r": "0.05",
    "pipeline.model.bg-init-g": "0.25",
    "pipeline.model.bg-init-b": "0.20",
}

EXPERIMENT_TEMPLATES = [
    {
        # Control: same as 008v02_gw0 but with SAT=1.5 fix
        "suffix": "009v00_ctrl",
        "extra_args": {
            **_BASE,
        },
    },
    {
        # Dataset-informed initialization (idea 009)
        "suffix": "009v00_dataset",
        "extra_args": {
            **_BASE,
            **_SALTPOND_INIT,
        },
    },
    {
        # Dataset init + lower GW (test if better init reduces GW requirement)
        "suffix": "009v00_gw030",
        "extra_args": {
            **_BASE,
            **_SALTPOND_INIT,
            "pipeline.model.gw-loss-lambda": "0.30",
        },
    },
]

VIS = "tensorboard"
VIEWER = False

RENDER_OUTPUT_NAMES = [
    "clean_rgb",
    "depth",
    "accumulation",
    "medium_rgb",
    "backscatter",
    "attenuation_map",
]
RENDER_TYPE = "all"
RENDER_SPLIT = "train+test+combined"

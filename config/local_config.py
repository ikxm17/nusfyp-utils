"""Batch 3 — 009v02: reverse_J refinement + backscatter activation.

Builds on the reverse_J breakthrough (009v01_revJ, gap=+3.13 dB):
1. Longer training (50K) to test if gap tightens further
2. revJ + rgb_sv at very low weight (0.001-0.002) for backscatter activation
3. revJ + higher GW (0.70) to push color correction with the new gradient path

Submit:
  submit.sh --paid --render --filter 009v02 --dataset saltpond_unprocessed --walltime 04:00:00
"""
import os

USERID = os.environ.get("USER", "e0908336")
WORKSPACE_DIR = f"/scratch/{USERID}/fyp-playground"

DATASETS = {
    "saltpond_unprocessed": WORKSPACE_DIR + "/datasets/saltpond/saltpond_unprocessed",
}

# Shared base: early medium + dataset init + GW from step 0 + SAT=1.5
_BASE = {
    "max-num-iterations": "30000",
    "steps-per-save": "5000",
    "pipeline.model.seathru-from-iter": "10000",
    "pipeline.model.dcp-loss-lambda": "0.10",
    "pipeline.model.color-activation": "linear",
    "pipeline.model.use-early-medium": "True",
    "pipeline.model.early-medium-warmup-steps": "200",
    "pipeline.model.gw-loss-lambda": "0.50",
    "pipeline.model.gw-from-iter": "0",
    "pipeline.model.sat-loss-lambda": "1.5",
}

# Dataset-informed init for saltpond
_SALTPOND_INIT = {
    "pipeline.model.beta-d-init-r": "2.3",
    "pipeline.model.beta-d-init-g": "0.5",
    "pipeline.model.beta-d-init-b": "0.4",
    "pipeline.model.beta-b-init-r": "0.01",
    "pipeline.model.beta-b-init-g": "0.05",
    "pipeline.model.beta-b-init-b": "0.04",
    "pipeline.model.bg-init-r": "0.05",
    "pipeline.model.bg-init-g": "0.25",
    "pipeline.model.bg-init-b": "0.20",
}

# revJ base: the breakthrough config from Batch 2
_REVJ_BASE = {
    **_BASE,
    **_SALTPOND_INIT,
    "pipeline.model.gw-reverse-J": "True",
}

# Keep previous batches for config completeness
EXPERIMENT_TEMPLATES = [
    # --- Batch 1 (complete) ---
    {
        "suffix": "009v00_ctrl",
        "extra_args": {**_BASE},
    },
    {
        "suffix": "009v00_dataset",
        "extra_args": {**_BASE, **_SALTPOND_INIT},
    },
    {
        "suffix": "009v00_gw030",
        "extra_args": {**_BASE, **_SALTPOND_INIT, "pipeline.model.gw-loss-lambda": "0.30"},
    },
    # --- Batch 2 (complete) ---
    {
        "suffix": "009v01_revJ",
        "extra_args": {
            **_BASE,
            **_SALTPOND_INIT,
            "pipeline.model.gw-reverse-J": "True",
        },
    },
    {
        "suffix": "009v01_rgbsv",
        "extra_args": {
            **_BASE,
            **_SALTPOND_INIT,
            "pipeline.model.use-rgb-sv-loss": "True",
            "pipeline.model.rgb-sv-lambda": "0.01",
        },
    },
    {
        "suffix": "009v01_atten",
        "extra_args": {
            **_BASE,
            **_SALTPOND_INIT,
            "pipeline.model.gw-reverse-J": "True",
            "pipeline.model.use-rgb-sv-loss": "True",
            "pipeline.model.rgb-sv-lambda": "0.01",
        },
    },
    # --- Batch 3 (009v02): reverse_J refinement ---
    {
        # Extended training: 50K iterations to test gap tightening
        # revJ converged at 30K but rgb_sat still DIVERGING
        "suffix": "009v02_50k",
        "extra_args": {
            **_REVJ_BASE,
            "max-num-iterations": "50000",
            "steps-per-save": "10000",
        },
    },
    {
        # revJ + rgb_sv at very low weight (0.002)
        # Goal: activate backscatter without subsuming GW
        # At 0.01 (Batch 2), GW collapsed to 0.0006 — need 5x lower
        "suffix": "009v02_sv002",
        "extra_args": {
            **_REVJ_BASE,
            "pipeline.model.use-rgb-sv-loss": "True",
            "pipeline.model.rgb-sv-lambda": "0.002",
        },
    },
    {
        # revJ + rgb_sv at even lower weight (0.001)
        # Conservative bracket: if 0.002 still overcorrects, 0.001 is the fallback
        "suffix": "009v02_sv001",
        "extra_args": {
            **_REVJ_BASE,
            "pipeline.model.use-rgb-sv-loss": "True",
            "pipeline.model.rgb-sv-lambda": "0.001",
        },
    },
    {
        # revJ + higher GW (0.70)
        # revJ R/G=0.510 (target >0.7) — more GW pressure with the reverse_J
        # gradient path may push color correction further
        "suffix": "009v02_gw070",
        "extra_args": {
            **_REVJ_BASE,
            "pipeline.model.gw-loss-lambda": "0.70",
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

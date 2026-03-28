"""Batch 2 — Loss Exploration: Unexplored attenuation pressure losses.

Tests three never-properly-tested loss mechanisms that provide direct gradient
pressure on the attenuation model — the key missing piece in decomposition.

NOTE: This batch runs IN PARALLEL with Batch 1 (009v00, already submitted).
Uses --filter to scope the PBS array to only these experiments.

Submit:
  submit.sh --parallel --paid --render --analyze --batch-prefix 009v01 --filter 009v01 --dataset saltpond_unprocessed --walltime 02:00:00
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

# Keep 009v00 experiments so submit --filter 009v01 scopes correctly
EXPERIMENT_TEMPLATES = [
    # --- Batch 1 (already submitted, kept for config completeness) ---
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
    # --- Batch 2: Loss exploration (direct attenuation pressure) ---
    {
        # gw_reverse_J alone (no binf_loss) — NEVER TESTED IN ISOLATION
        # GW on J_restored = direct/attenuation → direct gradient to β_D
        "suffix": "009v01_revJ",
        "extra_args": {
            **_BASE,
            **_SALTPOND_INIT,
            "pipeline.model.gw-reverse-J": "True",
        },
    },
    {
        # rgb_sv_loss — NEVER TESTED AT ALL
        # Constrains std(clean) ≈ std(direct) → attenuation preserves contrast
        "suffix": "009v01_rgbsv",
        "extra_args": {
            **_BASE,
            **_SALTPOND_INIT,
            "pipeline.model.use-rgb-sv-loss": "True",
            "pipeline.model.rgb-sv-lambda": "0.01",
        },
    },
    {
        # Both: reverse_J + rgb_sv — combined attenuation pressure
        "suffix": "009v01_atten",
        "extra_args": {
            **_BASE,
            **_SALTPOND_INIT,
            "pipeline.model.gw-reverse-J": "True",
            "pipeline.model.use-rgb-sv-loss": "True",
            "pipeline.model.rgb-sv-lambda": "0.01",
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

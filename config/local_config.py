"""Batch 00 — 010v00: β_D Minimum Regularization Validation.

Tests whether a softplus floor constraint on β_D prevents the degenerate
attractor (attenuation collapse to identity). The CRITICAL test is 50K
stability: revJ regressed from gap +3.13 to +10.10 at 50K without this fix.

Submit:
  submit.sh --paid --render --analyze --parallel --filter 010v00 --dataset saltpond_unprocessed --walltime 02:00:00
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
_REVJ = {
    **_BASE,
    **_SALTPOND_INIT,
    "pipeline.model.gw-reverse-J": "True",
}

# β_D min reg config
_BETA_D_REG = {
    "pipeline.model.use-beta-d-min-reg": "True",
    "pipeline.model.beta-d-min-reg-lambda": "0.1",
    "pipeline.model.beta-d-min-r": "0.2",
    "pipeline.model.beta-d-min-g": "0.1",
    "pipeline.model.beta-d-min-b": "0.05",
}

EXPERIMENT_TEMPLATES = [
    # --- 010v00: β_D Min Reg Validation ---
    {
        # Control: revJ config on NEW code, toggle OFF
        "suffix": "010v00_control",
        "extra_args": {**_REVJ},
    },
    {
        # Primary: revJ + β_D min reg at 30K
        "suffix": "010v00_reg",
        "extra_args": {**_REVJ, **_BETA_D_REG},
    },
    {
        # CRITICAL: 50K stability test — this is the key experiment
        "suffix": "010v00_50k",
        "extra_args": {
            **_REVJ,
            **_BETA_D_REG,
            "max-num-iterations": "50000",
            "steps-per-save": "10000",
        },
    },
    {
        # Backscatter probe: revJ + reg + rgb_sv at λ=0.005
        "suffix": "010v00_sv005",
        "extra_args": {
            **_REVJ,
            **_BETA_D_REG,
            "pipeline.model.use-rgb-sv-loss": "True",
            "pipeline.model.rgb-sv-lambda": "0.005",
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

"""Dynamics campaign dyn03: GW anneal + midpoint sweep on torpedo.

Test whether GW annealing (high start → low end) can activate the medium model
(backscatter, structured attenuation) during Phase 2 while preventing the
purple/magenta overcorrection artifact that occurs with constant GW=0.50.

Also tests constant GW=0.35 as a midpoint between the dead ends of 0.25 and 0.50.

  submit.sh --paid --render --dataset torpedo_unprocessed --walltime 02:30:00
"""
import os

USERID = os.environ.get("USER", "e0908336")
WORKSPACE_DIR = f"/scratch/{USERID}/fyp-playground"

DATASETS = {
    "saltpond_unprocessed": WORKSPACE_DIR + "/datasets/saltpond/saltpond_unprocessed",
    "torpedo_unprocessed": WORKSPACE_DIR + "/datasets/torpedo/torpedo_unprocessed",
}

# Base config: dyn01_tor_dcp005 (DCP=0.05, SAT=1.5, seathru@10K, 30K iters)
_BASE = {
    "steps-per-save": "5000",
    "max-num-iterations": "30000",
    "pipeline.model.seathru-from-iter": "10000",
    "pipeline.model.gw-from-iter": "10000",
    "pipeline.model.sat-loss-lambda": "1.5",
    "pipeline.model.dcp-loss-lambda": "0.05",
}

EXPERIMENT_TEMPLATES = [
    # --- dyn03_tor: GW anneal + midpoint sweep ---

    # 1. Constant GW=0.35 — untested midpoint between 0.25 (identity) and 0.50 (full + artifact)
    {
        "suffix": "dyn03_tor_gw035",
        "extra_args": {**_BASE, "pipeline.model.gw-loss-lambda": "0.35"},
    },

    # 2. GW anneal 0.50→0.20 — high start activates medium, moderate end preserves some correction
    {
        "suffix": "dyn03_tor_anneal_high",
        "extra_args": {
            **_BASE,
            "pipeline.model.use-gw-anneal": "True",
            "pipeline.model.gw-anneal-start": "0.50",
            "pipeline.model.gw-anneal-end": "0.20",
            "pipeline.model.gw-anneal-steps": "10000",
        },
    },

    # 3. GW anneal 0.50→0.10 — aggressive decay to test if backscatter persists at low final GW
    {
        "suffix": "dyn03_tor_anneal_low",
        "extra_args": {
            **_BASE,
            "pipeline.model.use-gw-anneal": "True",
            "pipeline.model.gw-anneal-start": "0.50",
            "pipeline.model.gw-anneal-end": "0.10",
            "pipeline.model.gw-anneal-steps": "10000",
        },
    },

    # 4. GW anneal 0.30→0.05 (default params) — transfer test from torpedo campaign validation
    {
        "suffix": "dyn03_tor_anneal_ref",
        "extra_args": {
            **_BASE,
            "pipeline.model.use-gw-anneal": "True",
            "pipeline.model.gw-anneal-start": "0.30",
            "pipeline.model.gw-anneal-end": "0.05",
            "pipeline.model.gw-anneal-steps": "10000",
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

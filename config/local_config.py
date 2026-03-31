"""Replication campaign repl07: SP gauss_cap resubmission + RS dwr1+binf_loss.

SP: Resubmit gauss_cap experiments that were lost to config race in repl06.
  - gauss_cap only (reduce 508K Gaussians with standard GW)
  - combined (gauss_cap + gw_reverse_J — test at lower Gaussian count)

RS: Combine the two proven mechanisms:
  - dwr_lambda=1.0 (repl06 showed gap +2.45 dB, β_D_blue in physical range)
  - binf_loss λ=0.1 (repl05 showed reliable B_inf anchoring to ~0.68)
  This addresses B_inf saturation [0.764, 0.910, 0.929] seen in repl06_rs_dwr1.

IMPORTANT: Submit SP first, verify sub-jobs running, then swap to _RS_TEMPLATES.
  submit.sh --paid --render --analyze --batch-prefix repl07 \
    --dataset saltpond_unprocessed --walltime 01:30:00
  [verify running, then swap EXPERIMENT_TEMPLATES]
  submit.sh --paid --render --analyze --batch-prefix repl07 \
    --dataset redsea_unprocessed --walltime 01:30:00
"""
import os

USERID = os.environ.get("USER", "e0908336")
WORKSPACE_DIR = f"/scratch/{USERID}/fyp-playground"

DATASETS = {
    "saltpond_unprocessed": WORKSPACE_DIR + "/datasets/saltpond/saltpond_unprocessed",
    "redsea_unprocessed": WORKSPACE_DIR + "/datasets/redsea/redsea_unprocessed",
}

# ---------------------------------------------------------------------------
# Shared base config
# ---------------------------------------------------------------------------
_COMMON = {
    "steps-per-save": "2000",
    "max-num-iterations": "30000",
    "pipeline.model.sat-loss-lambda": "1.5",
    "pipeline.model.use-gw-anneal": "True",
    "pipeline.model.gw-anneal-start": "0.50",
    "pipeline.model.gw-anneal-steps": "10000",
}

_C = {
    **_COMMON,
    "pipeline.model.seathru-from-iter": "10000",
    "pipeline.model.gw-from-iter": "10000",
    "pipeline.model.medium-warmup-steps": "3000",
    "pipeline.model.stop-split-at": "10000",
}

# ---------------------------------------------------------------------------
# SP base: repl00_sp_freeze (GW constant 0.50, DCP=0.10)
# ---------------------------------------------------------------------------
_SP_FREEZE_BASE = {
    **_C,
    "pipeline.model.gw-anneal-end": "0.50",
    "pipeline.model.dcp-loss-lambda": "0.10",
}

# SP Experiment 1: gauss_cap only (reduce 508K Gaussians, keep standard GW)
_SP_GAUSS_CAP = {
    **_SP_FREEZE_BASE,
    "pipeline.model.densify-grad-thresh": "0.0005",
    "pipeline.model.cull-alpha-thresh": "0.03",
}

# SP Experiment 2: gauss_cap + gw_reverse_J (test reverse_J at lower Gaussian count)
_SP_COMBINED = {
    **_SP_FREEZE_BASE,
    "pipeline.model.gw-reverse-J": "True",
    "pipeline.model.densify-grad-thresh": "0.0005",
    "pipeline.model.cull-alpha-thresh": "0.03",
}

# ---------------------------------------------------------------------------
# RS base: repl03_rs_combined + dwr_lambda=1.0 (proven in repl06)
# ---------------------------------------------------------------------------
_RS_DWR1_BASE = {
    **_C,
    "pipeline.model.gw-anneal-end": "0.15",
    "pipeline.model.dcp-loss-lambda": "0.025",
    "pipeline.model.gw-reverse-J": "True",
    "pipeline.model.densify-grad-thresh": "0.0005",
    "pipeline.model.cull-alpha-thresh": "0.03",
    "pipeline.model.bg-lambda": "0.05",
    "pipeline.model.medium-update-interval": "50",
    "pipeline.model.dwr-lambda": "1.0",
}

# RS Experiment: dwr=1.0 + binf_loss (anchor B_inf, prevent saturation)
_RS_DWR1_BINF01 = {
    **_RS_DWR1_BASE,
    "pipeline.model.use-binf-loss": "True",
    "pipeline.model.binf-loss-lambda": "0.1",
}

# ---------------------------------------------------------------------------
# Templates — start with SP, swap to _RS_TEMPLATES after SP jobs are RUNNING
# ---------------------------------------------------------------------------
_SP_TEMPLATES = [
    {"suffix": "repl07_sp_gauss_cap", "extra_args": _SP_GAUSS_CAP},
    {"suffix": "repl07_sp_combined",  "extra_args": _SP_COMBINED},
]

_RS_TEMPLATES = [
    {"suffix": "repl07_rs_dwr1_binf01", "extra_args": _RS_DWR1_BINF01},
]

EXPERIMENT_TEMPLATES = _SP_TEMPLATES

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

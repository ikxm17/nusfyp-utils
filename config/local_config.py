"""Replication campaign repl05: Joint β_D + B_inf constraint via sigmoid + binf_loss.

Hypothesis: repl04 sigmoid failed because bounding β_D pushed the optimizer to inflate
B_inf to [0.925, 0.991, 0.998] (water balloon effect). If we simultaneously anchor B_inf
via binf_loss (atmospheric light estimate from clean render), the degenerate basin is
eliminated and the optimizer must find the physical solution.

Three experiments:
  1. sig_binf: sigmoid@5.0 + binf_loss lambda=1.0 (full joint constraint)
  2. sig_binf01: sigmoid@5.0 + binf_loss lambda=0.1 (softer B_inf anchor)
  3. binf01: NO sigmoid + binf_loss lambda=0.1 (control: does B_inf anchoring alone
     prevent β_D over-activation by closing the escape route?)

Base: repl03_rs_combined (gw_reverse_J, gauss_cap, medium_update_interval=50, dwr_lambda=2.0).

  submit.sh --paid --render --analyze --batch-prefix repl05 \\
    --dataset redsea_unprocessed --walltime 02:00:00
"""
import os

USERID = os.environ.get("USER", "e0908336")
WORKSPACE_DIR = f"/scratch/{USERID}/fyp-playground"

DATASETS = {
    "saltpond_unprocessed": WORKSPACE_DIR + "/datasets/saltpond/saltpond_unprocessed",
    "curasao_unprocessed": WORKSPACE_DIR + "/datasets/curasao/curasao_unprocessed",
    "japanese-gardens_unprocessed": WORKSPACE_DIR + "/datasets/japanese-gardens/japanese-gardens_unprocessed",
    "panama_unprocessed": WORKSPACE_DIR + "/datasets/panama/panama_unprocessed",
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

# Variant C: extended freeze + early stop-split
_C = {
    **_COMMON,
    "pipeline.model.seathru-from-iter": "10000",
    "pipeline.model.gw-from-iter": "10000",
    "pipeline.model.medium-warmup-steps": "3000",
    "pipeline.model.stop-split-at": "10000",
}

# ---------------------------------------------------------------------------
# RS base: repl03_rs_combined (the breakthrough config)
# freeze + GW 0.50->0.15 + DCP=0.025 + gw_reverse_J + gauss_cap + medium boost
# ---------------------------------------------------------------------------
_RS_COMBINED_BASE = {
    **_C,
    "pipeline.model.gw-anneal-end": "0.15",
    "pipeline.model.dcp-loss-lambda": "0.025",
    "pipeline.model.gw-reverse-J": "True",
    "pipeline.model.densify-grad-thresh": "0.0005",
    "pipeline.model.cull-alpha-thresh": "0.03",
    "pipeline.model.bg-lambda": "0.05",
    "pipeline.model.medium-update-interval": "50",
    "pipeline.model.dwr-lambda": "2.0",
}

# ---------------------------------------------------------------------------
# Experiment 1: sigmoid@5.0 + binf_loss lambda=1.0 (full joint constraint)
# Sigmoid caps β_D at 5.0 (physical ceiling), binf_loss anchors B_inf to
# atmospheric light estimated from clean render. Both escape routes blocked.
# ---------------------------------------------------------------------------
_RS_SIG_BINF = {
    **_RS_COMBINED_BASE,
    "pipeline.model.attenuation-do-sigmoid": "True",
    "pipeline.model.backscatter-do-sigmoid": "True",
    "pipeline.model.use-binf-loss": "True",
    "pipeline.model.binf-loss-lambda": "1.0",
}

# ---------------------------------------------------------------------------
# Experiment 2: sigmoid@5.0 + binf_loss lambda=0.1 (softer B_inf anchor)
# Same sigmoid bound but gentler B_inf constraint. If lambda=1.0 causes
# instability (previous testing showed 24x spike), this may be more stable.
# ---------------------------------------------------------------------------
_RS_SIG_BINF01 = {
    **_RS_COMBINED_BASE,
    "pipeline.model.attenuation-do-sigmoid": "True",
    "pipeline.model.backscatter-do-sigmoid": "True",
    "pipeline.model.use-binf-loss": "True",
    "pipeline.model.binf-loss-lambda": "0.1",
}

# ---------------------------------------------------------------------------
# Experiment 3: binf_loss only, no sigmoid (control experiment)
# Tests whether anchoring B_inf alone prevents β_D over-activation.
# If the optimizer can't inflate B_inf as a pressure valve, does β_D
# self-regulate? Or does it still over-activate to 21.64?
# ---------------------------------------------------------------------------
_RS_BINF01 = {
    **_RS_COMBINED_BASE,
    "pipeline.model.use-binf-loss": "True",
    "pipeline.model.binf-loss-lambda": "0.1",
}

EXPERIMENT_TEMPLATES = [
    {"suffix": "repl05_rs_sig_binf",   "extra_args": _RS_SIG_BINF},
    {"suffix": "repl05_rs_sig_binf01", "extra_args": _RS_SIG_BINF01},
    {"suffix": "repl05_rs_binf01",     "extra_args": _RS_BINF01},
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

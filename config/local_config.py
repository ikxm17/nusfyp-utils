"""Replication campaign repl04: RS beta_D over-activation fix via existing config levers.

Three experiments addressing beta_D over-activation (21.64 vs 5.0 ceiling) using the
combined config (gw_reverse_J + gauss_cap) from repl03 as the base:
  1. sigmoid: attenuation_do_sigmoid + backscatter_do_sigmoid (scale=5.0) caps beta to [0, 5.0]
  2. sigmoid_s3: same but scale=3.0 for tighter physical bounds
  3. ampclamp: use_amplification_clamp=True, max_amplification=3.0 (caps beta_D*z product)

Base: repl03_rs_combined (gw_reverse_J, gauss_cap, medium_update_interval=50, dwr_lambda=2.0).

  submit.sh --paid --render --analyze --batch-prefix repl04 \\
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
# Experiment 1: Sigmoid bounds on beta_D and beta_B (scale=5.0)
# Caps both to [0, 5.0] — the physical upper bound for seawater.
# This is the primary test: does bounding beta_D prevent over-activation
# while preserving the gw_reverse_J decomposition mechanism?
# ---------------------------------------------------------------------------
_RS_SIGMOID = {
    **_RS_COMBINED_BASE,
    "pipeline.model.attenuation-do-sigmoid": "True",
    "pipeline.model.backscatter-do-sigmoid": "True",
    # scale stays at default 5.0 for both
}

# ---------------------------------------------------------------------------
# Experiment 2: Sigmoid with tighter scale=3.0
# If 5.0 is too loose, 3.0 provides tighter physical bounds.
# RedSea dataset analysis: 25m effective depth range (p5-p95).
# At beta_D=3.0 and z_max~34m (p95), max attenuation = exp(-3.0*34) ~ 0
# which is effectively complete absorption. Scale=3.0 should still allow
# full attenuation range while preventing the 21.64 over-activation.
# ---------------------------------------------------------------------------
_RS_SIGMOID_S3 = {
    **_RS_COMBINED_BASE,
    "pipeline.model.attenuation-do-sigmoid": "True",
    "pipeline.model.backscatter-do-sigmoid": "True",
    "pipeline.model.attenuation-scale": "3.0",
    "pipeline.model.backscatter-scale": "3.0",
}

# ---------------------------------------------------------------------------
# Experiment 3: Amplification clamp (caps beta_D * z product)
# Different mechanism: limits the total attenuation effect rather than
# bounding beta_D itself. max_amplification=3.0 means 1/T(z) <= 3,
# so beta_d*z <= log(3) ~ 1.099.
# This preserves the unconstrained beta_D parameterization but limits
# how much clean can diverge from direct signal.
# ---------------------------------------------------------------------------
_RS_AMPCLAMP = {
    **_RS_COMBINED_BASE,
    "pipeline.model.use-amplification-clamp": "True",
    "pipeline.model.max-amplification": "3.0",
}

EXPERIMENT_TEMPLATES = [
    {"suffix": "repl04_rs_sigmoid",    "extra_args": _RS_SIGMOID},
    {"suffix": "repl04_rs_sigmoid_s3", "extra_args": _RS_SIGMOID_S3},
    {"suffix": "repl04_rs_ampclamp",   "extra_args": _RS_AMPCLAMP},
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

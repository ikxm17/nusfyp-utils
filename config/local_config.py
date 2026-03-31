"""Replication campaign repl06: Unexplored config levers for SP and RS.

SP: gw_reverse_J and gauss_cap never tested on Saltpond. These are the
mechanisms that broke the RS deadlock in repl03. SP's core barrier is
GW divergence at 508K Gaussians — gw_reverse_J bypasses the GW-Gaussian
capacity asymmetry, and gauss_cap reduces count toward the <300K activation zone.

RS: dwr_lambda=2.0 and bg_lambda=0.05 were bundled with gauss_cap in repl03
and never ablated. All other scenes show defaults (1.0, 0.01) are optimal.
Depth-weighting at 2.0 may encourage β_D inflation.

Submit SP first, then RS:
  submit.sh --paid --render --analyze --batch-prefix repl06 \
    --dataset saltpond_unprocessed --walltime 01:30:00
  [then swap EXPERIMENT_TEMPLATES to _RS_TEMPLATES and submit]
  submit.sh --paid --render --analyze --batch-prefix repl06 \
    --dataset redsea_unprocessed --walltime 01:30:00
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
# SP base: repl00_sp_freeze (best SP config — constant GW, DCP=0.10)
# ---------------------------------------------------------------------------
_SP_FREEZE_BASE = {
    **_C,
    "pipeline.model.gw-anneal-end": "0.50",
    "pipeline.model.dcp-loss-lambda": "0.10",
}

# SP Experiment 1: gw_reverse_J on SP (never tested — RS breakthrough mechanism)
_SP_REVERSE_J = {
    **_SP_FREEZE_BASE,
    "pipeline.model.gw-reverse-J": "True",
}

# SP Experiment 2: gauss_cap on SP (reduce 508K Gaussians toward <300K zone)
_SP_GAUSS_CAP = {
    **_SP_FREEZE_BASE,
    "pipeline.model.densify-grad-thresh": "0.0005",
    "pipeline.model.cull-alpha-thresh": "0.03",
}

# SP Experiment 3: combined (reverse_J + gauss_cap)
_SP_COMBINED = {
    **_SP_FREEZE_BASE,
    "pipeline.model.gw-reverse-J": "True",
    "pipeline.model.densify-grad-thresh": "0.0005",
    "pipeline.model.cull-alpha-thresh": "0.03",
}

# ---------------------------------------------------------------------------
# RS base: repl03_rs_combined (gw_reverse_J + gauss_cap + medium boost)
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

# RS Experiment 1: dwr_lambda=1.0 (unbundle from gauss_cap — default optimal on 4 scenes)
_RS_DWR1 = {
    **_RS_COMBINED_BASE,
    "pipeline.model.dwr-lambda": "1.0",
}

# RS Experiment 2: bg_lambda=0.01 (unbundle from gauss_cap — default optimal on 3 scenes)
_RS_BG01 = {
    **_RS_COMBINED_BASE,
    "pipeline.model.bg-lambda": "0.01",
}

# RS Experiment 3: both defaults restored (dwr=1.0 + bg=0.01)
_RS_DEFAULTS = {
    **_RS_COMBINED_BASE,
    "pipeline.model.dwr-lambda": "1.0",
    "pipeline.model.bg-lambda": "0.01",
}

# ---------------------------------------------------------------------------
# Templates — start with SP, swap to _RS_TEMPLATES for RS submission
# ---------------------------------------------------------------------------
_SP_TEMPLATES = [
    {"suffix": "repl06_sp_reverse_j", "extra_args": _SP_REVERSE_J},
    {"suffix": "repl06_sp_gauss_cap", "extra_args": _SP_GAUSS_CAP},
    {"suffix": "repl06_sp_combined",  "extra_args": _SP_COMBINED},
]

_RS_TEMPLATES = [
    {"suffix": "repl06_rs_dwr1",     "extra_args": _RS_DWR1},
    {"suffix": "repl06_rs_bg01",     "extra_args": _RS_BG01},
    {"suffix": "repl06_rs_defaults", "extra_args": _RS_DEFAULTS},
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

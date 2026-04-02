"""Data loading layer — wraps read_tb.py functions for figure generation."""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add scripts/ to path for read_tb / eval_experiments imports
_scripts_dir = str(Path(__file__).resolve().parent.parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from read_tb import (
    load_scalars,
    load_phases,
    load_eval_metrics,
    detect_phase_transitions,
    compute_phase_boundaries,
    derive_label,
    find_event_file,
)
from eval_experiments import resolve_runs
from read_config import resolve_outputs_dir


# ---------------------------------------------------------------------------
# Tag name mapping (TB tags → short names)
# ---------------------------------------------------------------------------

# Training metrics: logged as "Train Metrics Dict/<tag>"
METRIC_TAGS = {
    "psnr": "Train Metrics Dict/psnr",
    "gaussian_count": "Train Metrics Dict/gaussian_count",
    # Medium model parameters
    "bg_r": "Train Metrics Dict/bg_r",
    "bg_g": "Train Metrics Dict/bg_g",
    "bg_b": "Train Metrics Dict/bg_b",
    "binf_r": "Train Metrics Dict/binf_r",
    "binf_g": "Train Metrics Dict/binf_g",
    "binf_b": "Train Metrics Dict/binf_b",
    "bs_beta_r": "Train Metrics Dict/bs_beta_r",
    "bs_beta_g": "Train Metrics Dict/bs_beta_g",
    "bs_beta_b": "Train Metrics Dict/bs_beta_b",
    "at_beta_r": "Train Metrics Dict/at_beta_r",
    "at_beta_g": "Train Metrics Dict/at_beta_g",
    "at_beta_b": "Train Metrics Dict/at_beta_b",
    "at_beta_eff_r": "Train Metrics Dict/at_beta_eff_r",
    "at_beta_eff_g": "Train Metrics Dict/at_beta_eff_g",
    "at_beta_eff_b": "Train Metrics Dict/at_beta_eff_b",
    # Gradient diagnostics
    "grad_backscatter": "Train Metrics Dict/grad_backscatter",
    "grad_attenuation": "Train Metrics Dict/grad_attenuation",
    # Clean render statistics
    "clean_mean_r": "Train Metrics Dict/clean_mean_r",
    "clean_mean_g": "Train Metrics Dict/clean_mean_g",
    "clean_mean_b": "Train Metrics Dict/clean_mean_b",
    "gw_weight_eff": "Train Metrics Dict/gw_weight_eff",
    # Decomposition activity (new)
    "medium_contribution": "Train Metrics Dict/medium_contribution",
    "attenuation_magnitude": "Train Metrics Dict/attenuation_magnitude",
    "backscatter_magnitude": "Train Metrics Dict/backscatter_magnitude",
    # Marine snow
    "snow_magnitude": "Train Metrics Dict/snow_magnitude",
}

# Loss tags: logged as "Train Loss Dict/<tag>" or "Train Loss"
LOSS_TAGS = {
    "total_loss": "Train Loss",
    "main_loss": "Train Loss Dict/main_loss",
    "gray_world": "Train Loss Dict/gray_world",
    "dcp": "Train Loss Dict/dcp",
    "rgb_sat": "Train Loss Dict/rgb_sat",
    "rgb_sv": "Train Loss Dict/rgb_sv",
}

# Combined lookup
ALL_TAGS = {**METRIC_TAGS, **LOSS_TAGS}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ExperimentData:
    """All data loaded for a single experiment."""
    run_dir: Path
    label: str
    scalars: Dict[str, list]  # {tb_tag: [(step, value), ...]}
    phases: Dict[str, int]    # {seathru_from_iter, gw_from_iter, max_num_iterations}
    transitions: Dict         # from detect_phase_transitions()
    boundaries: List[tuple]   # from compute_phase_boundaries()
    eval_metrics: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------

def load_experiment(run_dir, outputs_dir=None) -> ExperimentData:
    """Load all data for a single experiment run.

    Args:
        run_dir: path to the timestamp-level run directory
        outputs_dir: base outputs directory (for label derivation)

    Returns:
        ExperimentData with all fields populated
    """
    run_dir = Path(run_dir)

    if find_event_file(run_dir) is None:
        raise FileNotFoundError(
            f"No TensorBoard event file in {run_dir}. "
            f"Sync TB data first, or check the path."
        )

    scalars = load_scalars(run_dir) or {}
    phases = load_phases(run_dir)
    transitions = detect_phase_transitions(scalars, phases)
    boundaries = compute_phase_boundaries(phases, transitions)
    eval_metrics = load_eval_metrics(run_dir)

    if outputs_dir:
        label = derive_label(run_dir, outputs_dir)
    else:
        label = run_dir.name

    return ExperimentData(
        run_dir=run_dir,
        label=label,
        scalars=scalars,
        phases=phases,
        transitions=transitions,
        boundaries=boundaries,
        eval_metrics=eval_metrics,
    )


def load_experiments(specs, outputs_dir) -> List[ExperimentData]:
    """Load multiple experiments from path specs.

    Args:
        specs: list of path specs (passed to resolve_runs)
        outputs_dir: base outputs directory

    Returns:
        list of ExperimentData (skips specs with missing TB data, prints warning)
    """
    experiments = []
    for spec in specs:
        runs = resolve_runs(spec, outputs_dir)
        for run_dir in runs:
            try:
                exp = load_experiment(run_dir, outputs_dir)
                experiments.append(exp)
            except FileNotFoundError as e:
                print(f"  Warning: {e}", file=sys.stderr)
    return experiments


# ---------------------------------------------------------------------------
# Series extraction
# ---------------------------------------------------------------------------

def get_series(experiment: ExperimentData, tag: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract a time series as numpy arrays.

    Args:
        experiment: loaded ExperimentData
        tag: short tag name (e.g., "psnr") or full TB tag

    Returns:
        (steps, values) as numpy arrays, or None if tag not found
    """
    # Resolve short name → full TB tag
    tb_tag = ALL_TAGS.get(tag, tag)

    series = experiment.scalars.get(tb_tag)
    if series is None or len(series) == 0:
        return None

    steps = np.array([s for s, _ in series])
    values = np.array([v for _, v in series])
    return steps, values


def ema_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Exponential moving average smoothing.

    Args:
        values: raw values array
        window: smoothing window (number of samples, not steps)

    Returns:
        smoothed values array (same length)
    """
    if window <= 1 or len(values) <= 1:
        return values
    alpha = 2.0 / (window + 1)
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


# Short dataset prefixes for filenames (keep short for paths)
_DATASET_SHORT = {
    "torpedo_unprocessed": "tor",
    "saltpond_unprocessed": "sp",
    "curasao_unprocessed": "cur",
    "panama_unprocessed": "pan",
    "japanese_gardens_unprocessed": "jg",
    "redsea_unprocessed": "rs",
    "iui3_unprocessed": "iui3",
}

# Full dataset names for figure labels and legends
_DATASET_DISPLAY = {
    "torpedo_unprocessed": "Torpedo",
    "saltpond_unprocessed": "Saltpond",
    "curasao_unprocessed": "Curacao",
    "panama_unprocessed": "Panama",
    "japanese_gardens_unprocessed": "Japanese Gardens",
    "redsea_unprocessed": "Red Sea",
    "iui3_unprocessed": "Red Sea",
}


def get_short_label(experiment: ExperimentData) -> str:
    """Extract a short label suitable for file names.

    Derives from the experiment directory name. Strips dataset prefix
    and replaces with a short identifier (e.g., 'tor_dyn01_dcp005').
    """
    # Label format is typically "experiment_name/timestamp"
    parts = experiment.label.split("/")
    if len(parts) >= 1:
        name = parts[0]
        # Strip dataset prefix, replace with short form
        for full, short in _DATASET_SHORT.items():
            prefix = full + "-"
            if name.startswith(prefix):
                variant = name[len(prefix):]
                return f"{short}_{variant}"
        return name
    return experiment.label


def get_display_label(experiment: ExperimentData) -> str:
    """Extract a human-readable label for figure legends.

    Returns full dataset name (e.g., 'Curacao') when a single experiment
    per dataset is plotted, or 'Dataset (variant)' when disambiguation
    is needed.
    """
    parts = experiment.label.split("/")
    if len(parts) >= 1:
        name = parts[0]
        for full, display in _DATASET_DISPLAY.items():
            prefix = full + "-"
            if name.startswith(prefix):
                return display
        return name
    return experiment.label

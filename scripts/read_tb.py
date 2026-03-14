"""CLI tool for reading and analyzing TensorBoard event files from nerfstudio experiments.

Usage:
    python scripts/read_tb.py summary <paths...> [--outputs-dir <path>] [--window N]
    python scripts/read_tb.py compare <paths...> [--outputs-dir <path>] [--window N]
    python scripts/read_tb.py export <path> [--outputs-dir <path>] [--format csv|json]

Path resolution:
    Same as eval_experiments.py — accepts timestamp dirs, method dirs, or substring specs.

Subcommands:
    summary   Per-experiment training summary: loss components, PSNR trajectory,
              phase transitions, convergence assessment, medium parameters
    compare   Side-by-side comparison table across experiments
    export    Dump raw scalar time-series to CSV or JSON
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from eval_experiments import resolve_runs
from read_config import load_config, resolve_outputs_dir


# ---------------------------------------------------------------------------
# Analysis constants (internal windows, not decision boundaries)
# ---------------------------------------------------------------------------

# Number of steps before phase activation to average for baseline loss
_PRE_BASELINE_STEPS = 100

# Window (in iterations) after activation to search for loss spike
_SPIKE_DETECTION_WINDOW = 2000


# ---------------------------------------------------------------------------
# TensorBoard loading
# ---------------------------------------------------------------------------

def find_event_file(run_dir):
    """Glob for events.out.tfevents.* in a run directory."""
    run_dir = Path(run_dir)
    matches = sorted(run_dir.glob("events.out.tfevents.*"))
    if matches:
        return matches[-1]  # latest event file
    return None


def load_scalars(run_dir):
    """Parse event file via EventAccumulator, return {tag: [(step, value), ...]}.

    Returns None if no event file is found.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_file = find_event_file(run_dir)
    if event_file is None:
        return None

    ea = EventAccumulator(str(run_dir), size_guidance={
        "scalars": 0,        # keep all scalar events
        "images": 1,         # minimize memory
        "histograms": 1,
        "tensors": 1,
    })
    ea.Reload()

    scalars = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        scalars[tag] = [(e.step, e.value) for e in events]

    return scalars


def load_phases(run_dir):
    """Read config.yml for phase transition iteration numbers.

    Returns dict with keys: seathru_from_iter, gw_from_iter, max_num_iterations.
    Returns empty dict if config not found or fields missing.
    """
    config_path = Path(run_dir) / "config.yml"
    if not config_path.is_file():
        return {}

    config = load_config(config_path)
    phases = {}

    try:
        model = config.pipeline.model
        if hasattr(model, "seathru_from_iter"):
            phases["seathru_from_iter"] = model.seathru_from_iter
        if hasattr(model, "gw_from_iter"):
            phases["gw_from_iter"] = model.gw_from_iter
    except AttributeError:
        pass

    try:
        phases["max_num_iterations"] = config.max_num_iterations
    except AttributeError:
        pass

    return phases


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _window_avg(series, window):
    """Average the last `window` steps of a [(step, value), ...] series."""
    if not series:
        return None
    values = [v for _, v in series[-window:]]
    return np.mean(values)


def _final_value(series):
    """Get the last value in a series."""
    if not series:
        return None
    return series[-1][1]


def _peak_value(series):
    """Get the maximum value in a series."""
    if not series:
        return None
    return max(v for _, v in series)


def _step_of_peak(series):
    """Get the step at which the peak value occurs."""
    if not series:
        return None
    return max(series, key=lambda x: x[1])[0]


def detect_phase_transitions(scalars, phases, recovery_factor=1.1):
    """Find SeaThru activation spike and gray world onset.

    Args:
        scalars: {tag: [(step, value), ...]} from TensorBoard.
        phases: dict with seathru_from_iter, gw_from_iter, max_num_iterations.
        recovery_factor: loss must drop to this factor of pre-activation baseline
            to count as recovered (default: 1.1 = within 10% of baseline).

    Returns dict with:
        seathru_spike_magnitude: max loss increase after SeaThru activation
        seathru_spike_step: step of spike
        seathru_recovery_step: step where loss returns to pre-activation level
        gw_spike_magnitude: max loss increase after gray world activation
        gw_spike_step: step of spike
    """
    result = {}
    total_loss = scalars.get("Train Loss")

    if not total_loss or not phases:
        return result

    loss_by_step = {step: val for step, val in total_loss}
    steps = sorted(loss_by_step.keys())

    # SeaThru activation
    seathru_iter = phases.get("seathru_from_iter")
    if seathru_iter is not None and seathru_iter > 0:
        pre_steps = [s for s in steps if s < seathru_iter]
        post_steps = [s for s in steps if s >= seathru_iter]

        if pre_steps and post_steps:
            pre_baseline = np.mean([loss_by_step[s] for s in pre_steps[-_PRE_BASELINE_STEPS:]])

            spike_window = [s for s in post_steps if s < seathru_iter + _SPIKE_DETECTION_WINDOW]
            if spike_window:
                spike_values = [loss_by_step[s] for s in spike_window]
                spike_max = max(spike_values)
                spike_step = spike_window[spike_values.index(spike_max)]
                result["seathru_spike_magnitude"] = spike_max - pre_baseline
                result["seathru_spike_step"] = spike_step

                recovery_threshold = pre_baseline * recovery_factor
                for s in post_steps:
                    if s > spike_step and loss_by_step[s] <= recovery_threshold:
                        result["seathru_recovery_step"] = s
                        break

    # Gray world activation
    gw_iter = phases.get("gw_from_iter")
    if gw_iter is not None and gw_iter > 0:
        pre_steps = [s for s in steps if s < gw_iter]
        post_steps = [s for s in steps if s >= gw_iter]

        if pre_steps and post_steps:
            pre_baseline = np.mean([loss_by_step[s] for s in pre_steps[-_PRE_BASELINE_STEPS:]])
            spike_window = [s for s in post_steps if s < gw_iter + _SPIKE_DETECTION_WINDOW]
            if spike_window:
                spike_values = [loss_by_step[s] for s in spike_window]
                spike_max = max(spike_values)
                spike_step = spike_window[spike_values.index(spike_max)]
                result["gw_spike_magnitude"] = spike_max - pre_baseline
                result["gw_spike_step"] = spike_step

    return result


def _assess_convergence_series(series, window, threshold=0.05):
    """Linear regression on the final window of a [(step, value), ...] series.

    Args:
        series: [(step, value), ...] time series.
        window: number of final entries to fit.
        threshold: relative slope magnitude that separates CONVERGED from
            STILL_IMPROVING (negative) or DIVERGING (positive). Default 0.05
            means a 5% slope relative to the mean value.

    Returns one of: CONVERGED, STILL_IMPROVING, DIVERGING, UNKNOWN
    """
    if not series or len(series) < window:
        return "UNKNOWN"

    final = series[-window:]
    steps = np.array([s for s, _ in final], dtype=float)
    values = np.array([v for _, v in final], dtype=float)

    # Normalize steps to [0, 1] for stable regression
    steps_norm = (steps - steps[0]) / (steps[-1] - steps[0]) if steps[-1] != steps[0] else steps

    # Linear fit
    slope, _ = np.polyfit(steps_norm, values, 1)

    # Relative slope: slope / mean(values)
    mean_val = np.mean(values)
    if mean_val == 0:
        return "UNKNOWN"

    relative_slope = slope / mean_val

    if relative_slope > threshold:
        return "DIVERGING"
    elif relative_slope < -threshold:
        return "STILL_IMPROVING"
    else:
        return "CONVERGED"


def assess_convergence(scalars, window, threshold=0.05):
    """Linear regression on final-window loss to assess convergence.

    Returns one of: CONVERGED, STILL_IMPROVING, DIVERGING
    """
    total_loss = scalars.get("Train Loss")
    return _assess_convergence_series(total_loss, window, threshold=threshold)


def _filter_series(series, start, end):
    """Filter a [(step, value), ...] series to steps in [start, end)."""
    return [(s, v) for s, v in series if start <= s < end]


def _psnr_stats(series):
    """Compute PSNR start, end, peak, peak_step from a [(step, value), ...] series."""
    if not series or len(series) < 2:
        return {}
    values = [v for _, v in series]
    peak_idx = int(np.argmax(values))
    return {
        "start": values[0],
        "end": values[-1],
        "peak": values[peak_idx],
        "peak_step": series[peak_idx][0],
    }


def compute_phase_boundaries(phases, transitions, transition_estimate=3000):
    """Compute phase boundaries from config and detected transitions.

    Args:
        phases: dict with seathru_from_iter, max_num_iterations.
        transitions: dict from detect_phase_transitions().
        transition_estimate: fallback duration (in iterations) for Phase 2 when
            recovery step is not detected. Default 3000 (medium warm-up 1000 +
            GS adaptation 2000).

    Returns list of (name, start_iter, end_iter) tuples.
    Uses seathru_recovery_step (data-driven) as the transition→joint boundary
    when available, otherwise uses transition_estimate.
    """
    seathru_iter = phases.get("seathru_from_iter")
    max_iter = phases.get("max_num_iterations")

    if not seathru_iter or not max_iter:
        return []

    # Phase 2→3 boundary: use recovery step if available, else estimate
    recovery_step = transitions.get("seathru_recovery_step")
    if recovery_step:
        joint_start = recovery_step
    else:
        joint_start = seathru_iter + transition_estimate

    # Don't let joint_start exceed max_iter
    joint_start = min(joint_start, max_iter)

    boundaries = [
        ("phase1_vanilla", 0, seathru_iter),
        ("phase2_transition", seathru_iter, joint_start),
        ("phase3_joint", joint_start, max_iter + 1),
    ]

    return boundaries


def assess_per_phase(scalars, phases, transitions, window,
                     converge_threshold=0.05, transition_estimate=3000):
    """Compute per-phase convergence, PSNR, and loss component statistics.

    Returns dict with keys like:
        per_phase/phase1_vanilla/convergence
        per_phase/phase1_vanilla/psnr_start
        per_phase/phase1_vanilla/psnr_end
        per_phase/phase1_vanilla/psnr_peak
        per_phase/phase1_vanilla/loss_start
        per_phase/phase1_vanilla/loss_end
        per_phase/phase1_vanilla/losses/{component}_start
        per_phase/phase1_vanilla/losses/{component}_end
        per_phase/phase2_transition/spike_ratio
        per_phase/phase2_transition/recovery_steps
        per_phase/phase3_joint/convergence
        ...
    """
    boundaries = compute_phase_boundaries(phases, transitions,
                                          transition_estimate=transition_estimate)
    if not boundaries:
        return {}

    total_loss = scalars.get("Train Loss", [])
    psnr = scalars.get("Train Metrics Dict/psnr", [])

    # Collect all loss component series
    loss_prefix = "Train Loss Dict/"
    loss_components = {}
    for tag, series in scalars.items():
        if tag.startswith(loss_prefix):
            name = tag[len(loss_prefix):]
            loss_components[name] = series

    result = {}

    for phase_name, start, end in boundaries:
        prefix = f"per_phase/{phase_name}"
        phase_loss = _filter_series(total_loss, start, end)
        phase_psnr = _filter_series(psnr, start, end)

        # Convergence (use smaller window for shorter phases)
        phase_window = min(window, max(len(phase_loss) // 3, 10))
        result[f"{prefix}/convergence"] = _assess_convergence_series(
            phase_loss, phase_window, threshold=converge_threshold
        )
        result[f"{prefix}/steps"] = len(phase_loss)

        # Total loss start/end
        if phase_loss:
            result[f"{prefix}/loss_start"] = phase_loss[0][1]
            result[f"{prefix}/loss_end"] = phase_loss[-1][1]

        # PSNR stats
        psnr_stats = _psnr_stats(phase_psnr)
        for k, v in psnr_stats.items():
            result[f"{prefix}/psnr_{k}"] = v

        # Per-component loss breakdown: start, end, and convergence within phase
        for comp_name, comp_series in loss_components.items():
            phase_comp = _filter_series(comp_series, start, end)
            if not phase_comp:
                continue

            comp_prefix = f"{prefix}/losses/{comp_name}"
            result[f"{comp_prefix}_start"] = phase_comp[0][1]
            result[f"{comp_prefix}_end"] = phase_comp[-1][1]

            # For Phase 3, also assess per-component convergence
            if phase_name == "phase3_joint" and len(phase_comp) >= 10:
                comp_window = min(window, max(len(phase_comp) // 3, 10))
                result[f"{comp_prefix}_convergence"] = _assess_convergence_series(
                    phase_comp, comp_window, threshold=converge_threshold
                )

        # Transition-specific: spike ratio and recovery duration
        if phase_name == "phase2_transition" and phase_loss:
            pre_loss = _filter_series(total_loss, max(0, start - 100), start)
            if pre_loss:
                pre_baseline = np.mean([v for _, v in pre_loss])
                spike_max = max(v for _, v in phase_loss)
                result[f"{prefix}/spike_ratio"] = spike_max / pre_baseline if pre_baseline > 0 else None
                result[f"{prefix}/recovery_steps"] = end - start

    return result


def compute_summary(scalars, phases, window, converge_threshold=0.05,
                    recovery_factor=1.1, transition_estimate=3000):
    """Compute per-experiment summary dict from scalars and phases."""
    summary = {}

    # Total loss
    total_loss = scalars.get("Train Loss")
    if total_loss:
        summary["total_loss_final"] = _window_avg(total_loss, window)
        summary["total_steps"] = total_loss[-1][0]

    # Loss components
    loss_prefix = "Train Loss Dict/"
    for tag, series in sorted(scalars.items()):
        if tag.startswith(loss_prefix):
            name = tag[len(loss_prefix):]
            summary[f"loss/{name}"] = _window_avg(series, window)

    # PSNR
    psnr = scalars.get("Train Metrics Dict/psnr")
    if psnr:
        summary["psnr_final"] = _window_avg(psnr, window)
        summary["psnr_peak"] = _peak_value(psnr)
        summary["psnr_peak_step"] = _step_of_peak(psnr)

    # Gaussian count
    gc = scalars.get("Train Metrics Dict/gaussian_count")
    if gc:
        summary["gaussian_count"] = _final_value(gc)

    # Medium parameters (B_inf, bg)
    medium_tags = {
        "bg_r": "Train Metrics Dict/bg_r",
        "bg_g": "Train Metrics Dict/bg_g",
        "bg_b": "Train Metrics Dict/bg_b",
        "binf_r": "Train Metrics Dict/binf_r",
        "binf_g": "Train Metrics Dict/binf_g",
        "binf_b": "Train Metrics Dict/binf_b",
    }
    for name, tag in medium_tags.items():
        series = scalars.get(tag)
        if series:
            summary[f"medium/{name}"] = _final_value(series)

    # Phase transitions
    transitions = detect_phase_transitions(scalars, phases,
                                           recovery_factor=recovery_factor)
    for k, v in transitions.items():
        summary[f"phase/{k}"] = v

    # Convergence (global — kept for backwards compatibility)
    summary["convergence"] = assess_convergence(scalars, window,
                                                threshold=converge_threshold)

    # Per-phase assessment
    per_phase = assess_per_phase(scalars, phases, transitions, window,
                                 converge_threshold=converge_threshold,
                                 transition_estimate=transition_estimate)
    summary.update(per_phase)

    # Phase info from config
    for k, v in phases.items():
        summary[f"config/{k}"] = v

    return summary


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def derive_label(run_dir, outputs_dir):
    """Derive a short label from a run directory path."""
    try:
        rel = Path(run_dir).resolve().relative_to(Path(outputs_dir).resolve())
        parts = rel.parts
        # Expected: dataset / experiment / method / timestamp
        if len(parts) >= 4:
            return f"{parts[1]}/{parts[3]}"
        elif len(parts) >= 2:
            return f"{parts[0]}/{parts[-1]}"
        return str(rel)
    except ValueError:
        return str(run_dir)


def format_summary(summary, label):
    """Human-readable formatted output for a single experiment summary."""
    lines = []
    lines.append(f"=== {label} ===")
    lines.append("")

    # Training overview
    if "total_steps" in summary:
        lines.append(f"  Total steps:       {summary['total_steps']:,}")
    if "total_loss_final" in summary:
        lines.append(f"  Final loss:        {summary['total_loss_final']:.6f}")
    if "convergence" in summary:
        lines.append(f"  Convergence:       {summary['convergence']}")
    lines.append("")

    # PSNR
    if "psnr_final" in summary:
        lines.append("  PSNR:")
        lines.append(f"    Final (window):  {summary['psnr_final']:.2f} dB")
        if "psnr_peak" in summary:
            lines.append(f"    Peak:            {summary['psnr_peak']:.2f} dB (step {summary.get('psnr_peak_step', '?'):,})")
        lines.append("")

    # Loss components
    loss_keys = sorted([k for k in summary if k.startswith("loss/")])
    if loss_keys:
        lines.append("  Loss components (final window avg):")
        max_name = max(len(k.split("/", 1)[1]) for k in loss_keys)
        for k in loss_keys:
            name = k.split("/", 1)[1]
            lines.append(f"    {name:<{max_name}}  {summary[k]:.6f}")
        lines.append("")

    # Medium parameters
    medium_keys = sorted([k for k in summary if k.startswith("medium/")])
    if medium_keys:
        lines.append("  Medium parameters:")
        for k in medium_keys:
            name = k.split("/", 1)[1]
            lines.append(f"    {name}:  {summary[k]:.4f}")
        lines.append("")

    # Phase transitions
    phase_keys = sorted([k for k in summary if k.startswith("phase/")])
    if phase_keys:
        lines.append("  Phase transitions:")
        for k in phase_keys:
            name = k.split("/", 1)[1]
            val = summary[k]
            if isinstance(val, float):
                lines.append(f"    {name}:  {val:.6f}")
            else:
                lines.append(f"    {name}:  {val:,}")
        lines.append("")

    # Config phases
    config_keys = sorted([k for k in summary if k.startswith("config/")])
    if config_keys:
        lines.append("  Config phases:")
        for k in config_keys:
            name = k.split("/", 1)[1]
            lines.append(f"    {name}:  {summary[k]:,}")
        lines.append("")

    # Gaussian count
    if "gaussian_count" in summary:
        lines.append(f"  Gaussian count:    {int(summary['gaussian_count']):,}")
        lines.append("")

    # Per-phase assessment
    phase_names = [
        ("phase1_vanilla", "Phase 1: Vanilla 3DGS"),
        ("phase2_transition", "Phase 2: Transition"),
        ("phase3_joint", "Phase 3: Joint Optimization"),
    ]
    has_per_phase = any(k.startswith("per_phase/") for k in summary)
    if has_per_phase:
        lines.append("  Per-phase assessment:")
        for phase_key, phase_label in phase_names:
            prefix = f"per_phase/{phase_key}"
            conv = summary.get(f"{prefix}/convergence")
            if conv is None:
                continue

            steps = summary.get(f"{prefix}/steps", 0)
            psnr_start = summary.get(f"{prefix}/psnr_start")
            psnr_end = summary.get(f"{prefix}/psnr_end")
            psnr_peak = summary.get(f"{prefix}/psnr_peak")
            loss_start = summary.get(f"{prefix}/loss_start")
            loss_end = summary.get(f"{prefix}/loss_end")

            psnr_str = ""
            if psnr_start is not None and psnr_end is not None:
                psnr_str = f"  PSNR: {psnr_start:.1f} → {psnr_end:.1f}"
                if psnr_peak is not None:
                    psnr_str += f" (peak {psnr_peak:.1f})"

            loss_str = ""
            if loss_start is not None and loss_end is not None:
                loss_str = f"  Loss: {loss_start:.4f} → {loss_end:.4f}"

            lines.append(f"    {phase_label} ({steps:,} steps): {conv}{psnr_str}{loss_str}")

            # Transition-specific
            if phase_key == "phase2_transition":
                spike = summary.get(f"{prefix}/spike_ratio")
                recovery = summary.get(f"{prefix}/recovery_steps")
                if spike is not None:
                    lines.append(f"      Spike: {spike:.1f}x pre-activation loss, recovery: {recovery:,} steps")

            # Phase 3 per-component convergence
            if phase_key == "phase3_joint":
                comp_convs = sorted([
                    (k, summary[k])
                    for k in summary
                    if k.startswith(f"{prefix}/losses/") and k.endswith("_convergence")
                ])
                if comp_convs:
                    still_improving = [
                        k.split("/losses/")[1].replace("_convergence", "")
                        for k, v in comp_convs if v == "STILL_IMPROVING"
                    ]
                    diverging = [
                        k.split("/losses/")[1].replace("_convergence", "")
                        for k, v in comp_convs if v == "DIVERGING"
                    ]
                    if still_improving:
                        lines.append(f"      Still improving: {', '.join(still_improving)}")
                    if diverging:
                        lines.append(f"      Diverging: {', '.join(diverging)}")

        lines.append("")

    return "\n".join(lines)


def format_comparison(summaries, labels):
    """Column-oriented comparison table across experiments."""
    if not summaries:
        return "No summaries to compare."

    # Collect all keys across summaries
    all_keys = []
    seen = set()
    for s in summaries:
        for k in s:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Group keys by category
    categories = [
        ("Training", ["total_steps", "total_loss_final", "convergence"]),
        ("PSNR", ["psnr_final", "psnr_peak", "psnr_peak_step"]),
        ("Gaussian Count", ["gaussian_count"]),
    ]

    # Collect loss, medium, phase, config, per-phase keys
    loss_keys = sorted([k for k in all_keys if k.startswith("loss/")])
    medium_keys = sorted([k for k in all_keys if k.startswith("medium/")])
    phase_keys = sorted([k for k in all_keys if k.startswith("phase/")])
    config_keys = sorted([k for k in all_keys if k.startswith("config/")])

    if loss_keys:
        categories.append(("Loss Components", loss_keys))
    if medium_keys:
        categories.append(("Medium Parameters", medium_keys))
    if phase_keys:
        categories.append(("Phase Transitions", phase_keys))
    if config_keys:
        categories.append(("Config Phases", config_keys))

    # Per-phase summary (curated subset for comparison readability)
    per_phase_compare_keys = []
    for phase_key in ["phase1_vanilla", "phase2_transition", "phase3_joint"]:
        prefix = f"per_phase/{phase_key}"
        candidates = [
            f"{prefix}/convergence",
            f"{prefix}/psnr_start",
            f"{prefix}/psnr_end",
            f"{prefix}/psnr_peak",
            f"{prefix}/loss_start",
            f"{prefix}/loss_end",
        ]
        if phase_key == "phase2_transition":
            candidates.extend([
                f"{prefix}/spike_ratio",
                f"{prefix}/recovery_steps",
            ])
        per_phase_compare_keys.extend(
            k for k in candidates if any(k in s for s in summaries)
        )

    if per_phase_compare_keys:
        categories.append(("Per-Phase Assessment", per_phase_compare_keys))

    # Format values
    def fmt_val(val):
        if val is None:
            return "—"
        if isinstance(val, str):
            return val
        if isinstance(val, float):
            if abs(val) >= 1000:
                return f"{val:,.0f}"
            elif abs(val) >= 1:
                return f"{val:.2f}"
            else:
                return f"{val:.6f}"
        if isinstance(val, int):
            return f"{val:,}"
        return str(val)

    # Compute column widths
    col_width = max(max(len(l) for l in labels), 12)
    metric_width = max(
        max((len(k) for cat_keys in [keys for _, keys in categories] for k in cat_keys), default=20),
        20,
    )

    lines = []

    # Header
    header = f"{'Metric':<{metric_width}}"
    for label in labels:
        header += f"  {label:>{col_width}}"
    lines.append(header)
    lines.append("-" * len(header))

    for cat_name, keys in categories:
        # Check if any summary has any of these keys
        has_data = any(k in s for s in summaries for k in keys)
        if not has_data:
            continue

        lines.append(f"\n  [{cat_name}]")
        for k in keys:
            display_name = k.split("/", 1)[1] if "/" in k else k
            row = f"  {display_name:<{metric_width - 2}}"
            for s in summaries:
                val = s.get(k)
                row += f"  {fmt_val(val):>{col_width}}"
            lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_csv(scalars, output):
    """Write all scalar tags to CSV format."""
    writer = csv.writer(output)
    writer.writerow(["tag", "step", "value"])
    for tag in sorted(scalars.keys()):
        for step, value in scalars[tag]:
            writer.writerow([tag, step, value])


def export_json(scalars, output):
    """Write all scalar tags to JSON format."""
    data = {}
    for tag in sorted(scalars.keys()):
        data[tag] = [{"step": step, "value": value} for step, value in scalars[tag]]
    json.dump(data, output, indent=2)
    output.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _summary_kwargs(args):
    """Extract compute_summary keyword arguments from parsed CLI args."""
    return {
        "converge_threshold": args.converge_threshold,
        "recovery_factor": args.recovery_factor,
        "transition_estimate": args.transition_estimate,
    }


def cmd_summary(args):
    """Handle the summary subcommand."""
    outputs_dir = resolve_outputs_dir(args.outputs_dir)
    kwargs = _summary_kwargs(args)
    all_summaries = []

    for spec in args.paths:
        runs = resolve_runs(spec, outputs_dir)
        for run_dir in runs:
            label = derive_label(run_dir, outputs_dir)
            event_file = find_event_file(run_dir)
            if event_file is None:
                print(f"=== {label} ===")
                print("  No TensorBoard event file found.")
                print()
                continue

            print(f"Loading {label}...", file=sys.stderr)
            scalars = load_scalars(run_dir)
            if scalars is None:
                print(f"=== {label} ===")
                print("  Failed to load scalars.")
                print()
                continue

            phases = load_phases(run_dir)
            summary = compute_summary(scalars, phases, args.window, **kwargs)

            if args.json:
                all_summaries.append({"label": label, "summary": summary})
            else:
                print(format_summary(summary, label))

    if args.json and all_summaries:
        json.dump(all_summaries, sys.stdout, indent=2, default=_json_default)
        sys.stdout.write("\n")


def cmd_compare(args):
    """Handle the compare subcommand."""
    outputs_dir = resolve_outputs_dir(args.outputs_dir)
    kwargs = _summary_kwargs(args)

    summaries = []
    labels = []

    for spec in args.paths:
        runs = resolve_runs(spec, outputs_dir)
        for run_dir in runs:
            label = derive_label(run_dir, outputs_dir)
            event_file = find_event_file(run_dir)
            if event_file is None:
                print(f"Skipping {label}: no TensorBoard event file.", file=sys.stderr)
                continue

            print(f"Loading {label}...", file=sys.stderr)
            scalars = load_scalars(run_dir)
            if scalars is None:
                print(f"Skipping {label}: failed to load scalars.", file=sys.stderr)
                continue

            phases = load_phases(run_dir)
            summary = compute_summary(scalars, phases, args.window, **kwargs)
            summaries.append(summary)
            labels.append(label)

            # Release memory before loading next
            del scalars

    if not summaries:
        print("No experiments with TensorBoard data found.", file=sys.stderr)
        sys.exit(1)

    if args.json:
        data = [{"label": l, "summary": s} for l, s in zip(labels, summaries)]
        json.dump(data, sys.stdout, indent=2, default=_json_default)
        sys.stdout.write("\n")
    else:
        print(format_comparison(summaries, labels))


def cmd_export(args):
    """Handle the export subcommand."""
    outputs_dir = resolve_outputs_dir(args.outputs_dir)

    runs = resolve_runs(args.path, outputs_dir)
    if len(runs) > 1:
        print(f"Warning: multiple runs found, exporting first: {runs[0]}", file=sys.stderr)

    run_dir = runs[0]
    label = derive_label(run_dir, outputs_dir)
    event_file = find_event_file(run_dir)
    if event_file is None:
        print(f"No TensorBoard event file found in {label}.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {label}...", file=sys.stderr)
    scalars = load_scalars(run_dir)
    if scalars is None:
        print(f"Failed to load scalars from {label}.", file=sys.stderr)
        sys.exit(1)

    if args.tags:
        # Filter to requested tags (substring match)
        filtered = {}
        for tag, series in scalars.items():
            if any(t in tag for t in args.tags):
                filtered[tag] = series
        scalars = filtered
        if not scalars:
            print(f"No tags matching {args.tags}.", file=sys.stderr)
            sys.exit(1)

    if args.format == "csv":
        export_csv(scalars, sys.stdout)
    else:
        export_json(scalars, sys.stdout)


def main():
    parser = argparse.ArgumentParser(
        description="Read and analyze TensorBoard event files from nerfstudio experiments.",
    )

    # Shared arguments via parent parser
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--outputs-dir",
        default=None,
        help="Base outputs directory (default: $NERFSTUDIO_OUTPUTS or ./outputs)",
    )
    shared.add_argument(
        "--window",
        type=int,
        default=1000,
        help="Number of final steps to average over (default: 1000)",
    )
    shared.add_argument(
        "--converge-threshold",
        type=float,
        default=0.05,
        dest="converge_threshold",
        help="Relative slope threshold for convergence classification: "
             "> threshold = DIVERGING, < -threshold = STILL_IMPROVING (default: 0.05)",
    )
    shared.add_argument(
        "--recovery-factor",
        type=float,
        default=1.1,
        dest="recovery_factor",
        help="Factor of pre-activation loss baseline that defines recovery "
             "(default: 1.1 = within 10%% of baseline)",
    )
    shared.add_argument(
        "--transition-estimate",
        type=int,
        default=3000,
        dest="transition_estimate",
        help="Fallback Phase 2 duration estimate (iters) when recovery step "
             "is not detected (default: 3000)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # summary subcommand
    summary_parser = subparsers.add_parser(
        "summary", parents=[shared],
        help="Per-experiment training summary",
    )
    summary_parser.add_argument(
        "paths", nargs="+",
        help="Experiment path specs (timestamp dir, method dir, or substring)",
    )
    summary_parser.add_argument(
        "--json", action="store_true",
        help="Output JSON instead of human-readable text",
    )

    # compare subcommand
    compare_parser = subparsers.add_parser(
        "compare", parents=[shared],
        help="Side-by-side comparison table across experiments",
    )
    compare_parser.add_argument(
        "paths", nargs="+",
        help="Experiment path specs to compare",
    )
    compare_parser.add_argument(
        "--json", action="store_true",
        help="Output JSON instead of human-readable table",
    )

    # export subcommand
    export_parser = subparsers.add_parser(
        "export", parents=[shared],
        help="Dump raw scalar time-series to CSV or JSON",
    )
    export_parser.add_argument(
        "path",
        help="Experiment path spec (single experiment)",
    )
    export_parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv)",
    )
    export_parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Filter to tags containing these substrings",
    )

    args = parser.parse_args()

    if args.command == "summary":
        cmd_summary(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "export":
        cmd_export(args)


if __name__ == "__main__":
    main()

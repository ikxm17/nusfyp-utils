"""Phase 2 spike visualization — zoomed view of loss at medium onset.

Shows total loss around seathru_from_iter with pre-activation baseline,
spike peak, and recovery point annotated.
"""

import matplotlib.pyplot as plt
import numpy as np

from paper_figures.style import (
    FIGURE_WIDTH_SINGLE, FIGURE_WIDTH_DOUBLE, FIGURE_HEIGHT_DEFAULT,
    FONT_SIZE_LEGEND,
    apply_legend, save_figure, step_formatter,
)
from paper_figures.data import ExperimentData, get_series, get_short_label


# Zoom window around activation
_PRE_WINDOW = 500    # steps before activation to show
_POST_WINDOW = 5000  # steps after activation to show


def plot(experiment, output_dir, smooth_window=0, formats=("pdf", "png"),
         width="single", no_phase=False, **kwargs):
    """Generate Phase 2 spike visualization.

    Args:
        experiment: ExperimentData instance
        output_dir: directory to save figures
        smooth_window: not used for spike (raw data preferred)
    """
    series = get_series(experiment, "total_loss")
    if series is None:
        print("    Warning: No total loss data found, skipping.")
        return

    seathru_iter = experiment.phases.get("seathru_from_iter")
    if seathru_iter is None or seathru_iter <= 0:
        print("    Warning: No seathru_from_iter in config, skipping spike figure.")
        return

    steps, values = series

    # Zoom to window around activation
    zoom_start = seathru_iter - _PRE_WINDOW
    zoom_end = seathru_iter + _POST_WINDOW
    mask = (steps >= zoom_start) & (steps <= zoom_end)
    z_steps = steps[mask]
    z_values = values[mask]

    if len(z_steps) == 0:
        print("    Warning: No data in spike window, skipping.")
        return

    fig_width = FIGURE_WIDTH_DOUBLE if width == "double" else FIGURE_WIDTH_SINGLE
    fig, ax = plt.subplots(figsize=(fig_width, FIGURE_HEIGHT_DEFAULT))

    ax.plot(z_steps, z_values, linewidth=1.0, color="#333333", zorder=3)

    # Pre-activation baseline
    pre_mask = (steps >= zoom_start) & (steps < seathru_iter)
    if np.any(pre_mask):
        baseline = np.mean(values[pre_mask])
        ax.axhline(baseline, color="#999999", linestyle="--", linewidth=0.8,
                    label=f"Baseline: {baseline:.4f}", zorder=2)

        # Spike peak in post-activation window
        post_mask = (steps >= seathru_iter) & (steps <= zoom_end)
        post_values = values[post_mask]
        post_steps = steps[post_mask]
        if len(post_values) > 0:
            peak_idx = np.argmax(post_values)
            spike_val = post_values[peak_idx]
            spike_step = post_steps[peak_idx]
            spike_ratio = spike_val / baseline if baseline > 0 else 0

            ax.plot(spike_step, spike_val, marker="v", markersize=8,
                    color="#D32F2F", zorder=5)
            ax.annotate(
                f"Spike: {spike_ratio:.1f}x baseline",
                xy=(spike_step, spike_val),
                xytext=(8, 4), textcoords="offset points",
                fontsize=FONT_SIZE_LEGEND, color="#D32F2F",
            )

        # Recovery point
        recovery_step = experiment.transitions.get("seathru_recovery_step")
        if recovery_step and zoom_start <= recovery_step <= zoom_end:
            recovery_idx = np.searchsorted(steps, recovery_step)
            if recovery_idx < len(values):
                ax.axvline(recovery_step, color="#388E3C", linestyle=":",
                           linewidth=0.8, alpha=0.7)
                duration = recovery_step - seathru_iter
                ax.annotate(
                    f"Recovery: {duration:.0f} steps",
                    xy=(recovery_step, baseline),
                    xytext=(8, -12), textcoords="offset points",
                    fontsize=FONT_SIZE_LEGEND, color="#388E3C",
                )

    # Activation line
    ax.axvline(seathru_iter, color="#F57C00", linestyle="-", linewidth=1.0,
               alpha=0.8, label="SeaThru onset", zorder=2)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Total Loss")
    ax.xaxis.set_major_formatter(step_formatter())
    apply_legend(ax, loc="upper right")

    short = get_short_label(experiment)
    ax.set_title(f"Phase 2 Spike — {short}")

    fig.tight_layout()
    save_figure(fig, f"spike_{short}", output_dir, formats)

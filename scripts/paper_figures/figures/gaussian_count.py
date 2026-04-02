"""Gaussian count trajectory — densification saturation.

Shows number of Gaussians over training steps with saturation point annotated.
"""

import matplotlib.pyplot as plt
import numpy as np

from paper_figures.style import (
    FIGURE_WIDTH_SINGLE, FIGURE_WIDTH_DOUBLE, FIGURE_HEIGHT_DEFAULT,
    FONT_SIZE_LEGEND,
    add_phase_boundaries, add_phase_shading, save_figure, step_formatter,
)
from paper_figures.data import ExperimentData, get_series, get_short_label, get_display_label


def _find_saturation_step(steps, values, window=500, threshold=0.01):
    """Find the step where Gaussian count growth plateaus.

    Returns (step, value) or None if no saturation detected.
    """
    if len(values) < window * 2:
        return None

    # Compute growth rate over sliding windows
    for i in range(window, len(values) - window):
        early = np.mean(values[max(0, i - window):i])
        late = np.mean(values[i:i + window])
        if early > 0:
            growth_rate = (late - early) / early
            if growth_rate < threshold:
                return steps[i], values[i]
    return None


def plot(experiment, output_dir, smooth_window=100, formats=("pdf", "png"),
         width="single", no_phase=False, **kwargs):
    """Generate Gaussian count trajectory figure.

    Args:
        experiment: ExperimentData instance
        output_dir: directory to save figures
    """
    series = get_series(experiment, "gaussian_count")
    if series is None:
        print("    Warning: No gaussian_count data found, skipping.")
        return

    steps, values = series
    # Convert to thousands for readability
    values_k = values / 1000.0

    fig_width = FIGURE_WIDTH_DOUBLE if width == "double" else FIGURE_WIDTH_SINGLE
    fig, ax = plt.subplots(figsize=(fig_width, FIGURE_HEIGHT_DEFAULT))

    ax.plot(steps, values_k, linewidth=1.2, color="#666666", zorder=3)

    # Saturation annotation
    sat = _find_saturation_step(steps, values)
    if sat is not None:
        sat_step, sat_val = sat
        ax.axvline(sat_step, color="#999999", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.annotate(
            f"Saturates: {sat_val / 1000:.0f}K @ {sat_step / 1000:.0f}K steps",
            xy=(sat_step, sat_val / 1000),
            xytext=(12, -8), textcoords="offset points",
            fontsize=FONT_SIZE_LEGEND, color="#666666",
        )

    # Add top margin so phase labels clear the data peak
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + (ylim[1] - ylim[0]) * 0.08)

    if not no_phase and experiment.boundaries:
        add_phase_shading(ax, experiment.boundaries)
        add_phase_boundaries(ax, experiment.boundaries)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gaussian Count (thousands)")
    ax.xaxis.set_major_formatter(step_formatter())

    short = get_short_label(experiment)
    display = get_display_label(experiment)
    ax.set_title(f"Gaussian Count — {display}")

    fig.tight_layout()
    save_figure(fig, f"gaussians_{short}", output_dir, formats)

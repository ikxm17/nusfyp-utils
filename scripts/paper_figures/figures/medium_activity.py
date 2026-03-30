"""Medium activity metrics — decomposition health over training.

Shows medium_contribution, attenuation_magnitude, and backscatter_magnitude
as three lines on a single plot. These metrics directly show whether the
medium model is doing useful work or has collapsed to identity.

Only available for experiments with the new metrics (added 2026-03-30).
"""

import matplotlib.pyplot as plt
import numpy as np

from paper_figures.style import (
    FIGURE_WIDTH_SINGLE, FIGURE_WIDTH_DOUBLE, FIGURE_HEIGHT_DEFAULT,
    FONT_SIZE_LEGEND,
    add_phase_boundaries, add_phase_shading, apply_legend, save_figure,
    step_formatter,
)
from paper_figures.data import ExperimentData, get_series, ema_smooth, get_short_label


# Colors for the three activity metrics
ACTIVITY_COLORS = {
    "medium_contribution": "#1976D2",    # blue
    "attenuation_magnitude": "#D32F2F",  # red
    "backscatter_magnitude": "#388E3C",  # green
}

ACTIVITY_LABELS = {
    "medium_contribution": "Medium contribution",
    "attenuation_magnitude": "Attenuation |1-T|",
    "backscatter_magnitude": "Backscatter |B|",
}


def plot(experiment, output_dir, smooth_window=100, formats=("pdf", "png"),
         width="single", no_phase=False, **kwargs):
    """Generate medium activity figure.

    Args:
        experiment: ExperimentData instance
        output_dir: directory to save figures
    """
    fig_width = FIGURE_WIDTH_DOUBLE if width == "double" else FIGURE_WIDTH_SINGLE
    fig, ax = plt.subplots(figsize=(fig_width, FIGURE_HEIGHT_DEFAULT))

    plotted = 0
    for tag in ["medium_contribution", "attenuation_magnitude", "backscatter_magnitude"]:
        series = get_series(experiment, tag)
        if series is None:
            continue
        steps, values = series
        color = ACTIVITY_COLORS[tag]
        label = ACTIVITY_LABELS[tag]

        if smooth_window > 0 and len(values) > smooth_window:
            values = ema_smooth(values, smooth_window)

        ax.plot(steps, values, color=color, label=label, zorder=3)
        plotted += 1

    if plotted == 0:
        print("    Warning: No medium activity metrics found. "
              "These require the 2026-03-30 metrics update.")
        plt.close(fig)
        return

    if not no_phase and experiment.boundaries:
        add_phase_shading(ax, experiment.boundaries)
        add_phase_boundaries(ax, experiment.boundaries)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Magnitude")
    ax.xaxis.set_major_formatter(step_formatter())
    apply_legend(ax, outside=True, ncol=3)

    short = get_short_label(experiment)
    ax.set_title(f"Medium Activity — {short}")

    fig.tight_layout()
    save_figure(fig, f"activity_{short}", output_dir, formats)

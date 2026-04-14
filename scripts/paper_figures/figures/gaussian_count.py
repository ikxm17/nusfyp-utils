"""Gaussian count trajectory over training."""

import matplotlib.pyplot as plt

from paper_figures.style import (
    FIGURE_WIDTH_SINGLE, FIGURE_WIDTH_DOUBLE, FIGURE_HEIGHT_DEFAULT,
    add_phase_boundaries, add_phase_shading, save_figure, step_formatter,
)
from paper_figures.data import ExperimentData, get_series, get_short_label, get_display_label


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
    values_k = values / 1000.0

    fig_width = FIGURE_WIDTH_DOUBLE if width == "double" else FIGURE_WIDTH_SINGLE
    fig, ax = plt.subplots(figsize=(fig_width, FIGURE_HEIGHT_DEFAULT))

    ax.plot(steps, values_k, linewidth=1.2, color="#666666", zorder=3)

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

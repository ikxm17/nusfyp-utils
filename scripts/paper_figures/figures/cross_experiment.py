"""Cross-experiment comparison — overlay the same metric across experiments.

Shows a single metric (PSNR, loss, etc.) for multiple experiments overlaid
on the same axes. This is the "universality" figure proving the
peak-then-decline pattern repeats across configurations.
"""

import matplotlib.pyplot as plt
import numpy as np

from paper_figures.style import (
    FIGURE_WIDTH_SINGLE, FIGURE_WIDTH_DOUBLE, FIGURE_HEIGHT_DEFAULT,
    EXPERIMENT_PALETTE, FONT_SIZE_LEGEND,
    add_phase_boundaries, add_phase_shading, apply_legend, save_figure,
    step_formatter,
)
from paper_figures.data import (
    ExperimentData, get_series, ema_smooth, get_short_label,
)


# Display names for common metrics
METRIC_DISPLAY = {
    "psnr": "PSNR (dB)",
    "total_loss": "Total Loss",
    "gaussian_count": "Gaussian Count",
    "medium_contribution": "Medium Contribution",
    "attenuation_magnitude": "Attenuation Magnitude",
    "backscatter_magnitude": "Backscatter Magnitude",
}


def plot(experiments, output_dir, smooth_window=100, formats=("pdf", "png"),
         width="double", no_phase=False, metric="psnr", **kwargs):
    """Generate cross-experiment comparison figure.

    Args:
        experiments: list of ExperimentData instances
        output_dir: directory to save figures
        smooth_window: EMA window size
        metric: which metric to compare (short tag name)
    """
    if not experiments:
        print("    Warning: No experiments provided.")
        return

    fig_width = FIGURE_WIDTH_DOUBLE if width == "double" else FIGURE_WIDTH_SINGLE
    fig, ax = plt.subplots(figsize=(fig_width, FIGURE_HEIGHT_DEFAULT + 0.5))

    plotted = 0
    for i, exp in enumerate(experiments):
        series = get_series(exp, metric)
        if series is None:
            short = get_short_label(exp)
            print(f"    Warning: No {metric} data for {short}, skipping.")
            continue

        steps, values = series
        color = EXPERIMENT_PALETTE[i % len(EXPERIMENT_PALETTE)]
        label = get_short_label(exp)

        if smooth_window > 0 and len(values) > smooth_window:
            values = ema_smooth(values, smooth_window)

        ax.plot(steps, values, color=color, label=label, zorder=3)
        plotted += 1

    if plotted == 0:
        print(f"    Warning: No experiments had {metric} data.")
        plt.close(fig)
        return

    # Use phase boundaries from first experiment for reference
    if not no_phase and experiments[0].boundaries:
        add_phase_shading(ax, experiments[0].boundaries)
        add_phase_boundaries(ax, experiments[0].boundaries)

    ax.set_xlabel("Training Step")
    ylabel = METRIC_DISPLAY.get(metric, metric)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(step_formatter())

    # Legend below plot if many experiments
    if plotted > 4:
        apply_legend(ax, ncol=min(plotted, 4), outside=True)
    else:
        apply_legend(ax, loc="lower right")

    ax.set_title(f"{ylabel} — Cross-Experiment Comparison")

    fig.tight_layout()
    save_figure(fig, f"compare_{metric}", output_dir, formats)

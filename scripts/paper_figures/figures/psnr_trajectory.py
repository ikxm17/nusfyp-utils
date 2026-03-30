"""PSNR training trajectory with phase boundaries.

Shows training PSNR over iterations, with Phase 1/2/3 boundaries, peak
annotation, and decline from peak. Supports EMA smoothing (raw data shown
as thin semi-transparent line).
"""

import matplotlib.pyplot as plt
import numpy as np

from paper_figures.style import (
    FIGURE_WIDTH_SINGLE, FIGURE_WIDTH_DOUBLE, FIGURE_HEIGHT_DEFAULT,
    LINE_WIDTH_THIN, add_phase_boundaries, add_phase_shading,
    annotate_peak, save_figure, step_formatter,
)
from paper_figures.data import ExperimentData, get_series, ema_smooth, get_short_label


def plot(experiment, output_dir, smooth_window=100, formats=("pdf", "png"),
         width="single", no_phase=False, **kwargs):
    """Generate PSNR trajectory figure for a single experiment.

    Args:
        experiment: ExperimentData instance
        output_dir: directory to save figures
        smooth_window: EMA window size (0 to disable)
        formats: output format tuple
        width: "single" or "double"
        no_phase: suppress phase annotations
    """
    series = get_series(experiment, "psnr")
    if series is None:
        print("    Warning: No PSNR data found, skipping.")
        return

    steps, values = series
    fig_width = FIGURE_WIDTH_DOUBLE if width == "double" else FIGURE_WIDTH_SINGLE
    fig, ax = plt.subplots(figsize=(fig_width, FIGURE_HEIGHT_DEFAULT))

    # Raw data (thin, semi-transparent) + smoothed (main line)
    if smooth_window > 0 and len(values) > smooth_window:
        smoothed = ema_smooth(values, smooth_window)
        ax.plot(steps, values, linewidth=LINE_WIDTH_THIN, alpha=0.25,
                color="#1976D2", zorder=2)
        ax.plot(steps, smoothed, linewidth=1.2, color="#1976D2",
                label="PSNR (smoothed)", zorder=3)
        plot_values = smoothed
    else:
        ax.plot(steps, values, linewidth=1.2, color="#1976D2",
                label="PSNR", zorder=3)
        plot_values = values

    # Phase annotations
    if not no_phase and experiment.boundaries:
        add_phase_shading(ax, experiment.boundaries)
        add_phase_boundaries(ax, experiment.boundaries)

    # Peak annotation
    peak_idx = np.argmax(plot_values)
    peak_step = steps[peak_idx]
    peak_val = plot_values[peak_idx]
    final_val = plot_values[-1]
    decline = peak_val - final_val

    annotate_peak(ax, peak_step, peak_val)

    if decline > 0.5:  # Only annotate decline if meaningful
        ax.annotate(
            f"Δ = −{decline:.1f} dB",
            xy=(steps[-1], final_val),
            xytext=(-8, -12), textcoords="offset points",
            fontsize=7, color="#D32F2F", ha="right",
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("PSNR (dB)")
    ax.xaxis.set_major_formatter(step_formatter())

    short = get_short_label(experiment)
    ax.set_title(f"PSNR Trajectory — {short}")

    fig.tight_layout()
    save_figure(fig, f"psnr_{short}", output_dir, formats)

"""Early stopping analysis — peak window visualization.

Scatter plot of (peak_step, peak_PSNR) across experiments showing the
clustering of optimal checkpoints. Supports the paper's practical contribution
of early stopping as a deployment strategy.
"""

import matplotlib.pyplot as plt
import numpy as np

from paper_figures.style import (
    FIGURE_WIDTH_SINGLE, FIGURE_WIDTH_DOUBLE, FIGURE_HEIGHT_DEFAULT,
    EXPERIMENT_PALETTE, FONT_SIZE_LEGEND,
    save_figure, step_formatter,
)
from paper_figures.data import ExperimentData, get_series, ema_smooth, get_short_label


def plot(experiments, output_dir, smooth_window=100, formats=("pdf", "png"),
         width="single", no_phase=False, **kwargs):
    """Generate early stopping analysis figure.

    Args:
        experiments: list of ExperimentData instances
        output_dir: directory to save figures
    """
    # Collect peak data from each experiment
    peaks = []
    for exp in experiments:
        series = get_series(exp, "psnr")
        if series is None:
            continue
        steps, values = series
        if smooth_window > 0 and len(values) > smooth_window:
            values = ema_smooth(values, smooth_window)

        peak_idx = np.argmax(values)
        peak_step = steps[peak_idx]
        peak_val = values[peak_idx]
        final_val = values[-1]
        decline = peak_val - final_val
        label = get_short_label(exp)
        peaks.append((label, peak_step, peak_val, decline))

    if not peaks:
        print("    Warning: No PSNR data found across experiments.")
        return

    fig_width = FIGURE_WIDTH_DOUBLE if width == "double" else FIGURE_WIDTH_SINGLE
    fig, ax = plt.subplots(figsize=(fig_width, FIGURE_HEIGHT_DEFAULT + 0.3))

    peak_steps = [p[1] for p in peaks]
    peak_vals = [p[2] for p in peaks]
    declines = [p[3] for p in peaks]
    labels = [p[0] for p in peaks]

    # Scatter with size proportional to decline
    sizes = [max(20, d * 10) for d in declines]
    scatter = ax.scatter(peak_steps, peak_vals, s=sizes, c=declines,
                         cmap="RdYlGn_r", edgecolors="black", linewidth=0.5,
                         zorder=5, vmin=0)

    # Colorbar for decline
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Decline from peak (dB)", fontsize=FONT_SIZE_LEGEND)
    cbar.ax.tick_params(labelsize=FONT_SIZE_LEGEND)

    # Label each point
    for label, step, val, _ in peaks:
        ax.annotate(label, xy=(step, val), xytext=(4, 4),
                    textcoords="offset points", fontsize=5, alpha=0.8)

    # Summary statistics
    mean_step = np.mean(peak_steps)
    std_step = np.std(peak_steps)
    mean_decline = np.mean(declines)
    std_decline = np.std(declines)
    ax.text(0.02, 0.02,
            f"Peak: {mean_step / 1000:.0f}K ± {std_step / 1000:.0f}K steps\n"
            f"Decline: {mean_decline:.1f} ± {std_decline:.1f} dB",
            transform=ax.transAxes, fontsize=FONT_SIZE_LEGEND,
            verticalalignment="bottom", bbox=dict(boxstyle="round,pad=0.3",
            facecolor="white", alpha=0.8, edgecolor="#CCCCCC"))

    ax.set_xlabel("Peak Step")
    ax.set_ylabel("Peak PSNR (dB)")
    ax.xaxis.set_major_formatter(step_formatter())
    ax.set_title("Early Stopping — Peak Distribution")

    fig.tight_layout()
    save_figure(fig, "early_stopping", output_dir, formats)

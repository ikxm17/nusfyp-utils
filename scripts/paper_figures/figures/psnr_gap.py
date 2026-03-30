"""PSNR-to-clean gap analysis — decomposition activity proxy.

Bar chart showing PSNR, clean_PSNR, and the gap for each experiment.
The gap is the single best proxy for medium decomposition activity.
Ordered by gap magnitude to show the spectrum from identity to active.
"""

import matplotlib.pyplot as plt
import numpy as np

from paper_figures.style import (
    FIGURE_WIDTH_DOUBLE, FIGURE_HEIGHT_DEFAULT,
    FONT_SIZE_LEGEND, FONT_SIZE_TICK,
    save_figure,
)
from paper_figures.data import ExperimentData, get_short_label


def plot(experiments, output_dir, smooth_window=100, formats=("pdf", "png"),
         width="double", no_phase=False, **kwargs):
    """Generate PSNR gap bar chart across experiments.

    Args:
        experiments: list of ExperimentData instances
        output_dir: directory to save figures
    """
    # Collect metrics from eval results
    entries = []
    for exp in experiments:
        if exp.eval_metrics is None:
            continue
        psnr = exp.eval_metrics.get("psnr")
        clean_psnr = exp.eval_metrics.get("clean_psnr")
        if psnr is None:
            continue
        # If no clean_psnr, gap is 0 (medium inactive or not measured)
        if clean_psnr is None:
            clean_psnr = psnr
        gap = psnr - clean_psnr
        label = get_short_label(exp)
        entries.append((label, psnr, clean_psnr, gap))

    if not entries:
        print("    Warning: No experiments with eval metrics found, skipping.")
        return

    # Sort by gap magnitude (descending)
    entries.sort(key=lambda e: e[3], reverse=True)

    labels = [e[0] for e in entries]
    psnr_vals = [e[1] for e in entries]
    clean_vals = [e[2] for e in entries]
    gap_vals = [e[3] for e in entries]

    n = len(entries)
    x = np.arange(n)
    bar_width = 0.35

    fig_width = max(FIGURE_WIDTH_DOUBLE, n * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, FIGURE_HEIGHT_DEFAULT + 0.5))

    bars1 = ax.bar(x - bar_width / 2, psnr_vals, bar_width,
                   label="PSNR (medium)", color="#1976D2", alpha=0.8)
    bars2 = ax.bar(x + bar_width / 2, clean_vals, bar_width,
                   label="Clean PSNR", color="#66BB6A", alpha=0.8)

    # Annotate gap above each pair
    for i, gap in enumerate(gap_vals):
        y_max = max(psnr_vals[i], clean_vals[i])
        color = "#D32F2F" if gap > 2.0 else "#666666"
        ax.text(i, y_max + 0.3, f"Δ{gap:.1f}",
                ha="center", fontsize=FONT_SIZE_LEGEND, color=color, weight="bold")

    ax.set_xlabel("Experiment")
    ax.set_ylabel("PSNR (dB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT_SIZE_TICK)
    ax.legend(loc="upper right")
    ax.set_title("PSNR-to-Clean Gap (Decomposition Activity)")

    fig.tight_layout()
    save_figure(fig, "psnr_gap_comparison", output_dir, formats)

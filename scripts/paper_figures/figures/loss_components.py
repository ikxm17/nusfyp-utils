"""Per-loss component trajectories.

Shows individual loss curves (main, gray_world, dcp, rgb_sat) over training.
Two modes:
- Absolute: raw loss values (shows magnitudes)
- Budget: each loss as fraction of total (shows competition)
"""

import matplotlib.pyplot as plt
import numpy as np

from paper_figures.style import (
    FIGURE_WIDTH_SINGLE, FIGURE_WIDTH_DOUBLE, FIGURE_HEIGHT_DEFAULT,
    LOSS_COLORS, LOSS_LINESTYLES, FONT_SIZE_LEGEND,
    add_phase_boundaries, add_phase_shading, apply_legend, save_figure,
    step_formatter,
)
from paper_figures.data import (
    ExperimentData, get_series, ema_smooth, get_short_label, get_display_label,
    LOSS_TAGS,
)

# Losses to plot (in display order)
PLOT_LOSSES = ["main_loss", "gray_world", "dcp", "rgb_sat"]
LOSS_DISPLAY = {
    "main_loss": "Main (L1)",
    "gray_world": "Gray World",
    "dcp": "DCP",
    "rgb_sat": "Saturation",
}


def plot(experiment, output_dir, smooth_window=100, formats=("pdf", "png"),
         width="single", no_phase=False, budget=False, losses=None,
         ylim_max=None, **kwargs):
    """Generate loss component figure.

    Args:
        experiment: ExperimentData instance
        output_dir: directory to save figures
        smooth_window: EMA window size
        formats: output format tuple
        width: "single" or "double"
        no_phase: suppress phase annotations
        budget: if True, show losses as fraction of total
        losses: optional list of loss names to plot (overrides module
            default PLOT_LOSSES). Useful for narrative focus.
        ylim_max: optional upper y-axis limit. Budget mode ignores this
            (fixed to 1.05); absolute mode applies it in log scale.
    """
    active_losses = list(losses) if losses else list(PLOT_LOSSES)

    # Load all loss series
    loss_data = {}
    for name in active_losses:
        series = get_series(experiment, name)
        if series is not None:
            loss_data[name] = series

    if not loss_data:
        print("    Warning: No loss data found, skipping.")
        return

    # Load total loss for budget mode
    total_series = get_series(experiment, "total_loss")

    fig_width = FIGURE_WIDTH_DOUBLE if width == "double" else FIGURE_WIDTH_SINGLE
    fig, ax = plt.subplots(figsize=(fig_width, FIGURE_HEIGHT_DEFAULT))

    if budget and total_series is not None:
        _plot_budget(ax, active_losses, loss_data, total_series, smooth_window)
        ax.set_ylabel("Fraction of Total Loss")
        ax.set_ylim(0, 1.05)
    else:
        _plot_absolute(ax, active_losses, loss_data, smooth_window)
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        if ylim_max is not None:
            ax.set_ylim(top=ylim_max)

    if not no_phase and experiment.boundaries:
        add_phase_shading(ax, experiment.boundaries)
        add_phase_boundaries(ax, experiment.boundaries)

    ax.set_xlabel("Training Step")
    ax.xaxis.set_major_formatter(step_formatter())
    apply_legend(ax, outside=True, ncol=min(max(len(loss_data), 1), 4))

    short = get_short_label(experiment)
    display = get_display_label(experiment)
    mode = "budget" if budget else "absolute"
    ax.set_title(f"Loss Components ({mode}) — {display}")

    fig.tight_layout()
    suffix = "_budget" if budget else ""
    save_figure(fig, f"loss{suffix}_{short}", output_dir, formats)


def _plot_absolute(ax, active_losses, loss_data, smooth_window):
    """Plot raw loss values (log scale)."""
    for name in active_losses:
        if name not in loss_data:
            continue
        steps, values = loss_data[name]
        color = LOSS_COLORS.get(name, "#666666")
        linestyle = LOSS_LINESTYLES.get(name, "-")
        label = LOSS_DISPLAY.get(name, name)

        if smooth_window > 0 and len(values) > smooth_window:
            smoothed = ema_smooth(values, smooth_window)
            ax.plot(steps, smoothed, color=color, linestyle=linestyle,
                    label=label, zorder=3)
        else:
            ax.plot(steps, values, color=color, linestyle=linestyle,
                    label=label, zorder=3)


def _plot_budget(ax, active_losses, loss_data, total_series, smooth_window):
    """Plot losses as fraction of total loss."""
    total_steps, total_values = total_series

    # Build interpolator for total loss
    total_by_step = dict(zip(total_steps, total_values))

    for name in active_losses:
        if name not in loss_data:
            continue
        steps, values = loss_data[name]
        color = LOSS_COLORS.get(name, "#666666")
        linestyle = LOSS_LINESTYLES.get(name, "-")
        label = LOSS_DISPLAY.get(name, name)

        # Compute fraction at each step where total is available
        fractions = []
        frac_steps = []
        for s, v in zip(steps, values):
            t = total_by_step.get(s)
            if t is not None and t > 0:
                fractions.append(v / t)
                frac_steps.append(s)

        if not fractions:
            continue

        frac_steps = np.array(frac_steps)
        fractions = np.array(fractions)

        if smooth_window > 0 and len(fractions) > smooth_window:
            fractions = ema_smooth(fractions, smooth_window)

        ax.plot(frac_steps, fractions, color=color, linestyle=linestyle,
                label=label, zorder=3)

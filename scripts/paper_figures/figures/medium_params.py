"""Medium parameter evolution — β_D and B_inf per channel over training.

Generates two separate figures for layout flexibility:
- beta_D: attenuation coefficients (at_beta_r/g/b or at_beta_eff_r/g/b)
- B_inf: backscatter at infinity (binf_r/g/b)

Channels colored red/green/blue. Phase boundaries on both figures.
"""

import matplotlib.pyplot as plt
import numpy as np

from paper_figures.style import (
    FIGURE_WIDTH_SINGLE, FIGURE_WIDTH_DOUBLE, FIGURE_HEIGHT_DEFAULT,
    CHANNEL_COLORS, CHANNEL_LINESTYLES, FONT_SIZE_LEGEND,
    add_phase_boundaries, add_phase_shading, apply_legend,
    save_figure, step_formatter,
)
from paper_figures.data import (
    ExperimentData, get_series, ema_smooth, get_short_label, get_display_label,
)


def plot_beta(experiment, output_dir, smooth_window=100, formats=("pdf", "png"),
              width="single", no_phase=False, **kwargs):
    """Generate standalone attenuation β_D figure.

    Args:
        experiment: ExperimentData instance
        output_dir: directory to save figures
        smooth_window: EMA window size
        formats: output format tuple
        width: "single" or "double"
        no_phase: suppress phase annotations
    """
    fig_width = FIGURE_WIDTH_DOUBLE if width == "double" else FIGURE_WIDTH_SINGLE
    fig, ax = plt.subplots(figsize=(fig_width, FIGURE_HEIGHT_DEFAULT))

    # Try V3 (at_beta_*) first, fall back to V4 (at_beta_eff_*)
    beta_tags = [("at_beta_r", "at_beta_g", "at_beta_b")]
    if get_series(experiment, "at_beta_r") is None:
        beta_tags = [("at_beta_eff_r", "at_beta_eff_g", "at_beta_eff_b")]

    has_beta = False
    for tags in beta_tags:
        for tag, channel, color_key in zip(tags, ["R", "G", "B"], ["r", "g", "b"]):
            series = get_series(experiment, tag)
            if series is None:
                continue
            has_beta = True
            steps, values = series
            color = CHANNEL_COLORS[color_key]
            linestyle = CHANNEL_LINESTYLES[color_key]
            if smooth_window > 0 and len(values) > smooth_window:
                values = ema_smooth(values, smooth_window)
            ax.plot(steps, values, color=color, linestyle=linestyle,
                    label=f"β_D ({channel})", zorder=3)

    if not has_beta:
        print("    Warning: No beta_D data found, skipping.")
        plt.close(fig)
        return

    # Physical plausibility band
    ax.axhspan(0.1, 5.0, alpha=0.08, color="#4CAF50", zorder=0)
    ax.text(0.02, 0.02, "plausible range", transform=ax.transAxes,
            fontsize=FONT_SIZE_LEGEND, color="#4CAF50", ha="left", va="bottom",
            style="italic")

    ax.set_ylabel("β_D (attenuation)")
    ax.set_xlabel("Training Step")
    ax.xaxis.set_major_formatter(step_formatter())
    apply_legend(ax, outside=True, ncol=3)

    short = get_short_label(experiment)
    display = get_display_label(experiment)
    ax.set_title(f"Attenuation β_D — {display}")

    if not no_phase and experiment.boundaries:
        add_phase_shading(ax, experiment.boundaries)
        add_phase_boundaries(ax, experiment.boundaries)

    fig.tight_layout()
    save_figure(fig, f"medium_beta_{short}", output_dir, formats)


def plot_binf(experiment, output_dir, smooth_window=100, formats=("pdf", "png"),
              width="single", no_phase=False, **kwargs):
    """Generate standalone backscatter B_inf figure.

    Args:
        experiment: ExperimentData instance
        output_dir: directory to save figures
        smooth_window: EMA window size
        formats: output format tuple
        width: "single" or "double"
        no_phase: suppress phase annotations
    """
    fig_width = FIGURE_WIDTH_DOUBLE if width == "double" else FIGURE_WIDTH_SINGLE
    fig, ax = plt.subplots(figsize=(fig_width, FIGURE_HEIGHT_DEFAULT))

    has_binf = False
    for tag, channel, color_key in [
        ("binf_r", "R", "r"), ("binf_g", "G", "g"), ("binf_b", "B", "b")
    ]:
        series = get_series(experiment, tag)
        if series is None:
            continue
        has_binf = True
        steps, values = series
        color = CHANNEL_COLORS[color_key]
        linestyle = CHANNEL_LINESTYLES[color_key]
        if smooth_window > 0 and len(values) > smooth_window:
            values = ema_smooth(values, smooth_window)
        ax.plot(steps, values, color=color, linestyle=linestyle,
                label=f"B_∞ ({channel})", zorder=3)

    if not has_binf:
        print("    Warning: No B_inf data found, skipping.")
        plt.close(fig)
        return

    ax.set_ylabel("B_∞ (backscatter)")
    ax.set_xlabel("Training Step")
    ax.xaxis.set_major_formatter(step_formatter())
    apply_legend(ax, outside=True, ncol=3)

    short = get_short_label(experiment)
    display = get_display_label(experiment)
    ax.set_title(f"Backscatter B_∞ — {display}")

    if not no_phase and experiment.boundaries:
        add_phase_shading(ax, experiment.boundaries)
        add_phase_boundaries(ax, experiment.boundaries)

    fig.tight_layout()
    save_figure(fig, f"medium_binf_{short}", output_dir, formats)


def plot(experiment, output_dir, smooth_window=100, formats=("pdf", "png"),
         width="single", no_phase=False, **kwargs):
    """Generate medium parameter figures (separate β_D and B_inf).

    Args:
        experiment: ExperimentData instance
        output_dir: directory to save figures
        smooth_window: EMA window size
        formats: output format tuple
        width: "single" or "double"
        no_phase: suppress phase annotations
    """
    plot_beta(experiment, output_dir, smooth_window=smooth_window,
              formats=formats, width=width, no_phase=no_phase, **kwargs)
    plot_binf(experiment, output_dir, smooth_window=smooth_window,
              formats=formats, width=width, no_phase=no_phase, **kwargs)

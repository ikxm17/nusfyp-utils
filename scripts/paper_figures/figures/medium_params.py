"""Medium parameter evolution — β_D and B_inf per channel over training.

Two vertically stacked subplots:
- Top: attenuation coefficients (at_beta_r/g/b or at_beta_eff_r/g/b)
- Bottom: backscatter at infinity (binf_r/g/b)

Channels colored red/green/blue. Phase boundaries on both subplots.
"""

import matplotlib.pyplot as plt
import numpy as np

from paper_figures.style import (
    FIGURE_WIDTH_SINGLE, FIGURE_WIDTH_DOUBLE, CHANNEL_COLORS,
    FONT_SIZE_LEGEND,
    add_phase_boundaries, add_phase_shading, save_figure, step_formatter,
)
from paper_figures.data import ExperimentData, get_series, ema_smooth, get_short_label


def plot(experiment, output_dir, smooth_window=100, formats=("pdf", "png"),
         width="single", no_phase=False, **kwargs):
    """Generate medium parameter evolution figure (2-panel).

    Args:
        experiment: ExperimentData instance
        output_dir: directory to save figures
        smooth_window: EMA window size
        formats: output format tuple
        width: "single" or "double"
        no_phase: suppress phase annotations
    """
    fig_width = FIGURE_WIDTH_DOUBLE if width == "double" else FIGURE_WIDTH_SINGLE
    fig, (ax_beta, ax_binf) = plt.subplots(
        2, 1, figsize=(fig_width, 4.5), sharex=True,
    )

    # --- Top: attenuation beta_D ---
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
            if smooth_window > 0 and len(values) > smooth_window:
                values = ema_smooth(values, smooth_window)
            ax_beta.plot(steps, values, color=color, label=f"β_D ({channel})", zorder=3)

    if has_beta:
        # Physical plausibility band
        ax_beta.axhspan(0.1, 5.0, alpha=0.08, color="#4CAF50", zorder=0)
        ax_beta.text(0.98, 0.95, "plausible range", transform=ax_beta.transAxes,
                     fontsize=FONT_SIZE_LEGEND, color="#4CAF50", ha="right", va="top",
                     style="italic")

    ax_beta.set_ylabel("β_D (attenuation)")
    ax_beta.legend(loc="upper left", ncol=3)

    short = get_short_label(experiment)
    ax_beta.set_title(f"Medium Parameters — {short}")

    # --- Bottom: B_inf ---
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
        if smooth_window > 0 and len(values) > smooth_window:
            values = ema_smooth(values, smooth_window)
        ax_binf.plot(steps, values, color=color, label=f"B_inf ({channel})", zorder=3)

    ax_binf.set_ylabel("B_inf (backscatter ∞)")
    ax_binf.set_xlabel("Training Step")
    ax_binf.xaxis.set_major_formatter(step_formatter())
    ax_binf.legend(loc="upper left", ncol=3)

    if not has_beta and not has_binf:
        print("    Warning: No medium parameter data found, skipping.")
        plt.close(fig)
        return

    # Phase annotations on both
    if not no_phase and experiment.boundaries:
        for ax in (ax_beta, ax_binf):
            add_phase_shading(ax, experiment.boundaries)
            add_phase_boundaries(ax, experiment.boundaries, label=(ax is ax_beta))

    fig.tight_layout()
    save_figure(fig, f"medium_params_{short}", output_dir, formats)

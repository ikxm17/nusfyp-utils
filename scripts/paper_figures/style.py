"""Publication styling constants and helpers for paper figures."""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory
from pathlib import Path

# ---------------------------------------------------------------------------
# Figure dimensions (IEEE / ACM single/double column)
# ---------------------------------------------------------------------------

FIGURE_WIDTH_SINGLE = 3.5   # inches, single-column
FIGURE_WIDTH_DOUBLE = 7.0   # inches, double-column
FIGURE_HEIGHT_DEFAULT = 2.8  # inches
DPI = 300

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

FONT_FAMILY = "serif"
FONT_SIZE_TITLE = 10
FONT_SIZE_LABEL = 9
FONT_SIZE_TICK = 8
FONT_SIZE_LEGEND = 7
LINE_WIDTH = 1.2
LINE_WIDTH_THIN = 0.6  # for raw/secondary data

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

# Phase background shading
PHASE_COLORS = {
    "phase1_vanilla": "#E8E8E8",
    "phase2_transition": "#FFE0B2",
    "phase3_joint": "#E3F2FD",
}

PHASE_LABELS = {
    "phase1_vanilla": "Phase 1",
    "phase2_transition": "Phase 2",
    "phase3_joint": "Phase 3",
}

# RGB channel colors and line styles (always distinguish by line style too)
CHANNEL_COLORS = {"r": "#D32F2F", "g": "#388E3C", "b": "#1976D2"}
CHANNEL_LINESTYLES = {"r": "-", "g": "--", "b": ":"}

# Loss component colors and line styles
LOSS_COLORS = {
    "main_loss": "#333333",
    "gray_world": "#7B1FA2",
    "dcp": "#F57C00",
    "rgb_sat": "#0097A7",
    "rgb_sv": "#689F38",
}
LOSS_LINESTYLES = {
    "main_loss": "-",
    "gray_world": "--",
    "dcp": "-.",
    "rgb_sat": ":",
    "rgb_sv": (0, (3, 1, 1, 1)),  # densely dashdotted
}

# Colorblind-safe experiment palette (ColorBrewer Dark2)
EXPERIMENT_PALETTE = [
    "#1b9e77",  # teal
    "#d95f02",  # orange
    "#7570b3",  # purple
    "#e7298a",  # pink
    "#66a61e",  # green
    "#e6ab02",  # gold
    "#a6761d",  # brown
    "#666666",  # gray
]
# Line styles to pair with experiment palette (cycle if more experiments)
EXPERIMENT_LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------

def apply_style():
    """Apply publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE_LABEL,
        "axes.titlesize": FONT_SIZE_TITLE,
        "axes.labelsize": FONT_SIZE_LABEL,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "lines.linewidth": LINE_WIDTH,
    })


def apply_legend(ax, loc="best", ncol=1, outside=False, **kwargs):
    """Apply standardized legend to an axes.

    Args:
        ax: matplotlib axes
        loc: legend location (default: "best")
        ncol: number of columns
        outside: if True, place legend below the axes
        **kwargs: passed through to ax.legend()

    Returns:
        Legend object
    """
    legend_kw = dict(
        fontsize=FONT_SIZE_LEGEND,
        frameon=True,
        facecolor="white",
        framealpha=0.85,
        edgecolor="none",
        borderpad=0.4,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    if outside:
        # Place legend below the x-axis label. The xlabel sits around y=-0.15
        # to -0.20 of axes height after tight_layout, so -0.38 keeps the legend
        # clear of the xlabel text across the figure heights we use.
        legend_kw.update(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.38),
            ncol=ncol,
        )
    else:
        legend_kw.update(loc=loc, ncol=ncol)
    legend_kw.update(kwargs)
    return ax.legend(**legend_kw)


def step_formatter():
    """Format x-axis steps as '10K', '20K', etc."""
    def _fmt(x, _pos):
        if x >= 1000:
            return f"{x / 1000:.0f}K"
        return f"{x:.0f}"
    return FuncFormatter(_fmt)


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def add_phase_boundaries(ax, boundaries, label=True):
    """Draw vertical dashed lines at phase boundaries with optional labels.

    Args:
        ax: matplotlib axes
        boundaries: list of (name, start, end) from compute_phase_boundaries()
        label: whether to add phase labels in the upper margin
    """
    if not boundaries:
        return

    for name, start, _end in boundaries:
        if start > 0:
            ax.axvline(start, color="#999999", linestyle="--", linewidth=0.8,
                       alpha=0.7, zorder=1)

    if label:
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for name, start, end in boundaries:
            mid = (start + end) / 2
            phase_label = PHASE_LABELS.get(name, name)
            ax.text(mid, 0.97, phase_label, ha="center", va="top",
                    fontsize=FONT_SIZE_LEGEND, color="#666666", style="italic",
                    transform=trans,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.8, pad=1))


def add_phase_shading(ax, boundaries):
    """Add colored background shading for each phase.

    Args:
        ax: matplotlib axes
        boundaries: list of (name, start, end) from compute_phase_boundaries()
    """
    for name, start, end in boundaries:
        color = PHASE_COLORS.get(name, "#F5F5F5")
        ax.axvspan(start, end, alpha=0.15, color=color, zorder=0)


def annotate_peak(ax, step, value, label=None, color="black"):
    """Add a marker and text annotation at a peak point.

    Args:
        ax: matplotlib axes
        step: x-coordinate (step number)
        value: y-coordinate (metric value)
        label: text to display (default: auto-generated)
        color: marker/text color
    """
    ax.plot(step, value, marker="*", markersize=8, color=color, zorder=5)
    if label is None:
        label = f"Peak: {value:.2f} @ {step / 1000:.0f}K"
    ax.annotate(label, xy=(step, value), xytext=(8, 8),
                textcoords="offset points", fontsize=FONT_SIZE_LEGEND,
                color=color, ha="left", va="bottom")


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def save_figure(fig, name, output_dir, formats=("pdf", "png")):
    """Save figure to output directory in specified formats.

    Args:
        fig: matplotlib figure
        name: filename stem (no extension)
        output_dir: output directory path
        formats: tuple of format strings (default: both pdf and png)

    Returns:
        list of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path)
        saved.append(path)
        print(f"  Saved: {path}")
    plt.close(fig)
    return saved

"""Generate publication-quality figures from TensorBoard training data.

Usage:
    python scripts/paper_figures.py <figure> <experiments...> [options]

Figures:
    psnr              PSNR training trajectory with phase boundaries
    medium-params     Beta_D and B_inf evolution (per-channel RGB)
    loss-components   Per-loss component curves (absolute or budget mode)
    gaussian-count    Gaussian count trajectory (densification saturation)
    phase2-spike      Phase 2 onset spike (zoomed view)
    cross-compare     Cross-experiment overlay of a single metric
    psnr-gap          PSNR-to-clean gap analysis (bar chart)
    early-stopping    Early stopping window visualization
    medium-activity   Medium contribution + attenuation/backscatter magnitude
    all               Generate all single-experiment figures

Path resolution:
    Same as read_tb.py / eval_experiments.py — accepts experiment name substrings,
    method dirs, or full timestamp paths.

Examples:
    python scripts/paper_figures.py psnr baseline_30k
    python scripts/paper_figures.py all dyn03_tor_anneal_high
    python scripts/paper_figures.py cross-compare dyn01_tor_dcp005 dyn01_tor_dcp010 --metric psnr
    python scripts/paper_figures.py psnr-gap dyn01_tor_* dyn02_tor_*
"""

import argparse
import sys
from pathlib import Path

# Add scripts/ to path
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from paper_figures.style import apply_style
from paper_figures.data import load_experiments, get_short_label
from read_config import resolve_outputs_dir


# ---------------------------------------------------------------------------
# Figure registry
# ---------------------------------------------------------------------------

SINGLE_FIGURES = [
    "psnr",
    "medium-params",
    "loss-components",
    "gaussian-count",
    "phase2-spike",
    "medium-activity",
]

MULTI_FIGURES = [
    "cross-compare",
    "psnr-gap",
    "early-stopping",
]

ALL_FIGURES = SINGLE_FIGURES + MULTI_FIGURES


# Subcommand name → module name (when they differ)
_MODULE_MAP = {
    "psnr": "psnr_trajectory",
    "cross-compare": "cross_experiment",
}


def _import_figure_module(name):
    """Lazily import a figure module by subcommand name."""
    module_name = _MODULE_MAP.get(name, name.replace("-", "_"))
    mod = __import__(f"paper_figures.figures.{module_name}", fromlist=[module_name])
    return mod


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_single_figure(args, figure_name):
    """Generate a single-experiment figure."""
    outputs_dir = resolve_outputs_dir(args.outputs_dir)
    experiments = load_experiments(args.experiments, outputs_dir)
    if not experiments:
        print("Error: No experiments found with TensorBoard data.", file=sys.stderr)
        sys.exit(1)

    # Apply custom labels if provided
    if args.label:
        labels = args.label.split(",")
        for i, exp in enumerate(experiments):
            if i < len(labels):
                exp.label = labels[i].strip()

    mod = _import_figure_module(figure_name)
    formats = _parse_formats(args.format)
    output_dir = Path(args.output_dir)

    for exp in experiments:
        short = get_short_label(exp)
        print(f"Generating {figure_name} for {short}...")
        mod.plot(
            exp,
            output_dir=output_dir / "single",
            smooth_window=args.smooth,
            formats=formats,
            width=args.width,
            no_phase=args.no_phase,
            # Pass figure-specific args
            budget=getattr(args, "budget", False),
        )


def cmd_multi_figure(args, figure_name):
    """Generate a multi-experiment figure."""
    outputs_dir = resolve_outputs_dir(args.outputs_dir)
    experiments = load_experiments(args.experiments, outputs_dir)
    if not experiments:
        print("Error: No experiments found with TensorBoard data.", file=sys.stderr)
        sys.exit(1)

    if args.label:
        labels = args.label.split(",")
        for i, exp in enumerate(experiments):
            if i < len(labels):
                exp.label = labels[i].strip()

    mod = _import_figure_module(figure_name)
    formats = _parse_formats(args.format)
    output_dir = Path(args.output_dir)

    print(f"Generating {figure_name} for {len(experiments)} experiments...")
    mod.plot(
        experiments,
        output_dir=output_dir / "comparison",
        smooth_window=args.smooth,
        formats=formats,
        width=args.width,
        no_phase=args.no_phase,
        metric=getattr(args, "metric", "psnr"),
    )


def cmd_all(args):
    """Generate all single-experiment figures."""
    outputs_dir = resolve_outputs_dir(args.outputs_dir)
    experiments = load_experiments(args.experiments, outputs_dir)
    if not experiments:
        print("Error: No experiments found with TensorBoard data.", file=sys.stderr)
        sys.exit(1)

    formats = _parse_formats(args.format)
    output_dir = Path(args.output_dir)

    for exp in experiments:
        short = get_short_label(exp)
        print(f"\n{'='*60}")
        print(f"Generating all figures for {short}")
        print(f"{'='*60}")

        for fig_name in SINGLE_FIGURES:
            try:
                mod = _import_figure_module(fig_name)
                print(f"\n  {fig_name}...")
                mod.plot(
                    exp,
                    output_dir=output_dir / "single",
                    smooth_window=args.smooth,
                    formats=formats,
                    width=args.width,
                    no_phase=args.no_phase,
                    budget=False,
                )
            except Exception as e:
                print(f"  Warning: {fig_name} failed: {e}", file=sys.stderr)

        # Also generate budget view of loss-components
        try:
            mod = _import_figure_module("loss-components")
            print(f"\n  loss-components (budget)...")
            mod.plot(
                exp,
                output_dir=output_dir / "single",
                smooth_window=args.smooth,
                formats=formats,
                width=args.width,
                no_phase=args.no_phase,
                budget=True,
            )
        except Exception as e:
            print(f"  Warning: loss-components (budget) failed: {e}", file=sys.stderr)


def _parse_formats(fmt_str):
    """Parse format string into tuple of formats."""
    if fmt_str == "both":
        return ("pdf", "png")
    return (fmt_str,)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures from TensorBoard data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[1] if "Examples:" in __doc__ else "",
    )
    sub = parser.add_subparsers(dest="figure", help="Figure type to generate")

    # Common arguments added to all subparsers
    def add_common_args(p):
        p.add_argument("experiments", nargs="+", help="Experiment path specs")
        p.add_argument("--outputs-dir", default=None,
                       help="Base outputs directory (default: auto-detect)")
        p.add_argument("--output-dir", default="../fyp-playground/paper/figures",
                       help="Where to save figures (default: ../fyp-playground/paper/figures/)")
        p.add_argument("--format", choices=["pdf", "png", "both"], default="both",
                       help="Output format (default: both)")
        p.add_argument("--smooth", type=int, default=100,
                       help="EMA smoothing window in samples (default: 100, 0=disable)")
        p.add_argument("--width", choices=["single", "double"], default="single",
                       help="Figure width (default: single)")
        p.add_argument("--no-phase", action="store_true",
                       help="Suppress phase boundary annotations")
        p.add_argument("--label", default=None,
                       help="Custom labels for experiments (comma-separated)")

    # Single-experiment figures
    for name in SINGLE_FIGURES:
        p = sub.add_parser(name, help=f"Generate {name} figure")
        add_common_args(p)
        if name == "loss-components":
            p.add_argument("--budget", action="store_true",
                           help="Show losses as fraction of total")

    # Multi-experiment figures
    for name in MULTI_FIGURES:
        p = sub.add_parser(name, help=f"Generate {name} figure")
        add_common_args(p)
        if name == "cross-compare":
            p.add_argument("--metric", default="psnr",
                           help="Metric to compare (default: psnr)")

    # 'all' subcommand
    p = sub.add_parser("all", help="Generate all single-experiment figures")
    add_common_args(p)

    return parser


def main():
    apply_style()
    parser = build_parser()
    args = parser.parse_args()

    if args.figure is None:
        parser.print_help()
        sys.exit(1)

    if args.figure == "all":
        cmd_all(args)
    elif args.figure in SINGLE_FIGURES:
        cmd_single_figure(args, args.figure)
    elif args.figure in MULTI_FIGURES:
        cmd_multi_figure(args, args.figure)
    else:
        print(f"Unknown figure type: {args.figure}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

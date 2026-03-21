"""Run ns-eval on experiment runs and save metrics to each run's directory.

Usage:
    # Config mode — evaluate all experiments defined in experiment_config
    python scripts/eval_experiments.py --dry-run

    # Config mode — filter by experiment name substring
    python scripts/eval_experiments.py --filter torpedo --dry-run

    # Path mode — evaluate all runs under a method directory
    python scripts/eval_experiments.py ../fyp-playground/outputs/saltpond_unprocessed/saltpond_unprocessed-a_exploration/sea-splatfacto

    # Path mode — evaluate a single run by timestamp directory
    python scripts/eval_experiments.py ../fyp-playground/outputs/.../2026-03-08_015758

    # Path mode — evaluate multiple path specs
    python scripts/eval_experiments.py a_exploration b_exploration --outputs-dir ../fyp-playground/outputs

    # Skip runs that already have metrics
    python scripts/eval_experiments.py <path> --skip-existing

    # Also save rendered eval images
    python scripts/eval_experiments.py <path> --render-images
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from read_config import (
    _looks_like_timestamps,
    resolve_config_path,
    resolve_outputs_dir,
)
from log_experiments import find_runs, resolve_experiment_dir

sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))
from run_experiments import load_config

TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")

# Metrics to display (in order). Mean values only, skip std columns.
DISPLAY_METRICS = ["psnr", "clean_psnr", "ssim", "lpips"]


def read_metrics(metrics_path):
    """Read metrics from a saved ns-eval JSON file. Returns the results dict or None."""
    try:
        data = json.loads(metrics_path.read_text())
        return data.get("results", {})
    except (json.JSONDecodeError, OSError):
        return None


def format_metrics(metrics):
    """Format a metrics dict into a compact one-line string for display."""
    if not metrics:
        return ""
    parts = []
    for key in DISPLAY_METRICS:
        if key in metrics:
            parts.append(f"{key.replace('_', ' ').upper()}: {metrics[key]:.4f}")
    # Include any remaining non-std metrics not in the display list
    for key, val in metrics.items():
        if key not in DISPLAY_METRICS and not key.endswith("_std") and isinstance(val, (int, float)):
            parts.append(f"{key}: {val:.4f}")
    return "  |  ".join(parts)


def resolve_runs_from_config(experiments):
    """Resolve experiment config dicts to a flat list of timestamp run directories."""
    all_runs = []
    for exp in experiments:
        method_dir = Path(exp["output_dir"]) / exp["extra_args"]["experiment-name"] / exp["model"]
        if not method_dir.is_dir():
            print(f"Warning: directory not found, skipping: {method_dir}", file=sys.stderr)
            continue
        runs = find_runs(method_dir)
        if not runs:
            print(f"Warning: no runs found in {method_dir}", file=sys.stderr)
            continue
        all_runs.extend(runs)
    return all_runs


def resolve_runs(spec, outputs_dir):
    """Resolve a path spec to a list of timestamp run directories.

    Accepts:
        1. A timestamp directory directly
        2. A method-level directory containing timestamp runs
        3. A substring/path spec resolvable via resolve_experiment_dir or resolve_config_path
    """
    spec_path = Path(spec).expanduser().resolve()

    # Direct timestamp directory
    if spec_path.is_dir() and TIMESTAMP_RE.match(spec_path.name):
        return [spec_path]

    # Try resolving as a method-level directory with timestamp runs
    try:
        experiment_dir = resolve_experiment_dir(spec, outputs_dir)
        runs = find_runs(experiment_dir)
        if runs:
            return runs
    except SystemExit:
        pass

    # Fall back to single-run resolution via config path
    try:
        config_path = resolve_config_path(spec, outputs_dir)
        run_dir = config_path.parent
        if run_dir.is_dir() and TIMESTAMP_RE.match(run_dir.name):
            return [run_dir]
    except SystemExit:
        pass

    print(f"Error: Could not resolve '{spec}' to any experiment runs.", file=sys.stderr)
    sys.exit(1)


def validate_run(run_dir):
    """Check that a run directory has config.yml and at least one checkpoint."""
    config_path = run_dir / "config.yml"
    if not config_path.is_file():
        return False, "missing config.yml"

    models_dir = run_dir / "nerfstudio_models"
    if not models_dir.is_dir():
        return False, "missing nerfstudio_models/"

    checkpoints = list(models_dir.glob("*.ckpt"))
    if not checkpoints:
        return False, "no checkpoints in nerfstudio_models/"

    return True, None


def build_eval_command(run_dir, output_name, render_images, render_dir_name):
    """Build the ns-eval command for a run."""
    config_path = run_dir / "config.yml"
    output_path = run_dir / output_name

    cmd = [
        "ns-eval",
        "--load-config", str(config_path),
        "--output-path", str(output_path),
    ]

    if render_images:
        render_path = run_dir / render_dir_name
        cmd += ["--render-output-path", str(render_path)]

    return cmd


def run_eval(run_dir, output_name, render_images, render_dir_name):
    """Run ns-eval for a single run directory. Returns a result dict."""
    cmd = build_eval_command(run_dir, output_name, render_images, render_dir_name)
    label = f"{run_dir.parent.name}/{run_dir.name}"

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        duration = time.time() - start

        if proc.returncode != 0:
            print(proc.stdout, end="")
            return {
                "label": label,
                "status": f"failed (exit code {proc.returncode})",
                "duration": duration,
            }

        # Read back saved metrics
        metrics_path = run_dir / output_name
        metrics = read_metrics(metrics_path)
        return {"label": label, "status": "success", "duration": duration, "metrics": metrics}

    except FileNotFoundError:
        duration = time.time() - start
        return {"label": label, "status": "failed (ns-eval not found)", "duration": duration}


def print_summary(results):
    """Print a summary table of evaluation results."""
    # Check if any results have metrics to determine table layout
    has_metrics = any(r.get("metrics") for r in results)

    if has_metrics:
        metric_headers = [m.replace("_", " ").upper() for m in DISPLAY_METRICS]
        metric_hdr = "".join(f" {h:>8}" for h in metric_headers)
        print(f"\n{'Run':<50} {'Status':<15} {'Duration':>8}{metric_hdr}")
        print("-" * (73 + 9 * len(DISPLAY_METRICS)))
        for r in results:
            dur = f"{r['duration']:.1f}s" if r["duration"] > 0 else "\u2014"
            metrics = r.get("metrics", {}) or {}
            metric_vals = ""
            for key in DISPLAY_METRICS:
                if key in metrics:
                    metric_vals += f" {metrics[key]:>8.4f}"
                else:
                    metric_vals += f" {'—':>8}"
            print(f"{r['label']:<50} {r['status']:<15} {dur:>8}{metric_vals}")
    else:
        print(f"\n{'Run':<50} {'Status':<25} {'Duration':>10}")
        print("-" * 85)
        for r in results:
            dur = f"{r['duration']:.1f}s" if r["duration"] > 0 else "\u2014"
            print(f"{r['label']:<50} {r['status']:<25} {dur:>10}")


def main():
    parser = argparse.ArgumentParser(
        description="Run ns-eval on experiment runs and save metrics.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Experiment path specs (timestamp dir, method dir, or substring). "
        "If omitted, uses config mode.",
    )
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="Base outputs directory (default: $NERFSTUDIO_OUTPUTS or ./outputs)",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent.parent / "config" / "experiment_config.py"),
        help="Path to config .py file or module name (default: config/experiment_config.py)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only evaluate experiments whose name contains this substring (config mode only)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Only evaluate experiments for this dataset (matches dataset prefix in name)",
    )
    parser.add_argument(
        "--output-name",
        default="metrics.json",
        help="Metrics JSON filename (default: metrics.json)",
    )
    parser.add_argument(
        "--render-images",
        action="store_true",
        help="Also save rendered eval images",
    )
    parser.add_argument(
        "--render-dir-name",
        default="eval_renders",
        help="Subdirectory name for rendered images (default: eval_renders)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have a metrics file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args()

    if args.paths and args.filter:
        parser.error("--filter cannot be used with positional paths")

    # Resolve runs from either config mode or path mode
    if args.paths:
        outputs_dir = resolve_outputs_dir(args.outputs_dir)
        all_runs = []
        for spec in args.paths:
            runs = resolve_runs(spec, outputs_dir)
            all_runs.extend(runs)
    else:
        config = load_config(args.config)
        experiments = config.EXPERIMENTS
        if args.dataset:
            experiments = [e for e in experiments if e["name"].startswith(args.dataset + "/")]
        if args.filter:
            experiments = [e for e in experiments if args.filter in e["name"]]
        if not experiments:
            print("No experiments match the filter.", file=sys.stderr)
            sys.exit(1)
        all_runs = resolve_runs_from_config(experiments)

    # Deduplicate while preserving order
    seen = set()
    unique_runs = []
    for run in all_runs:
        key = run.resolve()
        if key not in seen:
            seen.add(key)
            unique_runs.append(run)

    if not unique_runs:
        print("No runs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Resolved {len(unique_runs)} run(s)\n")

    results = []
    for i, run_dir in enumerate(unique_runs):
        label = f"{run_dir.parent.name}/{run_dir.name}"

        # Validate run
        valid, reason = validate_run(run_dir)
        if not valid:
            print(f"[{i + 1}/{len(unique_runs)}] Skipping {label}: {reason}")
            results.append({"label": label, "status": f"skipped ({reason})", "duration": 0})
            continue

        # Check for existing metrics
        metrics_path = run_dir / args.output_name
        if args.skip_existing and metrics_path.is_file():
            print(f"[{i + 1}/{len(unique_runs)}] Skipping {label}: {args.output_name} already exists")
            results.append({"label": label, "status": "skipped (existing)", "duration": 0})
            continue

        cmd = build_eval_command(run_dir, args.output_name, args.render_images, args.render_dir_name)

        if args.dry_run:
            print(f"[{i + 1}/{len(unique_runs)}] {label}")
            print(f"    {' '.join(cmd)}")
            print()
            continue

        print(f"[{i + 1}/{len(unique_runs)}] Evaluating {label}")
        print(f"    {' '.join(cmd)}")
        result = run_eval(run_dir, args.output_name, args.render_images, args.render_dir_name)
        results.append(result)
        print(f"  -> {result['status']} ({result['duration']:.1f}s)")
        if result.get("metrics"):
            print(f"     {format_metrics(result['metrics'])}")

    if not args.dry_run and results:
        print_summary(results)


if __name__ == "__main__":
    main()

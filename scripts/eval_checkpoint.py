"""Evaluate and render at a specific training checkpoint.

Patches config.yml to set load_step, then runs ns-eval and optionally ns-render
at that checkpoint. Produces metrics_step{N}.json alongside the regular metrics.json.

Usage:
    python scripts/eval_checkpoint.py <experiment_path> --step <N>
    python scripts/eval_checkpoint.py <experiment_path> --step <N> --render
    python scripts/eval_checkpoint.py <experiment_path> --step <N> --dry-run

Path resolution:
    Same as eval_experiments.py — accepts experiment name substrings,
    method dirs, or full timestamp paths.

Examples:
    # Eval at peak decomposition step
    python scripts/eval_checkpoint.py dyn03_tor_anneal_high --step 19999

    # Eval + render at specific step
    python scripts/eval_checkpoint.py saltpond-repl00_anneal --step 23999 --render
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from eval_experiments import resolve_runs, read_metrics
from read_config import resolve_outputs_dir


def find_checkpoint(run_dir, step):
    """Find the checkpoint file for a given step.

    Args:
        run_dir: timestamp-level run directory
        step: target step number

    Returns:
        Path to checkpoint file, or None if not found
    """
    models_dir = Path(run_dir) / "nerfstudio_models"
    if not models_dir.is_dir():
        return None
    ckpt = models_dir / f"step-{step:09d}.ckpt"
    return ckpt if ckpt.is_file() else None


def list_checkpoints(run_dir):
    """List all available checkpoint steps.

    Returns:
        sorted list of step numbers
    """
    models_dir = Path(run_dir) / "nerfstudio_models"
    if not models_dir.is_dir():
        return []
    steps = []
    for f in models_dir.glob("step-*.ckpt"):
        match = re.search(r"step-(\d+)\.ckpt", f.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def snap_to_checkpoint(run_dir, target_step):
    """Find the nearest available checkpoint to a target step.

    Returns:
        (actual_step, checkpoint_path) or (None, None) if no checkpoints
    """
    steps = list_checkpoints(run_dir)
    if not steps:
        return None, None

    closest = min(steps, key=lambda s: abs(s - target_step))
    ckpt = Path(run_dir) / "nerfstudio_models" / f"step-{closest:09d}.ckpt"
    return closest, ckpt


def create_patched_config(run_dir, step):
    """Create a config.yml copy with load_step set to the target step.

    Args:
        run_dir: timestamp-level run directory
        step: checkpoint step to load

    Returns:
        Path to the patched config file
    """
    config_path = Path(run_dir) / "config.yml"
    patched_path = Path(run_dir) / f"config_step{step}.yml"

    config_text = config_path.read_text()

    # Patch load_step: null → load_step: <step>
    patched = re.sub(
        r"^load_step:.*$",
        f"load_step: {step}",
        config_text,
        flags=re.MULTILINE,
    )

    if patched == config_text:
        print(f"Warning: load_step field not found in {config_path}", file=sys.stderr)

    patched_path.write_text(patched)
    return patched_path


def run_eval_at_step(run_dir, step, render=False, dry_run=False):
    """Run ns-eval (and optionally ns-render) at a specific checkpoint.

    Returns:
        dict with status, metrics, duration
    """
    run_dir = Path(run_dir)
    label = f"{run_dir.parent.parent.name}/{run_dir.name}"

    # Validate checkpoint exists
    actual_step, ckpt_path = snap_to_checkpoint(run_dir, step)
    if actual_step is None:
        return {"label": label, "status": "no checkpoints found"}

    if actual_step != step:
        print(f"  Note: Step {step} not found, using nearest: {actual_step}")
        step = actual_step

    print(f"  Checkpoint: {ckpt_path.name}")

    # Create patched config
    patched_config = create_patched_config(run_dir, step)
    metrics_path = run_dir / f"metrics_step{step}.json"

    # Build eval command
    eval_cmd = [
        "ns-eval",
        "--load-config", str(patched_config),
        "--output-path", str(metrics_path),
    ]

    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(eval_cmd)}")
        if render:
            print(f"  [DRY RUN] Would also render at step {step}")
        return {"label": label, "status": "dry-run", "step": step}

    # Run eval
    print(f"  Running ns-eval at step {step}...")
    start = time.time()
    proc = subprocess.run(eval_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    duration = time.time() - start

    if proc.returncode != 0:
        print(proc.stdout, end="")
        return {"label": label, "status": f"eval failed (exit {proc.returncode})", "step": step, "duration": duration}

    # Read metrics
    metrics = read_metrics(metrics_path) if metrics_path.is_file() else None
    result = {"label": label, "status": "success", "step": step, "duration": duration, "metrics": metrics}

    # Run render if requested
    if render:
        print(f"  Running ns-render at step {step}...")
        render_cmd = [
            "ns-render", "dataset",
            "--load-config", str(patched_config),
            "--rendered-output-names", "clean_rgb", "medium_rgb", "backscatter", "attenuation_map", "depth",
            "--split", "test",
            "--output-path", str(run_dir / f"renders_step{step}"),
        ]
        render_start = time.time()
        render_proc = subprocess.run(render_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        render_duration = time.time() - render_start

        if render_proc.returncode != 0:
            print(render_proc.stdout, end="")
            result["render_status"] = f"failed (exit {render_proc.returncode})"
        else:
            result["render_status"] = "success"
        result["render_duration"] = render_duration

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate at a specific training checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[1] if "Examples:" in __doc__ else "",
    )
    parser.add_argument("experiment", help="Experiment path spec")
    parser.add_argument("--step", type=int, required=True,
                        help="Checkpoint step to evaluate at")
    parser.add_argument("--render", action="store_true",
                        help="Also render at the checkpoint")
    parser.add_argument("--outputs-dir", default=None,
                        help="Base outputs directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without running")
    parser.add_argument("--list-checkpoints", action="store_true",
                        help="List available checkpoints and exit")

    args = parser.parse_args()
    outputs_dir = resolve_outputs_dir(args.outputs_dir)
    runs = resolve_runs(args.experiment, outputs_dir)

    for run_dir in runs:
        label = f"{run_dir.parent.parent.name}/{run_dir.name}"
        print(f"\n=== {label} ===")

        if args.list_checkpoints:
            steps = list_checkpoints(run_dir)
            if steps:
                print(f"  Available checkpoints: {', '.join(str(s) for s in steps)}")
            else:
                print("  No checkpoints found.")
            continue

        result = run_eval_at_step(run_dir, args.step, render=args.render, dry_run=args.dry_run)

        if result.get("metrics"):
            metrics = result["metrics"]
            for key, val in sorted(metrics.items()):
                if isinstance(val, float):
                    print(f"  {key}: {val:.4f}")

        print(f"  Status: {result['status']}")
        if result.get("duration"):
            print(f"  Duration: {result['duration']:.1f}s")


if __name__ == "__main__":
    main()

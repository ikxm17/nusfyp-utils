"""
Run nerfstudio training experiments defined in a config file.

Usage:
    # Preview all commands without executing
    python scripts/run_experiments.py --dry-run

    # Run all experiments
    python scripts/run_experiments.py

    # Run only experiments whose name contains "torpedo"
    python scripts/run_experiments.py --filter torpedo

    # Use a custom config file
    python scripts/run_experiments.py --config /path/to/my_config.py
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def load_config(config_path: str):
    """Import a Python config module by file path or module name.

    If config_path points to an existing .py file, its parent directory is
    added to sys.path and the module is imported by stem name.  Otherwise it
    is treated as a module name relative to this script's directory.
    """
    path = Path(config_path)
    if path.suffix == ".py" and path.exists():
        sys.path.insert(0, str(path.resolve().parent))
        module_name = path.stem
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        module_name = config_path
    return importlib.import_module(module_name)


def build_command(experiment: dict) -> list[str]:
    """Build an ns-train CLI command from an experiment config dict."""
    cmd = ["ns-train", experiment["model"]]

    cmd += ["--data", str(experiment["data"])]

    output_dir = experiment.get("output_dir", "./outputs")
    cmd += ["--output-dir", str(output_dir)]

    vis = experiment.get("vis", "viewer")
    cmd += ["--vis", vis]

    if experiment.get("viewer") is False:
        cmd += ["--viewer.quit-on-train-completion", "True"]

    for key, value in experiment.get("extra_args", {}).items():
        cmd += [f"--{key}", str(value)]

    for key, value in experiment.get("method_args", {}).items():
        cmd += [f"--{key}", str(value)]

    return cmd


def run_experiment(
    experiment: dict,
    index: int,
    total: int,
    log_dir: str,
    log_all: bool,
) -> dict:
    """Run a single experiment and return a result dict."""
    cmd = build_command(experiment)
    name = experiment["name"]
    print(f"[{index + 1}/{total}] Starting: {name}")
    print(f"    {' '.join(cmd)}")

    Path(experiment.get("output_dir", "./outputs")).mkdir(parents=True, exist_ok=True)
    dataset_name = name.split("/")[0]
    log_subdir = Path(log_dir) / dataset_name
    log_subdir.mkdir(parents=True, exist_ok=True)

    exp_name = experiment["extra_args"].get("experiment-name", name.replace("/", "-"))
    if log_all:
        log_file = log_subdir / f"{exp_name}_{index}.log"
    else:
        log_file = log_subdir / f"{exp_name}.log"

    start = time.time()
    try:
        with open(log_file, "w") as f:
            proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in proc.stdout:
                sys.stdout.write(line)
                f.write(line)
            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
        status = "success"
    except subprocess.CalledProcessError as e:
        status = f"failed (exit code {e.returncode})"
        failed_log = log_subdir / f"{index}_{exp_name}_FAILED_{int(time.time())}.log"
        shutil.copy2(log_file, failed_log)
        log_file = failed_log
    except FileNotFoundError:
        status = "failed (ns-train not found)"
        failed_log = log_subdir / f"{index}_{exp_name}_FAILED_{int(time.time())}.log"
        if log_file.exists():
            shutil.copy2(log_file, failed_log)
            log_file = failed_log

    duration = time.time() - start
    result = {"name": name, "status": status, "duration": duration, "log": str(log_file)}
    print(f"  -> {status} ({duration / 60:.1f} min) -- log: {log_file}")
    return result


def print_summary(results: list[dict]) -> None:
    """Print a summary table of experiment results."""
    print(f"\n{'Experiment':<40} {'Status':<25} {'Duration':>10}")
    print("-" * 75)
    for r in results:
        dur = f"{r['duration'] / 60:.1f} min" if r["duration"] > 0 else "—"
        print(f"{r['name']:<40} {r['status']:<25} {dur:>10}")


def main():
    parser = argparse.ArgumentParser(
        description="Run nerfstudio training experiments.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview commands without executing",
    )
    parser.add_argument(
        "--config",
        default="experiment_config",
        help="Path to config .py file or module name (default: experiment_config)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only run experiments whose name contains this substring",
    )
    parser.add_argument(
        "--no-log-index",
        action="store_true",
        help="Disable appending _{index} to log filenames",
    )
    args = parser.parse_args()

    # Cleaner logs — nerfstudio uses rich for terminal output
    os.environ["NO_COLOR"] = "1"
    os.environ["TERM"] = "dumb"

    # Startup info
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
    print(f"Conda environment: {env_name}")
    print(f"Python: {sys.executable}")
    print(f"ns-train found: {shutil.which('ns-train')}")

    # Load config
    config = load_config(args.config)
    experiments = config.EXPERIMENTS
    log_dir = config.LOG_DIR

    if args.filter:
        experiments = [e for e in experiments if args.filter in e["name"]]

    print(f"Running {len(experiments)} experiments {'(dry run)' if args.dry_run else ''}\n")

    if args.dry_run:
        for i, exp in enumerate(experiments):
            cmd = build_command(exp)
            print(f"[{i}] {exp['name']}")
            print(f"    {' '.join(cmd)}")
            print()
        return

    results = []
    for i, exp in enumerate(experiments):
        result = run_experiment(exp, i, len(experiments), log_dir, log_all=not args.no_log_index)
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()

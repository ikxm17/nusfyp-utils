"""
Run nerfstudio training experiments defined in a config file.

Usage:
    # Preview all commands without executing
    python scripts/experiments/run_experiments.py --dry-run

    # Run all experiments
    python scripts/experiments/run_experiments.py

    # Run only experiments whose name contains "torpedo"
    python scripts/experiments/run_experiments.py --filter torpedo

    # Use a custom config file
    python scripts/experiments/run_experiments.py --config /path/to/my_config.py
"""

from __future__ import annotations

import argparse
import dataclasses
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


def validate_extra_args(experiments: list[dict]) -> list[str]:
    """Check that extra_args keys are valid ns-train flags. Returns error list.

    Returns an empty list when validation passes or when nerfstudio is not
    importable (prints a warning to stderr and skips validation).
    """
    try:
        from nerfstudio.configs.method_configs import all_methods
    except ImportError:
        print("Warning: nerfstudio not importable — skipping flag validation", file=sys.stderr)
        return []

    warnings = []
    checked: set[tuple[str, str]] = set()

    for exp in experiments:
        model = exp["model"]
        if model not in all_methods:
            warnings.append(f"Unknown model: {model}")
            continue

        config = all_methods[model]
        for flag in exp.get("extra_args", {}):
            if (model, flag) in checked:
                continue
            checked.add((model, flag))

            parts = flag.split(".")
            current = config
            valid = True
            for part in parts:
                attr = part.replace("-", "_")
                if not dataclasses.is_dataclass(type(current)):
                    valid = False
                    break
                field_names = {f.name for f in dataclasses.fields(type(current))}
                if attr not in field_names:
                    valid = False
                    break
                current = getattr(current, attr)

            if not valid:
                warnings.append(f"  {exp['name']}: --{flag} not found for model '{model}'")

    return warnings


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
    log_index: bool,
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
    prefix = f"expt_{index}_" if log_index else ""
    log_file = log_subdir / f"{prefix}{exp_name}.log"

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
        final_log = log_subdir / f"SUCCESS_{prefix}{exp_name}.log"
    except subprocess.CalledProcessError as e:
        status = f"failed (exit code {e.returncode})"
        fail_prefix = f"expt_{index}_" if not log_index else prefix
        final_log = log_subdir / f"FAILED_{fail_prefix}{exp_name}.log"
    except FileNotFoundError:
        status = "failed (ns-train not found)"
        fail_prefix = f"expt_{index}_" if not log_index else prefix
        final_log = log_subdir / f"FAILED_{fail_prefix}{exp_name}.log"

    if log_file.exists():
        log_file.rename(final_log)
        log_file = final_log

    duration = time.time() - start
    result = {"name": name, "status": status, "duration": duration, "log": str(log_file)}
    print(f"  -> {status} ({duration / 60:.1f} min) -- log: {log_file}")
    return result


def print_summary(results: list[dict]) -> None:
    """Print a summary table of experiment results."""
    print(f"\n{'Experiment':<40} {'Status':<25} {'Duration':>10}")
    print("-" * 75)
    for r in results:
        dur = f"{r['duration'] / 60:.1f} min" if r["duration"] > 0 else "\u2014"
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
        default=str(Path(__file__).resolve().parent.parent.parent / "config" / "experiment_config.py"),
        help="Path to config .py file or module name (default: config/experiment_config.py)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only run experiments whose name contains this substring",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Only run experiments for this dataset (matches dataset prefix in name)",
    )
    parser.add_argument(
        "--log-index",
        action="store_true",
        help="Prefix expt_{index}_ to log filenames",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Run only the experiment at this 0-based index (applied after --filter)",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Print the number of experiments and exit",
    )
    args = parser.parse_args()

    # Load config early so --count can exit before startup info
    config = load_config(args.config)
    experiments = config.EXPERIMENTS
    log_dir = config.LOG_DIR

    if args.dataset:
        experiments = [e for e in experiments if e["name"].startswith(args.dataset + "/")]
    if args.filter:
        experiments = [e for e in experiments if args.filter in e["name"]]

    if args.count:
        print(len(experiments))
        return

    # Cleaner logs — nerfstudio uses rich for terminal output
    os.environ["NO_COLOR"] = "1"
    os.environ["TERM"] = "dumb"

    # Startup info
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
    print(f"Conda environment: {env_name}")
    print(f"Python: {sys.executable}")
    print(f"ns-train found: {shutil.which('ns-train')}")

    if args.index is not None:
        if args.index < 0 or args.index >= len(experiments):
            print(
                f"Error: --index {args.index} out of range "
                f"(0-{len(experiments) - 1}, {len(experiments)} experiments)",
                file=sys.stderr,
            )
            sys.exit(1)
        experiments = [experiments[args.index]]

    # Validate extra_args flags — fail fast on unrecognized options
    warnings = validate_extra_args(experiments)
    if warnings:
        print("\nFlag validation FAILED:", file=sys.stderr)
        for w in warnings:
            print(f"  {w}", file=sys.stderr)
        print(file=sys.stderr)
        sys.exit(1)

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
        result = run_experiment(exp, i, len(experiments), log_dir, log_index=args.log_index)
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()

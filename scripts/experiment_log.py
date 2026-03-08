"""Generate an experiment log describing runs and their config diffs against a baseline.

Usage:
    python scripts/experiment_log.py <experiment-dir> [options]

    # Use the earliest run as baseline (default), log all runs in the directory
    python scripts/experiment_log.py ../fyp-playground/outputs/saltpond_unprocessed/saltpond_unprocessed-a_exploration/sea-splatfacto

    # Pick a specific baseline by timestamp substring
    python scripts/experiment_log.py <dir> --baseline 024717

    # Include extra runs from other directories
    python scripts/experiment_log.py <dir> --extra ../other-outputs/experiment/method

    # Write output to a file instead of stdout
    python scripts/experiment_log.py <dir> -o experiment_log.txt

    # Only diff the model section (default), or choose optimizers/all
    python scripts/experiment_log.py <dir> --section model

    # Use a different outputs-dir for path resolution
    python scripts/experiment_log.py a_exploration --outputs-dir ../fyp-playground/outputs
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

from config_reader import (
    _descend_to_config,
    _looks_like_timestamps,
    extract_section,
    load_config,
    resolve_config_path,
    resolve_outputs_dir,
)

TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")


def find_runs(directory):
    """Find all timestamp-based run directories under the given path."""
    directory = Path(directory)
    runs = sorted(
        [d for d in directory.iterdir() if d.is_dir() and TIMESTAMP_RE.match(d.name)]
    )
    return runs


def resolve_experiment_dir(spec, outputs_dir):
    """Resolve a spec to a method-level directory containing timestamp runs.

    Accepts:
        1. Direct path to a directory containing timestamp subdirs
        2. Path spec resolved via config_reader (descends to the method level)
    """
    spec_path = Path(spec).expanduser().resolve()

    # Direct path: check if it contains timestamp subdirs
    if spec_path.is_dir():
        subdirs = sorted([d for d in spec_path.iterdir() if d.is_dir()])
        if subdirs and _looks_like_timestamps(subdirs):
            return spec_path

        # Maybe it's a higher-level dir (dataset or experiment) — descend
        # Try to find a method dir with timestamps underneath
        for d in subdirs:
            inner = sorted([dd for dd in d.iterdir() if dd.is_dir()] if d.is_dir() else [])
            if inner and _looks_like_timestamps(inner):
                return d
            for dd in inner:
                deepest = sorted([ddd for ddd in dd.iterdir() if ddd.is_dir()] if dd.is_dir() else [])
                if deepest and _looks_like_timestamps(deepest):
                    return dd

    # Fall back to config_reader path resolution — get the config, then go up to method dir
    try:
        config_path = resolve_config_path(spec, outputs_dir)
        # config_path is like .../method/timestamp/config.yml
        method_dir = config_path.parent.parent
        subdirs = sorted([d for d in method_dir.iterdir() if d.is_dir()])
        if subdirs and _looks_like_timestamps(subdirs):
            return method_dir
    except SystemExit:
        pass

    print(f"Error: Could not resolve '{spec}' to an experiment directory with timestamp runs.", file=sys.stderr)
    sys.exit(1)


def pick_baseline(runs, baseline_spec):
    """Select the baseline run from the list.

    If baseline_spec is None, picks the earliest (first sorted).
    Otherwise, matches by substring on the directory name.
    """
    if baseline_spec is None:
        return runs[0]

    matches = [r for r in runs if baseline_spec in r.name]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) == 0:
        print(f"Error: No run matching baseline '{baseline_spec}'. Available:", file=sys.stderr)
        for r in runs:
            print(f"  {r.name}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Error: Multiple runs matching baseline '{baseline_spec}':", file=sys.stderr)
        for m in matches:
            print(f"  {m.name}", file=sys.stderr)
        sys.exit(1)


def format_timestamp(ts_name):
    """Convert '2026-03-08_024717' to a readable datetime string."""
    try:
        dt = datetime.strptime(ts_name, "%Y-%m-%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return ts_name


def diff_dicts(baseline_dict, other_dict):
    """Compute differences between two flat dicts.

    Returns (changed, only_baseline, only_other) where:
        changed = [(key, baseline_val, other_val), ...]
        only_baseline = [(key, val), ...]
        only_other = [(key, val), ...]
    """
    all_keys = sorted(set(list(baseline_dict.keys()) + list(other_dict.keys())))
    changed, only_baseline, only_other = [], [], []

    for k in all_keys:
        in_b = k in baseline_dict
        in_o = k in other_dict

        if in_b and in_o:
            vb, vo = baseline_dict[k], other_dict[k]
            if str(vb) != str(vo):
                changed.append((k, vb, vo))
        elif in_b:
            only_baseline.append((k, baseline_dict[k]))
        else:
            only_other.append((k, other_dict[k]))

    return changed, only_baseline, only_other


def format_run_header(run_dir, label=None, is_baseline=False):
    """Format a header line for a run."""
    ts = format_timestamp(run_dir.name)
    tag = " [BASELINE]" if is_baseline else ""
    name = label or run_dir.name
    return f"--- {name} ({ts}){tag} ---"


def generate_log(runs, baseline_run, section, extra_runs=None, extra_labels=None):
    """Generate the experiment log as a list of lines."""
    lines = []

    # Header
    experiment_dir = baseline_run.parent
    lines.append(f"Experiment Log")
    lines.append(f"==============")
    lines.append(f"Directory: {experiment_dir}")
    lines.append(f"Method:    {experiment_dir.name}")
    lines.append(f"Section:   {section}")
    lines.append(f"Runs:      {len(runs)}" + (f" + {len(extra_runs)} external" if extra_runs else ""))
    lines.append(f"Baseline:  {baseline_run.name} ({format_timestamp(baseline_run.name)})")
    lines.append("")

    # Load baseline config
    baseline_config_path = baseline_run / "config.yml"
    if not baseline_config_path.is_file():
        baseline_config_path = _descend_to_config(baseline_run)
    baseline_config = load_config(baseline_config_path)
    baseline_dict = extract_section(baseline_config, section)

    # Baseline section
    lines.append(format_run_header(baseline_run, is_baseline=True))
    lines.append(f"Config: {baseline_config_path}")
    lines.append("")
    if section == "model":
        lines.append("Model config:")
    elif section == "optimizers":
        lines.append("Optimizer config:")
    else:
        lines.append("Full config:")
    for k, v in sorted(baseline_dict.items()):
        lines.append(f"  {k}: {v}")
    lines.append("")

    # Other runs from the same directory
    other_runs = [r for r in runs if r != baseline_run]

    all_other = [(r, None) for r in other_runs]
    if extra_runs:
        for i, er in enumerate(extra_runs):
            label = extra_labels[i] if extra_labels and i < len(extra_labels) else None
            all_other.append((er, label))

    for run_dir, label in all_other:
        config_path = run_dir / "config.yml"
        if not config_path.is_file():
            try:
                config_path = _descend_to_config(run_dir)
            except SystemExit:
                lines.append(format_run_header(run_dir, label=label))
                lines.append("  (no config.yml found)")
                lines.append("")
                continue

        config = load_config(config_path)
        other_dict = extract_section(config, section)

        display_label = label or run_dir.name
        lines.append(format_run_header(run_dir, label=display_label))
        lines.append(f"Config: {config_path}")

        changed, only_base, only_other = diff_dicts(baseline_dict, other_dict)

        if not changed and not only_base and not only_other:
            lines.append("  No differences from baseline.")
        else:
            if changed:
                lines.append("")
                lines.append("  Changed:")
                max_key = max(len(k) for k, _, _ in changed)
                for k, vb, vo in changed:
                    lines.append(f"    {k:<{max_key}}  baseline: {vb}")
                    lines.append(f"    {'':<{max_key}}  this:     {vo}")
            if only_base:
                lines.append("")
                lines.append("  Only in baseline:")
                for k, v in only_base:
                    lines.append(f"    {k}: {v}")
            if only_other:
                lines.append("")
                lines.append("  Only in this run:")
                for k, v in only_other:
                    lines.append(f"    {k}: {v}")

        lines.append("")

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Generate an experiment log with config diffs against a baseline.",
    )
    parser.add_argument(
        "experiment_dir",
        help="Path to experiment directory (method-level, containing timestamp runs), "
             "or a spec resolvable via config_reader",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Timestamp substring to select as baseline (default: earliest run)",
    )
    parser.add_argument(
        "--extra",
        nargs="+",
        default=None,
        help="Additional experiment directories or run paths to include",
    )
    parser.add_argument(
        "--extra-labels",
        nargs="+",
        default=None,
        help="Labels for extra runs (matched by position)",
    )
    parser.add_argument(
        "--section",
        choices=["model", "optimizers", "all"],
        default="model",
        help="Config section to show/diff (default: model)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="Base outputs directory for path resolution",
    )

    args = parser.parse_args()
    outputs_dir = resolve_outputs_dir(args.outputs_dir)

    # Resolve main experiment directory
    experiment_dir = resolve_experiment_dir(args.experiment_dir, outputs_dir)
    runs = find_runs(experiment_dir)

    if not runs:
        print(f"Error: No timestamp runs found in {experiment_dir}", file=sys.stderr)
        sys.exit(1)

    baseline = pick_baseline(runs, args.baseline)

    # Resolve extra runs
    extra_runs = []
    if args.extra:
        for spec in args.extra:
            spec_path = Path(spec).expanduser().resolve()
            if spec_path.is_dir() and TIMESTAMP_RE.match(spec_path.name):
                # Direct timestamp directory
                extra_runs.append(spec_path)
            elif spec_path.is_dir():
                # Method-level directory — add all its runs
                extra_runs.extend(find_runs(spec_path))
            else:
                # Try config_reader resolution
                try:
                    extra_dir = resolve_experiment_dir(spec, outputs_dir)
                    extra_runs.extend(find_runs(extra_dir))
                except SystemExit:
                    print(f"Warning: Could not resolve extra spec '{spec}', skipping.", file=sys.stderr)

    log_lines = generate_log(runs, baseline, args.section, extra_runs, args.extra_labels)
    output_text = "\n".join(log_lines) + "\n"

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text)
        print(f"Log written to {output_path}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()

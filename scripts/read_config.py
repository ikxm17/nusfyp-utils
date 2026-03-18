"""CLI tool for reading and diffing nerfstudio experiment configs.

Usage:
    python scripts/config_reader.py read <path> [--section model|optimizers|all] [--param <name>]
    python scripts/config_reader.py diff <path-a> <path-b> [--section model|optimizers|all] [--name-a X] [--name-b Y]

Path resolution:
    - Full path to config.yml or its parent directory
    - Relative path from outputs dir (e.g. saltpond_unprocessed/saltpond_unprocessed-a_exploration)
    - Substring match on dataset/experiment names (e.g. a_exploration)

When multiple methods or timestamps exist, the script auto-selects if there's only one,
or picks the latest timestamp. If multiple methods exist, it lists them and exits.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import yaml


def resolve_outputs_dir(cli_arg):
    """Resolve the base outputs directory from CLI arg, env var, or fallback."""
    if cli_arg:
        return Path(cli_arg).expanduser().resolve()
    env = os.environ.get("NERFSTUDIO_OUTPUTS")
    if env:
        return Path(env).expanduser().resolve()
    return Path("./outputs").resolve()


def _looks_like_timestamps(dirs):
    """Check if directory names match YYYY-MM-DD_HHMMSS format."""
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")
    return all(pattern.match(d.name) for d in dirs)


def _descend_to_config(directory):
    """Walk from any level in the hierarchy down to config.yml.

    Handles: dataset/ -> experiment/ -> method/ -> timestamp/ -> config.yml
    At each level, auto-selects if only one subdir exists.
    For timestamps, picks the latest. For methods, errors if ambiguous.
    """
    directory = Path(directory)

    # Already pointing at config.yml
    config_file = directory / "config.yml"
    if config_file.is_file():
        return config_file

    subdirs = sorted([d for d in directory.iterdir() if d.is_dir()])
    if not subdirs:
        print(f"Error: No subdirectories found in {directory}", file=sys.stderr)
        sys.exit(1)

    # Check if subdirs look like timestamps -> pick latest
    if _looks_like_timestamps(subdirs):
        latest = subdirs[-1]  # lexicographic sort works for ISO format
        config_file = latest / "config.yml"
        if config_file.is_file():
            return config_file
        print(f"Error: No config.yml found in {latest}", file=sys.stderr)
        sys.exit(1)

    # If only one subdir, descend into it
    if len(subdirs) == 1:
        return _descend_to_config(subdirs[0])

    # Multiple non-timestamp subdirs (likely methods) -> ambiguous
    print(f"Error: Multiple subdirectories found in {directory}:", file=sys.stderr)
    for d in subdirs:
        print(f"  {d.name}", file=sys.stderr)
    print("Please specify which one to use.", file=sys.stderr)
    sys.exit(1)


def _find_matching_dirs(spec, outputs_dir):
    """Search for matching dataset/experiment directories.

    Prefers exact match on the experiment suffix (after stripping dataset
    prefix) before falling back to substring matching.  This avoids
    ambiguity when one experiment name is a prefix of another (e.g.
    ``tune02_seathru18k`` vs ``tune02_seathru18k_gw20``).
    """
    outputs_dir = Path(outputs_dir)
    matches = []
    exact = []

    if not outputs_dir.is_dir():
        return matches

    for dataset_dir in sorted(outputs_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        prefix = dataset_dir.name + "-"
        for experiment_dir in sorted(dataset_dir.iterdir()):
            if not experiment_dir.is_dir():
                continue
            relative = f"{dataset_dir.name}/{experiment_dir.name}"
            # Derive the short experiment name (strip dataset prefix)
            short = experiment_dir.name
            if short.startswith(prefix):
                short = short[len(prefix):]
            # Check for exact match on short name or full relative path
            if spec == short or spec == experiment_dir.name or spec == relative:
                exact.append(experiment_dir)
            elif spec in relative:
                matches.append(experiment_dir)

    return exact if exact else matches


def resolve_config_path(spec, outputs_dir):
    """Resolve a user-provided spec to a config.yml path.

    Accepts:
        1. Full path to config.yml or its parent directory
        2. Relative path from outputs dir (e.g. dataset/experiment)
        3. Substring match on dataset/experiment names
    """
    spec_path = Path(spec).expanduser()

    # 1. Full/explicit path
    if spec_path.is_absolute():
        if spec_path.is_file() and spec_path.name == "config.yml":
            return spec_path
        if spec_path.is_dir():
            return _descend_to_config(spec_path)

    # 2. Relative path from outputs dir
    candidate = Path(outputs_dir) / spec
    if candidate.is_dir():
        return _descend_to_config(candidate)
    if candidate.is_file() and candidate.name == "config.yml":
        return candidate

    # 3. Substring match
    matches = _find_matching_dirs(spec, outputs_dir)
    if len(matches) == 1:
        return _descend_to_config(matches[0])
    elif len(matches) > 1:
        print(f"Error: Multiple matches for '{spec}':", file=sys.stderr)
        for m in matches:
            print(f"  {m.relative_to(outputs_dir)}", file=sys.stderr)
        print("Please be more specific.", file=sys.stderr)
        sys.exit(1)

    print(f"Error: Could not resolve '{spec}' to a config path.", file=sys.stderr)
    sys.exit(1)


def load_config(config_path):
    """Load a nerfstudio config.yml using unsafe_load for Python dataclasses."""
    try:
        with open(config_path, "r") as f:
            return yaml.unsafe_load(f)
    except Exception as e:
        print(f"Error loading {config_path}: {e}", file=sys.stderr)
        sys.exit(1)


def extract_section(config, section):
    """Extract a section from the config as a flat dict."""
    if section == "model":
        return vars(config.pipeline.model)
    elif section == "optimizers":
        out = {}
        for group, opts in config.optimizers.items():
            opt = opts.get("optimizer")
            sch = opts.get("scheduler")
            if opt:
                for k, v in vars(opt).items():
                    if k != "_target":
                        out[f"{group}.optimizer.{k}"] = v
            if sch:
                for k, v in vars(sch).items():
                    if k != "_target":
                        out[f"{group}.scheduler.{k}"] = v
        return out
    else:
        return vars(config)


def print_config(params, section, param):
    """Display config for the read subcommand."""
    if param:
        if param in params:
            print(f"{param}: {params[param]}")
        else:
            print(f"Param '{param}' not found. Available params:")
            for k in sorted(params.keys()):
                print(f"  {k}")
    else:
        for k, v in sorted(params.items()):
            print(f"{k}: {v}")


def _derive_label(config_path, outputs_dir):
    """Auto-derive a diff label from the config path.

    Returns <experiment>/<timestamp> (e.g. saltpond-a_exploration/2026-03-08_031123).
    Falls back to the full path if it can't be parsed.
    """
    try:
        rel = Path(config_path).relative_to(outputs_dir)
        parts = rel.parts
        # Expected: dataset / experiment / method / timestamp / config.yml
        if len(parts) >= 4:
            return f"{parts[1]}/{parts[3]}"
        return str(rel)
    except ValueError:
        return str(config_path)


def print_diff(dict_a, dict_b, section, name_a, name_b):
    """Display diff for the diff subcommand."""
    all_keys = sorted(set(list(dict_a.keys()) + list(dict_b.keys())))

    changed, only_a, only_b = [], [], []

    for k in all_keys:
        in_a = k in dict_a
        in_b = k in dict_b

        if in_a and in_b:
            va, vb = dict_a[k], dict_b[k]
            if str(va) != str(vb):
                changed.append((k, va, vb))
        elif in_a:
            only_a.append((k, dict_a[k]))
        else:
            only_b.append((k, dict_b[k]))

    section_label = section if section != "all" else "top-level"

    if not changed and not only_a and not only_b:
        print(f"[{section_label}] Configs are identical.")
        return

    if changed:
        print(f"=== Changed ({section_label}) ===")
        max_key_len = max(len(k) for k, _, _ in changed)
        for k, va, vb in changed:
            print(f"  {k:<{max_key_len}}  {name_a}: {va}")
            print(f"  {'':<{max_key_len}}  {name_b}: {vb}")
            print()

    if only_a:
        print(f"=== Only in {name_a} ({section_label}) ===")
        for k, v in only_a:
            print(f"  {k}: {v}")
        print()

    if only_b:
        print(f"=== Only in {name_b} ({section_label}) ===")
        for k, v in only_b:
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="Read and diff nerfstudio experiment configs.",
    )

    # Shared arguments via parent parser so --outputs-dir works after the subcommand
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--outputs-dir",
        default=None,
        help="Base outputs directory (default: $NERFSTUDIO_OUTPUTS or ./outputs)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # read subcommand
    read_parser = subparsers.add_parser("read", parents=[shared], help="Read a config")
    read_parser.add_argument("path", help="Path or prefix to config")
    read_parser.add_argument(
        "--section",
        choices=["model", "optimizers", "all"],
        default="model",
        help="Config section to display (default: model)",
    )
    read_parser.add_argument(
        "--param",
        default=None,
        help="Specific parameter name to look up",
    )

    # diff subcommand
    diff_parser = subparsers.add_parser("diff", parents=[shared], help="Diff two configs")
    diff_parser.add_argument("path_a", help="Path or prefix to first config")
    diff_parser.add_argument("path_b", help="Path or prefix to second config")
    diff_parser.add_argument(
        "--section",
        choices=["model", "optimizers", "all"],
        default="model",
        help="Config section to diff (default: model)",
    )
    diff_parser.add_argument("--name-a", default=None, help="Label for first config")
    diff_parser.add_argument("--name-b", default=None, help="Label for second config")

    args = parser.parse_args()
    outputs_dir = resolve_outputs_dir(args.outputs_dir)

    if args.command == "read":
        config_path = resolve_config_path(args.path, outputs_dir)
        config = load_config(config_path)
        params = extract_section(config, args.section)
        print_config(params, args.section, args.param)

    elif args.command == "diff":
        config_path_a = resolve_config_path(args.path_a, outputs_dir)
        config_path_b = resolve_config_path(args.path_b, outputs_dir)
        config_a = load_config(config_path_a)
        config_b = load_config(config_path_b)
        dict_a = extract_section(config_a, args.section)
        dict_b = extract_section(config_b, args.section)
        name_a = args.name_a or _derive_label(config_path_a, outputs_dir)
        name_b = args.name_b or _derive_label(config_path_b, outputs_dir)
        print_diff(dict_a, dict_b, args.section, name_a, name_b)


if __name__ == "__main__":
    main()

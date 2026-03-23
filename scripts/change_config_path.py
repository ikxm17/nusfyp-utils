"""
Rewrite absolute PosixPath entries in a Nerfstudio config.yml so the
config can be loaded on a different machine. Edits the file in-place.

Usage:
    python change_config_path.py config.yml --old-base /home/olduser --new-base /home/newuser
    python change_config_path.py config.yml --old-base /home/olduser  # auto-detects new base as $HOME

Examples:
    # Replace /home/saber with /home/alice
    python change_config_path.py config.yml --old-base /home/saber --new-base /home/alice

    # Replace /home/saber with current $HOME, and also remap data path
    python change_config_path.py config.yml --old-base /home/saber \
        --old-data /home/islabella/workspaces/irwin_ws/fyp-playground/datasets \
        --new-data /home/alice/datasets

    # Create a .bak backup before editing
    python change_config_path.py config.yml --old-base /home/saber --backup
"""

import argparse
import os
import re
import sys
import shutil
from pathlib import Path


def fix_config(
    config_path: str,
    old_base: str,
    new_base: str,
    old_data: str | None = None,
    new_data: str | None = None,
    backup: bool = False,
) -> str:
    """
    Read a nerfstudio YAML config and replace path prefixes.

    The YAML contains PosixPath objects serialized as e.g.:

        !!python/object/apply:pathlib.PosixPath
        - /
        - home
        - saber
        - workspaces
        - ...

    We do a line-level rewrite: split old_base into components and match
    the sequence of "- component" lines, then replace with new_base components.
    """
    with open(config_path, "r") as f:
        lines = f.readlines()

    def path_to_components(p: str) -> list[str]:
        """'/home/saber/work' -> ['/', 'home', 'saber', 'work']"""
        parts = Path(p).parts
        return list(parts)

    def make_replacement_map(old: str, new: str) -> tuple[list[str], list[str]]:
        return path_to_components(old), path_to_components(new)

    replacements = [make_replacement_map(old_base, new_base)]
    if old_data and new_data:
        replacements.insert(0, make_replacement_map(old_data, new_data))

    output_lines = []
    i = 0
    while i < len(lines):
        matched = False
        for old_parts, new_parts in replacements:
            # Check if lines[i:i+len(old_parts)] match "- part" for each part
            if i + len(old_parts) <= len(lines):
                match = True
                for j, part in enumerate(old_parts):
                    stripped = lines[i + j].rstrip("\n")
                    # Match lines like "- home" or "- /" (with possible leading whitespace)
                    expected_pattern = re.compile(
                        r"^(\s*)-\s+" + re.escape(part) + r"\s*$"
                    )
                    if not expected_pattern.match(stripped):
                        match = False
                        break
                if match:
                    # Grab the indentation from the first matched line
                    indent_match = re.match(r"^(\s*)-", lines[i])
                    indent = indent_match.group(1) if indent_match else ""
                    # Write new path components
                    for part in new_parts:
                        output_lines.append(f"{indent}- {part}\n")
                    i += len(old_parts)
                    matched = True
                    break
        if not matched:
            output_lines.append(lines[i])
            i += 1

    result = "".join(output_lines)

    if backup:
        backup_path = config_path + ".bak"
        shutil.copy2(config_path, backup_path)
        print(f"Backup saved to {backup_path}", file=sys.stderr)

    with open(config_path, "w") as f:
        f.write(result)
    print(f"Updated {config_path}", file=sys.stderr)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fix absolute paths in nerfstudio config.yml for cross-machine use."
    )
    parser.add_argument("config", help="Path to the nerfstudio config.yml")
    parser.add_argument(
        "--old-base",
        required=True,
        help="Old base path prefix to replace (e.g. /home/saber)",
    )
    parser.add_argument(
        "--new-base",
        default=None,
        help="New base path prefix (default: current $HOME)",
    )
    parser.add_argument(
        "--old-data",
        default=None,
        help="Optional: old data path prefix to replace separately",
    )
    parser.add_argument(
        "--new-data",
        default=None,
        help="Optional: new data path prefix",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a .bak backup before editing",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    new_base = args.new_base or os.environ.get("HOME", str(Path.home()))

    fix_config(
        config_path=args.config,
        old_base=args.old_base,
        new_base=new_base,
        old_data=args.old_data,
        new_data=args.new_data,
        backup=args.backup,
    )


if __name__ == "__main__":
    main()

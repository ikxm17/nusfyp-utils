"""Shared argparse helpers for the scripts/ CLIs.

Surfaces module docstring examples in --help so usage hints are visible at the
command line, not just in the source file.
"""

import argparse


_EXAMPLE_MARKERS = ("Examples:", "Usage:")


def _extract_examples(module_doc):
    """Return the substring of module_doc starting at the first 'Examples:' or 'Usage:'.

    Returns None if neither marker is present or the doc is empty.
    """
    if not module_doc:
        return None
    earliest = None
    for marker in _EXAMPLE_MARKERS:
        idx = module_doc.find(marker)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is None:
        return None
    return module_doc[earliest:].rstrip()


def make_parser(description, module_doc=None, **kwargs):
    """Construct an ArgumentParser whose --help epilog shows usage examples.

    `module_doc` should be the calling module's `__doc__`. The first 'Examples:'
    or 'Usage:' section onwards is rendered verbatim as the epilog (preserving
    line breaks via RawDescriptionHelpFormatter).
    """
    epilog = _extract_examples(module_doc)
    kwargs.setdefault("formatter_class", argparse.RawDescriptionHelpFormatter)
    return argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        **kwargs,
    )


def format_header(title, **fields):
    """Render a metadata header block for the top of a report.

    Returns a string suitable for printing once at the start of stdout output:

        === Title ===
        Field Name: value
        Field Name: value

    Keys are converted from snake_case to Title Case for display. Pairs whose
    value is None or "" are omitted, so callers can pass through optional flags
    without needing to filter at the call site.
    """
    lines = [f"=== {title} ==="]
    for key, value in fields.items():
        if value is None or value == "":
            continue
        display_key = key.replace("_", " ").title()
        lines.append(f"{display_key}: {value}")
    return "\n".join(lines)

"""Gather all quantitative analysis data for a batch of experiments into structured JSON.

Replaces ~25 individual tool calls during auto-analyze with a single script invocation.

Usage:
    python scripts/agents/analyze_batch.py tune10 \
        --outputs-dir ../fyp-playground/outputs \
        --analysis-dir /tmp/batch-analysis \
        --dataset-analysis ../fyp-playground/datasets/saltpond/saltpond_unprocessed/analysis.md \
        --num-frames 3 \
        --output-types rgb underwater_rgb depth \
        --max-width 480

Steps:
    1. Find experiments matching <batch_prefix>_*
    2. Read each experiment's metrics.json
    3. Run read_tb.py compare for TB analysis
    4. Run compare_renders.py info for frame counts
    5. Pick evenly-spaced representative frames
    6. Generate comparison grids (compare_renders.py grid)
    7. Extract clean renders (compare_renders.py extract)
    8. Run dataset_underwater.py on extracted rgb renders
    9. Parse dataset input analysis (if provided)
    10. Write structured JSON report
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent.parent  # scripts/ (parent of agents/)


def log(msg):
    """Print progress to stderr."""
    print(msg, file=sys.stderr)


def run_script(script_name, args, capture_stdout=True):
    """Run a sibling Python script, returning (stdout, stderr, returncode).

    Uses the same interpreter as the current process.
    """
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)] + args
    log(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(SCRIPTS_DIR),
    )
    return result.stdout, result.stderr, result.returncode


def pick_frames(total, num_frames):
    """Pick evenly-spaced frame indices from 0 to total-1."""
    if total <= 0:
        return []
    if num_frames <= 0:
        return []
    if num_frames == 1:
        return [0]
    if num_frames >= total:
        return list(range(total))
    # Evenly space: first, last, and middle points
    return [round(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)]


# ---------------------------------------------------------------------------
# Step 1: Find experiments
# ---------------------------------------------------------------------------

def find_experiments(outputs_dir, batch_prefix, dataset=None):
    """Glob for experiments matching the batch prefix.

    Looks for: <outputs_dir>/<dataset_dir>/<dataset_dir>-<batch_prefix>_*/sea-splatfacto/*/

    When *dataset* is provided, only that dataset directory is searched,
    avoiding cross-dataset contamination when multiple datasets share
    the same batch prefix (e.g. ``tune02_*`` in both saltpond and curasao).

    Returns list of (short_name, full_spec, timestamp_dir) tuples sorted
    by short_name, where:
        short_name  – display label, e.g. ``tune02_dwr2``
        full_spec   – unambiguous relative path for sub-scripts,
                      e.g. ``curasao_unprocessed/curasao_unprocessed-tune02_dwr2``
        timestamp_dir – absolute Path to the latest timestamp directory
    """
    outputs_path = Path(outputs_dir)
    experiments = []

    if dataset:
        # Only search the specified dataset directory
        dataset_dirs = [outputs_path / dataset]
        if not dataset_dirs[0].is_dir():
            return experiments
    else:
        dataset_dirs = sorted(
            d for d in outputs_path.iterdir() if d.is_dir()
        )

    for dataset_dir in dataset_dirs:
        pattern = f"{dataset_dir.name}-{batch_prefix}_*"
        for exp_dir in sorted(dataset_dir.glob(pattern)):
            if not exp_dir.is_dir():
                continue

            # Extract short experiment name (strip dataset prefix)
            exp_name = exp_dir.name
            prefix = dataset_dir.name + "-"
            if exp_name.startswith(prefix):
                exp_name = exp_name[len(prefix):]

            # Build unambiguous spec: dataset/experiment (resolves via
            # read_config.py's relative-path step without substring search)
            full_spec = f"{dataset_dir.name}/{exp_dir.name}"

            # Find method dir and latest timestamp
            method_dir = exp_dir / "sea-splatfacto"
            if not method_dir.is_dir():
                continue

            timestamp_dirs = sorted([
                d for d in method_dir.iterdir()
                if d.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}_\d{6}$", d.name)
            ])
            if not timestamp_dirs:
                continue

            # Use latest timestamp
            latest = timestamp_dirs[-1]
            experiments.append((exp_name, full_spec, latest))

    return experiments


# ---------------------------------------------------------------------------
# Step 1b: Extract timing from logs
# ---------------------------------------------------------------------------

def extract_timing(experiments, logs_dir, dataset):
    """Extract training, eval, and render durations from synced log files.

    Parses:
        - train.log:  "<dataset>/<suffix> success   54.2 min"
        - eval.log:   "sea-splatfacto/<timestamp>   success   29.7s ..."
        - render.log: "sea-splatfacto/<timestamp>   dataset: success   42.9s"

    Returns dict mapping experiment short name to timing dict.
    """
    timing = {}
    logs_path = Path(logs_dir)

    # Build lookup: short_name -> timestamp dir name
    ts_lookup = {}
    for exp_name, _, ts_dir in experiments:
        ts_lookup[exp_name] = ts_dir.name

    # --- Training time from train.log ---
    train_log = logs_path / "train.log"
    if train_log.is_file():
        text = train_log.read_text()
        for exp_name, _, _ in experiments:
            # Match: "<dataset>/<exp_name> success   54.2 min"
            # The suffix in the log is the short name without dataset prefix
            pattern = rf"{re.escape(dataset)}/{re.escape(exp_name)}\s+success\s+([\d.]+)\s+min"
            match = re.search(pattern, text)
            if match:
                timing.setdefault(exp_name, {})["train_min"] = float(match.group(1))

    # --- Eval time from eval.log ---
    eval_log = logs_path / "eval.log"
    if eval_log.is_file():
        text = eval_log.read_text()
        for exp_name, _, ts_dir in experiments:
            ts = ts_dir.name
            # Match: "sea-splatfacto/<timestamp>   success   29.7s ..."
            pattern = rf"sea-splatfacto/{re.escape(ts)}\s+success\s+([\d.]+)s"
            match = re.search(pattern, text)
            if match:
                timing.setdefault(exp_name, {})["eval_sec"] = float(match.group(1))

    # --- Render time from render.log ---
    render_log = logs_path / "render.log"
    if render_log.is_file():
        text = render_log.read_text()
        for exp_name, _, ts_dir in experiments:
            ts = ts_dir.name
            # Match: "sea-splatfacto/<timestamp>   dataset: success   42.9s"
            pattern = rf"sea-splatfacto/{re.escape(ts)}\s+dataset: success\s+([\d.]+)s"
            match = re.search(pattern, text)
            if match:
                timing.setdefault(exp_name, {})["render_sec"] = float(match.group(1))

    # --- Compute per-step ms from training time + step count ---
    # Step count comes from metrics.json (num_rays_per_sec implies steps ran)
    # or from the TB analysis total_steps. We just use train_min / max_iters.
    for exp_name, _, ts_dir in experiments:
        if exp_name not in timing or "train_min" not in timing[exp_name]:
            continue
        config_path = ts_dir / "config.yml"
        if config_path.is_file():
            # Quick parse: look for max_num_iterations in config
            config_text = config_path.read_text()
            match = re.search(r"max_num_iterations:\s*(\d+)", config_text)
            if match:
                max_iters = int(match.group(1))
                train_sec = timing[exp_name]["train_min"] * 60
                timing[exp_name]["per_step_ms"] = round(train_sec / max_iters * 1000, 2)

    # --- Render FPS from metrics.json ---
    for exp_name, _, ts_dir in experiments:
        metrics_path = ts_dir / "metrics.json"
        if metrics_path.is_file():
            try:
                data = json.loads(metrics_path.read_text())
                fps = data.get("results", {}).get("fps")
                if fps is not None:
                    timing.setdefault(exp_name, {})["render_fps"] = round(float(fps), 1)
            except (json.JSONDecodeError, OSError):
                pass

    return timing


# ---------------------------------------------------------------------------
# Step 2: Read metrics
# ---------------------------------------------------------------------------

def read_metrics(timestamp_dir):
    """Read metrics.json from a run directory.

    Returns dict with normalized keys (psnr, ssim, lpips, etc.) or None.
    """
    metrics_path = timestamp_dir / "metrics.json"
    if not metrics_path.is_file():
        return None

    try:
        data = json.loads(metrics_path.read_text())
        results = data.get("results", {})

        # Normalize keys — strip 'eval/' prefix if present, keep only mean values
        normalized = {}
        for key, val in results.items():
            if key.endswith("_std"):
                continue
            # Strip common prefixes
            clean_key = key.replace("eval/", "")
            if isinstance(val, (int, float)):
                normalized[clean_key] = val

        return normalized
    except (json.JSONDecodeError, OSError) as e:
        log(f"  Warning: failed to read {metrics_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Step 3: TB analysis
# ---------------------------------------------------------------------------

def run_tb_analysis(experiments, outputs_dir):
    """Run read_tb.py compare on all experiments.

    Returns the full text output.
    """
    if not experiments:
        return "No experiments to compare."

    exp_specs = [spec for _, spec, _ in experiments]
    args = ["compare"] + exp_specs + ["--verbose", "--outputs-dir", str(outputs_dir)]
    stdout, stderr, rc = run_script("read_tb.py", args)

    if rc != 0:
        msg = f"read_tb.py compare failed (exit {rc})"
        if stderr:
            msg += f":\n{stderr.strip()}"
        return msg

    return stdout.strip() if stdout.strip() else "No TB data available."


# ---------------------------------------------------------------------------
# Step 4: Render info
# ---------------------------------------------------------------------------

def get_render_info(first_experiment_name, outputs_dir):
    """Run compare_renders.py info on the first experiment to get frame count.

    Returns total_frames (int) or 0.
    """
    args = ["info", first_experiment_name, "--outputs-dir", str(outputs_dir)]
    stdout, stderr, rc = run_script("compare_renders.py", args)

    if rc != 0:
        log(f"  Warning: compare_renders.py info failed: {stderr.strip()}")
        return 0

    # Parse frame count from output table
    # Format: "  output_type    N   WxH"
    max_frames = 0
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                count = int(parts[1])
                max_frames = max(max_frames, count)
            except ValueError:
                continue

    return max_frames


# ---------------------------------------------------------------------------
# Step 6: Comparison grids
# ---------------------------------------------------------------------------

def generate_grids(experiments, frames, output_types, analysis_dir, outputs_dir,
                   max_width):
    """Run compare_renders.py grid to generate comparison grids.

    Returns list of saved grid image paths.
    """
    if not experiments or not frames:
        return []

    exp_specs = [spec for _, spec, _ in experiments]
    grid_dir = Path(analysis_dir) / "grids"

    args = (
        ["grid"]
        + exp_specs
        + ["--frames"] + [str(f) for f in frames]
        + ["--output-types"] + output_types
        + ["--output-dir", str(grid_dir)]
        + ["--outputs-dir", str(outputs_dir)]
    )
    if max_width:
        args += ["--max-width", str(max_width)]

    stdout, stderr, rc = run_script("compare_renders.py", args)

    if rc != 0:
        log(f"  Warning: compare_renders.py grid failed: {stderr.strip()}")
        return []

    # Collect saved grid paths from stdout ("Saved: <path>")
    grid_images = []
    for line in stdout.splitlines():
        if line.startswith("Saved: "):
            grid_images.append(line[7:].strip())

    # Also glob in case stdout parsing misses anything
    if not grid_images and grid_dir.is_dir():
        grid_subdir = grid_dir / "grid"
        if grid_subdir.is_dir():
            grid_images = sorted(str(p) for p in grid_subdir.glob("*.png"))

    return grid_images


# ---------------------------------------------------------------------------
# Step 7: Extract clean renders
# ---------------------------------------------------------------------------

def extract_renders(experiments, frames, analysis_dir, outputs_dir, max_width):
    """Run compare_renders.py extract for rgb frames of each experiment.

    Returns dict mapping experiment name to directory of extracted frames.
    """
    if not experiments or not frames:
        return {}

    renders_dir = Path(analysis_dir) / "renders"
    extract_dirs = {}

    for exp_name, exp_spec, _ in experiments:
        args = (
            ["extract", exp_spec]
            + ["--frames"] + [str(f) for f in frames]
            + ["--output-types", "rgb"]
            + ["--output-dir", str(renders_dir)]
            + ["--outputs-dir", str(outputs_dir)]
        )
        if max_width:
            args += ["--max-width", str(max_width)]

        stdout, stderr, rc = run_script("compare_renders.py", args)

        if rc != 0:
            log(f"  Warning: extract failed for {exp_name}: {stderr.strip()}")
            continue

        # Extract saves to output_dir/extract/<name>/
        exp_extract_dir = renders_dir / "extract" / exp_name
        if exp_extract_dir.is_dir():
            extract_dirs[exp_name] = str(exp_extract_dir)

    return extract_dirs


# ---------------------------------------------------------------------------
# Step 8: Color analysis
# ---------------------------------------------------------------------------

def run_color_analysis(extract_dirs):
    """Run dataset_underwater.py on extracted rgb frames for each experiment.

    Returns dict mapping experiment name to color metrics summary.
    """
    color_analysis = {}

    for exp_name, extract_dir in extract_dirs.items():
        # Check there are images to analyze
        extract_path = Path(extract_dir)
        pngs = list(extract_path.glob("*.png"))
        if not pngs:
            log(f"  Warning: no PNGs found in {extract_dir} for color analysis")
            continue

        args = [str(extract_dir), "--json"]
        stdout, stderr, rc = run_script("dataset_underwater.py", args)

        if rc != 0:
            log(f"  Warning: dataset_underwater.py failed for {exp_name}: {stderr.strip()}")
            continue

        try:
            data = json.loads(stdout)
            summary = data.get("summary", {})

            # Extract key metrics (mean values)
            color_analysis[exp_name] = {
                "rg_ratio": summary.get("rg_ratio", {}).get("mean"),
                "gw_deviation": summary.get("gw_deviation", {}).get("mean"),
                "lab_a_mean": summary.get("mean_a_star", {}).get("mean"),
                "lab_b_mean": summary.get("mean_b_star", {}).get("mean"),
                "dcp_mean": summary.get("dcp_mean", {}).get("mean"),
                "uciqe": summary.get("uciqe", {}).get("mean"),
                "uiqm": summary.get("uiqm", {}).get("mean"),
            }
        except (json.JSONDecodeError, KeyError) as e:
            log(f"  Warning: failed to parse color analysis for {exp_name}: {e}")
            continue

    return color_analysis


# ---------------------------------------------------------------------------
# Step 9: Parse dataset input analysis
# ---------------------------------------------------------------------------

def parse_dataset_analysis(analysis_path):
    """Parse an analysis.md file for input dataset color metrics.

    Looks for the Underwater Characteristics section with tables containing
    metrics like R/G ratio, Gray-world dev, CIELAB a*, DCP mean.

    Returns dict of metric values or None.
    """
    if not analysis_path:
        return None

    path = Path(analysis_path)
    if not path.is_file():
        log(f"  Warning: dataset analysis file not found: {analysis_path}")
        return None

    try:
        text = path.read_text()
    except OSError as e:
        log(f"  Warning: failed to read {analysis_path}: {e}")
        return None

    metrics = {}

    # Parse markdown tables for specific metrics
    # Table format: | Metric | Mean | Median | Std | Min | Max | ... |
    metric_patterns = {
        "rg_ratio": r"R/G ratio",
        "bg_ratio": r"B/G ratio",
        "gw_deviation": r"Gray-world dev",
        "lab_a_mean": r"CIELAB a\*",
        "lab_b_mean": r"CIELAB b\*",
        "uciqe": r"UCIQE",
        "uiqm": r"UIQM",
        "dcp_mean": r"DCP mean",
        "rms_contrast": r"RMS contrast",
        "edge_density": r"Edge density",
    }

    for key, pattern in metric_patterns.items():
        # Match table rows: | <metric_name> | <mean> | ...
        match = re.search(
            rf"\|\s*{pattern}\s*\|\s*([0-9eE.+-]+)\s*\|",
            text,
        )
        if match:
            try:
                metrics[key] = float(match.group(1))
            except ValueError:
                continue

    return metrics if metrics else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gather all quantitative analysis for a batch of experiments.",
    )
    parser.add_argument(
        "batch_prefix",
        help="Batch prefix to match experiments (e.g. tune10 matches tune10_*)",
    )
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="Base outputs directory (default: $NERFSTUDIO_OUTPUTS or ./outputs)",
    )
    parser.add_argument(
        "--analysis-dir",
        default="/tmp/batch-analysis",
        help="Directory for analysis artifacts (grids, renders) (default: /tmp/batch-analysis)",
    )
    parser.add_argument(
        "--dataset-analysis",
        default=None,
        help="Path to dataset analysis.md for input color metrics comparison",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=3,
        help="Number of representative frames to extract (default: 3)",
    )
    parser.add_argument(
        "--output-types",
        nargs="+",
        default=["rgb", "underwater_rgb", "depth", "accumulation", "backscatter",
                 "attenuation_map"],
        help="Output types for comparison grids (default: rgb underwater_rgb depth "
             "accumulation backscatter attenuation_map)",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=480,
        help="Max image width for renders (default: 480)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset directory name to scope the search (e.g. curasao_unprocessed). "
             "When omitted, all datasets are searched (may cause cross-dataset "
             "contamination if the batch prefix is not unique).",
    )
    parser.add_argument(
        "--logs-dir",
        default=None,
        help="Directory containing train.log, eval.log, render.log "
             "(default: <outputs-dir>/../logs)",
    )
    parser.add_argument(
        "--cleanup-tb",
        action="store_true",
        default=False,
        help="Delete local tfevents files after TB data has been extracted. "
             "Safe to use — the TB metrics are persisted to the JSON report.",
    )

    args = parser.parse_args()

    # Resolve outputs dir (same logic as read_config.py)
    if args.outputs_dir:
        outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    else:
        env = os.environ.get("NERFSTUDIO_OUTPUTS")
        if env:
            outputs_dir = Path(env).expanduser().resolve()
        else:
            outputs_dir = Path("./outputs").resolve()

    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    warnings = []

    report = {
        "batch_prefix": args.batch_prefix,
        "dataset": args.dataset,
        "experiments": [],
        "analysis_dir": str(analysis_dir),
        "warnings": warnings,
        "metrics": {},
        "tb_analysis": None,
        "render_info": {
            "total_frames": 0,
            "selected_frames": [],
        },
        "grid_images": [],
        "color_analysis": {},
        "timing": {},
        "dataset_input_metrics": None,
    }

    # -----------------------------------------------------------------------
    # Step 1: Find experiments
    # -----------------------------------------------------------------------
    log(f"[1/10] Finding experiments matching '{args.batch_prefix}_*'"
        + (f" in {args.dataset}..." if args.dataset else "..."))
    experiments = find_experiments(outputs_dir, args.batch_prefix, dataset=args.dataset)

    if not experiments:
        msg = f"No experiments found matching '{args.batch_prefix}_*' in {outputs_dir}"
        log(f"  {msg}")
        warnings.append({"step": "find_experiments", "message": msg, "path": str(outputs_dir)})
        report["tb_analysis"] = "No experiments found."
        # Write report and exit
        report_path = analysis_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(str(report_path))
        return

    exp_names = [name for name, _, _ in experiments]
    report["experiments"] = exp_names
    log(f"  Found {len(experiments)} experiments: {', '.join(exp_names)}")

    # -----------------------------------------------------------------------
    # Step 2: Read metrics
    # -----------------------------------------------------------------------
    log("[2/10] Reading metrics.json for each experiment...")
    for exp_name, _, ts_dir in experiments:
        metrics = read_metrics(ts_dir)
        if metrics:
            report["metrics"][exp_name] = metrics
            log(f"  {exp_name}: PSNR={metrics.get('psnr', '?'):.2f}, "
                f"SSIM={metrics.get('ssim', '?'):.3f}, "
                f"LPIPS={metrics.get('lpips', '?'):.3f}")
        else:
            msg = f"{exp_name}: metrics.json not found at {ts_dir / 'metrics.json'}"
            log(f"  {msg}")
            warnings.append({"step": "read_metrics", "message": msg, "path": str(ts_dir / "metrics.json")})

    # -----------------------------------------------------------------------
    # Step 2b: Extract timing from logs
    # -----------------------------------------------------------------------
    logs_dir = args.logs_dir or str(outputs_dir.parent / "logs")
    if args.dataset:
        log(f"[2b/10] Extracting timing from logs ({logs_dir})...")
        timing = extract_timing(experiments, logs_dir, args.dataset)
        report["timing"] = timing
        if timing:
            for exp_name in exp_names:
                t = timing.get(exp_name, {})
                parts = []
                if "train_min" in t:
                    parts.append(f"train={t['train_min']:.1f}min")
                if "eval_sec" in t:
                    parts.append(f"eval={t['eval_sec']:.1f}s")
                if "render_sec" in t:
                    parts.append(f"render={t['render_sec']:.1f}s")
                if "per_step_ms" in t:
                    parts.append(f"step={t['per_step_ms']:.1f}ms")
                if "render_fps" in t:
                    parts.append(f"fps={t['render_fps']:.1f}")
                if parts:
                    log(f"  {exp_name}: {', '.join(parts)}")
                else:
                    log(f"  {exp_name}: no timing data found")
        else:
            log("  No timing data found in logs")
    else:
        log("[2b/10] Skipping timing — --dataset required to match train.log entries")

    # -----------------------------------------------------------------------
    # Step 3: TB analysis
    # -----------------------------------------------------------------------
    log("[3/10] Running TensorBoard analysis...")
    report["tb_analysis"] = run_tb_analysis(experiments, outputs_dir)

    # -----------------------------------------------------------------------
    # Step 3b: Clean up TensorBoard files (optional)
    # -----------------------------------------------------------------------
    if args.cleanup_tb:
        log("[3b/10] Cleaning up local tfevents files...")
        deleted_count = 0
        deleted_bytes = 0
        for _, _, ts_dir in experiments:
            for tf_file in ts_dir.glob("events.out.tfevents.*"):
                deleted_bytes += tf_file.stat().st_size
                tf_file.unlink()
                deleted_count += 1
        if deleted_count:
            log(f"  Deleted {deleted_count} tfevents files, "
                f"reclaimed {deleted_bytes / 1024**3:.2f} GB")
        else:
            log("  No tfevents files found to clean up")

    # -----------------------------------------------------------------------
    # Step 4: Render info
    # -----------------------------------------------------------------------
    log("[4/10] Getting render frame count...")
    exp_specs = [spec for _, spec, _ in experiments]
    total_frames = get_render_info(exp_specs[0], outputs_dir)
    if total_frames == 0:
        log("  Warning: could not determine frame count, trying other experiments...")
        for spec in exp_specs[1:]:
            total_frames = get_render_info(spec, outputs_dir)
            if total_frames > 0:
                break

    report["render_info"]["total_frames"] = total_frames
    log(f"  Total frames: {total_frames}")

    if total_frames == 0:
        msg = "No render frames found for any experiment — renders may have failed"
        warnings.append({"step": "render_info", "message": msg})

    # -----------------------------------------------------------------------
    # Step 5: Pick representative frames
    # -----------------------------------------------------------------------
    log("[5/10] Selecting representative frames...")
    selected_frames = pick_frames(total_frames, args.num_frames)
    report["render_info"]["selected_frames"] = selected_frames
    log(f"  Selected frames: {selected_frames}")

    # -----------------------------------------------------------------------
    # Step 6: Generate comparison grids
    # -----------------------------------------------------------------------
    if selected_frames:
        log("[6/10] Generating comparison grids...")
        grid_images = generate_grids(
            experiments, selected_frames, args.output_types,
            analysis_dir, outputs_dir, args.max_width,
        )
        report["grid_images"] = grid_images
        log(f"  Generated {len(grid_images)} grid images")
    else:
        log("[6/10] Skipping grids — no frames available")

    # -----------------------------------------------------------------------
    # Step 7: Extract clean renders
    # -----------------------------------------------------------------------
    if selected_frames:
        log("[7/10] Extracting clean renders (rgb)...")
        extract_dirs = extract_renders(
            experiments, selected_frames, analysis_dir, outputs_dir,
            args.max_width,
        )
        log(f"  Extracted renders for {len(extract_dirs)} experiments")
    else:
        log("[7/10] Skipping render extraction — no frames available")
        extract_dirs = {}

    # -----------------------------------------------------------------------
    # Step 8: Color analysis
    # -----------------------------------------------------------------------
    if extract_dirs:
        log("[8/10] Running color analysis on extracted renders...")
        report["color_analysis"] = run_color_analysis(extract_dirs)
        log(f"  Color analysis complete for {len(report['color_analysis'])} experiments")
    else:
        log("[8/10] Skipping color analysis — no extracted renders")

    # -----------------------------------------------------------------------
    # Step 9: Parse dataset input analysis
    # -----------------------------------------------------------------------
    log("[9/10] Parsing dataset input analysis...")
    report["dataset_input_metrics"] = parse_dataset_analysis(args.dataset_analysis)
    if report["dataset_input_metrics"]:
        log(f"  Parsed {len(report['dataset_input_metrics'])} input metrics")
    elif args.dataset_analysis:
        msg = f"Dataset analysis file not found or unreadable: {args.dataset_analysis}"
        warnings.append({"step": "dataset_analysis", "message": msg, "path": args.dataset_analysis})
        log(f"  {msg}")
    else:
        log("  No dataset analysis path provided (--dataset-analysis not set)")

    # -----------------------------------------------------------------------
    # Warnings summary
    # -----------------------------------------------------------------------
    if warnings:
        log(f"\n⚠ {len(warnings)} warning(s) during data gathering:")
        for w in warnings:
            log(f"  [{w['step']}] {w['message']}")

    # -----------------------------------------------------------------------
    # Write report
    # -----------------------------------------------------------------------
    report_path = analysis_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    log(f"\nReport written to {report_path}")

    # Print just the path to stdout for callers to capture
    print(str(report_path))


if __name__ == "__main__":
    main()

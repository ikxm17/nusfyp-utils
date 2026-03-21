"""Batch-render nerfstudio experiments and convert frames to video.

Usage:
    # Config mode — render all experiments defined in experiment_config
    python scripts/render_experiments.py --dry-run

    # Config mode — filter by experiment name substring
    python scripts/render_experiments.py --filter saltpond --dry-run

    # Path mode — render all runs under a method directory
    python scripts/render_experiments.py ../fyp-playground/outputs/saltpond_unprocessed/saltpond_unprocessed-a_exploration/sea-splatfacto

    # Path mode — render a single run by timestamp directory
    python scripts/render_experiments.py ../fyp-playground/outputs/.../2026-03-08_031123

    # Path mode — render multiple path specs
    python scripts/render_experiments.py a_exploration b_exploration --outputs-dir ../fyp-playground/outputs

    # Render camera paths instead of (or in addition to) dataset splits
    python scripts/render_experiments.py a_exploration --render-type camera-path --camera-path ../fyp-playground/datasets/saltpond/camera_paths/1.json
    python scripts/render_experiments.py a_exploration --render-type all --camera-paths-dir ../fyp-playground/datasets/saltpond/camera_paths

    # Skip runs that already have renders
    python scripts/render_experiments.py a_exploration --skip-existing

Wraps render.py to process multiple experiments in one invocation.
Follows the same dual-mode pattern as eval_experiments.py (config mode + path mode).
"""

import argparse
import sys
import time
from pathlib import Path

from eval_experiments import resolve_runs, resolve_runs_from_config, validate_run
from read_config import resolve_outputs_dir

sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))
from run_experiments import load_config

from render import check_prerequisites, render_camera_path, render_dataset


def extract_data_path(config_path):
    """Extract the dataset path from a nerfstudio config.yml."""
    try:
        from read_config import load_config as load_yaml_config

        config = load_yaml_config(config_path)
        data = getattr(config, "data", None)
        if data:
            return Path(data)
    except Exception:
        pass
    return None


def resolve_camera_paths(run_dir, explicit_path=None, explicit_dir=None):
    """Resolve camera path JSON files for a run.

    Resolution order:
        1. Explicit --camera-path: a single file for all runs
        2. Explicit --camera-paths-dir: all *.json in that directory
        3. Auto-discover from config.yml data path:
           datasets/{group}/camera_paths/*.json

    Returns a list of Path objects, or empty list if none found.
    """
    if explicit_path:
        return [Path(explicit_path).expanduser().resolve()]

    if explicit_dir:
        cp_dir = Path(explicit_dir).expanduser().resolve()
        return sorted(cp_dir.glob("*.json"))

    # Auto-discover from config.yml data path
    config_path = run_dir / "config.yml"
    data_path = extract_data_path(config_path)
    if data_path is None:
        return []

    # data_path = .../datasets/saltpond/saltpond_unprocessed
    # camera_paths = .../datasets/saltpond/camera_paths/
    dataset_group_dir = data_path.parent
    cp_dir = dataset_group_dir / "camera_paths"
    if cp_dir.is_dir():
        return sorted(cp_dir.glob("*.json"))

    return []


def check_existing_renders(
    run_dir, render_type, rendered_output_names, splits=None, camera_paths=None
):
    """Check if renders already exist for a run.

    Returns (all_exist, existing, missing) where each list contains
    relative paths from run_dir.
    """
    renders_dir = run_dir / "renders"
    existing = []
    missing = []

    if render_type in ("dataset", "all"):
        all_splits = splits or ["train", "test"]
        for split in all_splits:
            for output_name in rendered_output_names:
                video = renders_dir / "dataset" / split / f"{output_name}.mp4"
                rel = str(video.relative_to(run_dir))
                if video.is_file() and video.stat().st_size > 0:
                    existing.append(rel)
                else:
                    missing.append(rel)

    if render_type in ("camera-path", "all"):
        for cp in camera_paths or []:
            for output_name in rendered_output_names:
                video = renders_dir / "camera-path" / cp.stem / f"{output_name}.mp4"
                rel = str(video.relative_to(run_dir))
                if video.is_file() and video.stat().st_size > 0:
                    existing.append(rel)
                else:
                    missing.append(rel)

    all_exist = len(missing) == 0 and len(existing) > 0
    return all_exist, existing, missing


def make_render_args(
    experiment,
    command,
    outputs_dir,
    rendered_output_names,
    fps,
    keep_frames,
    image_format,
    jpeg_quality,
    downscale_factor,
    dry_run,
    output_dir=None,
    split=None,
    camera_path=None,
    camera_path_name=None,
):
    """Build an argparse.Namespace matching render.py's expected args."""
    return argparse.Namespace(
        experiment=experiment,
        command=command,
        outputs_dir=outputs_dir,
        output_dir=output_dir,
        rendered_output_names=list(rendered_output_names),
        fps=fps,
        keep_frames=keep_frames,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        downscale_factor=downscale_factor,
        dry_run=dry_run,
        split=split,
        camera_path=camera_path,
        camera_path_name=camera_path_name,
    )


def render_single_run(run_dir, render_type, camera_paths, args):
    """Render a single run directory. Returns a result dict."""
    config_path = str(run_dir / "config.yml")
    label = f"{run_dir.parent.name}/{run_dir.name}"
    renders = []
    start = time.time()

    shared = dict(
        experiment=config_path,
        outputs_dir=args.outputs_dir,
        rendered_output_names=args.rendered_output_names,
        fps=args.fps,
        keep_frames=args.keep_frames,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        downscale_factor=args.downscale_factor,
        dry_run=args.dry_run,
    )

    if render_type in ("dataset", "all"):
        render_args = make_render_args(**shared, command="dataset", split=args.split)
        try:
            render_dataset(render_args)
            renders.append(("dataset", "success"))
        except SystemExit:
            renders.append(("dataset", "failed"))
        except Exception as e:
            renders.append(("dataset", f"failed ({e})"))

    if render_type in ("camera-path", "all"):
        for cp in camera_paths:
            render_args = make_render_args(
                **shared,
                command="camera-path",
                camera_path=str(cp),
                camera_path_name=cp.stem,
            )
            try:
                render_camera_path(render_args)
                renders.append((f"camera-path/{cp.stem}", "success"))
            except SystemExit:
                renders.append((f"camera-path/{cp.stem}", "failed"))
            except Exception as e:
                renders.append((f"camera-path/{cp.stem}", f"failed ({e})"))

    duration = time.time() - start
    return {"label": label, "renders": renders, "duration": duration}


def print_summary(results):
    """Print a summary table of rendering results."""
    print(f"\n{'Run':<50} {'Result':<30} {'Duration':>10}")
    print("-" * 90)
    for r in results:
        dur = f"{r['duration']:.1f}s" if r["duration"] > 0 else "—"
        if r.get("renders"):
            summary = ", ".join(
                f"{name}: {status}" for name, status in r["renders"]
            )
        else:
            summary = r.get("status", "—")
        print(f"{r['label']:<50} {summary:<30} {dur:>10}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch-render nerfstudio experiments and convert to video.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Experiment path specs (timestamp dir, method dir, or substring). "
        "If omitted, uses config mode.",
    )
    parser.add_argument(
        "--render-type",
        choices=["dataset", "camera-path", "all"],
        default=None,
        help="Render type(s) to run (default: dataset)",
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
        help="Only render experiments whose name contains this substring (config mode only)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Only render experiments for this dataset (matches dataset prefix in name)",
    )

    # Camera path options
    cp_group = parser.add_argument_group("camera path options")
    cp_group.add_argument(
        "--camera-path",
        default=None,
        dest="camera_path",
        help="Explicit camera path JSON file (used for all runs)",
    )
    cp_group.add_argument(
        "--camera-paths-dir",
        default=None,
        dest="camera_paths_dir",
        help="Directory of camera path JSONs (all *.json rendered per run)",
    )

    # Render options (matching render.py)
    render_group = parser.add_argument_group("render options")
    render_group.add_argument(
        "--rendered-output-names",
        nargs="+",
        default=None,
        help="Output names to render (default: rgb)",
    )
    render_group.add_argument(
        "--split",
        default=None,
        help="Dataset split(s) to render, '+'-separated (default: train+test)",
    )
    render_group.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate (default: 30)",
    )
    render_group.add_argument(
        "--keep-frames",
        action="store_true",
        help="Preserve frame images after video creation",
    )
    render_group.add_argument(
        "--image-format",
        choices=["jpeg", "png"],
        default="jpeg",
        help="Image format for rendered frames (default: jpeg)",
    )
    render_group.add_argument(
        "--jpeg-quality",
        type=int,
        default=100,
        help="JPEG quality 1-100 (default: 100)",
    )
    render_group.add_argument(
        "--downscale-factor",
        type=int,
        default=1,
        help="Resolution downscale factor (default: 1)",
    )

    # Workflow options
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose expected render videos already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be rendered without executing",
    )

    args = parser.parse_args()

    if args.paths and args.filter:
        parser.error("--filter cannot be used with positional paths")

    # Resolve runs and apply config/default values for render settings
    if args.paths:
        outputs_dir = resolve_outputs_dir(args.outputs_dir)
        all_runs = []
        for spec in args.paths:
            runs = resolve_runs(spec, outputs_dir)
            all_runs.extend(runs)

        # Path mode — use hardcoded defaults for unset args
        if args.render_type is None:
            args.render_type = "dataset"
        if args.rendered_output_names is None:
            args.rendered_output_names = ["rgb"]
        if args.split is None:
            args.split = "train+test"
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

        # Config mode — use config values for unset args, fall back to defaults
        if args.render_type is None:
            args.render_type = getattr(config, "RENDER_TYPE", "dataset")
        if args.rendered_output_names is None:
            args.rendered_output_names = getattr(config, "RENDER_OUTPUT_NAMES", ["rgb"])
        if args.split is None:
            args.split = getattr(config, "RENDER_SPLIT", "train+test")

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

    # Resolve outputs_dir for render.py functions
    args.outputs_dir = resolve_outputs_dir(args.outputs_dir)

    # Prerequisites check
    if not args.dry_run:
        need_ffmpeg = args.render_type in ("dataset", "all") or args.keep_frames
        check_prerequisites(need_ffmpeg=need_ffmpeg)

    print(f"Resolved {len(unique_runs)} run(s)")
    print(f"Render type: {args.render_type}")
    print()

    results = []
    for i, run_dir in enumerate(unique_runs):
        label = f"{run_dir.parent.name}/{run_dir.name}"

        # Validate run (config.yml + checkpoints)
        valid, reason = validate_run(run_dir)
        if not valid:
            print(f"[{i + 1}/{len(unique_runs)}] Skipping {label}: {reason}")
            results.append({"label": label, "status": f"skipped ({reason})", "duration": 0})
            continue

        # Resolve camera paths if needed
        camera_paths = []
        if args.render_type in ("camera-path", "all"):
            camera_paths = resolve_camera_paths(
                run_dir, args.camera_path, args.camera_paths_dir
            )
            if not camera_paths:
                print(
                    f"[{i + 1}/{len(unique_runs)}] Warning: no camera paths found for {label}"
                )
                if args.render_type == "camera-path":
                    results.append({
                        "label": label,
                        "status": "skipped (no camera paths)",
                        "duration": 0,
                    })
                    continue

        # Check for existing renders
        if args.skip_existing:
            splits = args.split.split("+") if args.render_type in ("dataset", "all") else None
            all_exist, _, _ = check_existing_renders(
                run_dir, args.render_type, args.rendered_output_names, splits, camera_paths
            )
            if all_exist:
                print(f"[{i + 1}/{len(unique_runs)}] Skipping {label}: renders already exist")
                results.append({"label": label, "status": "skipped (existing)", "duration": 0})
                continue

        # Dry run: show what would be rendered
        if args.dry_run:
            print(f"[{i + 1}/{len(unique_runs)}] {label}")
            if args.render_type in ("dataset", "all"):
                print(f"    dataset: splits={args.split}, outputs={args.rendered_output_names}")
            if args.render_type in ("camera-path", "all") and camera_paths:
                for cp in camera_paths:
                    print(f"    camera-path: {cp.name}, outputs={args.rendered_output_names}")
            print()
            continue

        # Execute render
        print(f"[{i + 1}/{len(unique_runs)}] Rendering {label}")
        result = render_single_run(run_dir, args.render_type, camera_paths, args)
        results.append(result)

        status_parts = []
        for name, status in result["renders"]:
            status_parts.append(f"{name}: {status}")
        print(f"  -> {', '.join(status_parts)} ({result['duration']:.1f}s)")
        print()

    if not args.dry_run and results:
        print_summary(results)


if __name__ == "__main__":
    main()

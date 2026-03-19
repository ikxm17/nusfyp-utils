"""Render nerfstudio experiments and convert frames to video.

Usage:
    python scripts/render.py dataset <experiment> [options]
    python scripts/render.py camera-path <experiment> --camera-path <file> [options]

Wraps ns-render with automatic video creation and cleanup.

Dataset mode renders train/test split images and converts each output to video.
Camera-path mode renders smooth trajectories from camera path JSON files.

Output structure:
    <timestamp>/renders/dataset/{split}/{output_name}.mp4
    <timestamp>/renders/camera-path/{path-name}/{output_name}.mp4
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from read_config import resolve_config_path, resolve_outputs_dir


def natural_sort_key(path):
    """Sort key for natural ordering of filenames containing numbers."""
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", path.name)
    ]


def check_prerequisites(need_ffmpeg=True):
    """Verify required external tools are available."""
    if not shutil.which("ns-render"):
        print("Error: ns-render not found on PATH.", file=sys.stderr)
        sys.exit(1)
    if need_ffmpeg and not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found on PATH.", file=sys.stderr)
        sys.exit(1)


def run_command(cmd, dry_run=False):
    """Run a command with real-time output tee-ing. Returns the exit code."""
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        return 0

    proc = subprocess.Popen(
        cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    for line in proc.stdout:
        sys.stdout.write(line)
    proc.wait()
    return proc.returncode


def frames_to_video(frame_dir, output_file, fps, glob_pattern=None):
    """Convert a directory of image frames to an MP4 video using ffmpeg.

    Two modes:
        - glob_pattern set: use -pattern_type glob (for sequential 00000.jpeg frames)
        - glob_pattern None: use concat demuxer with natural-sorted file list
          (for arbitrary filenames from dataset renders)

    Returns True on success, False on failure. Never deletes frames.
    """
    frame_dir = Path(frame_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    list_file = None

    if glob_pattern:
        frames = list(frame_dir.glob(glob_pattern))
        if not frames:
            print(
                f"  Warning: No frames matching {glob_pattern} in {frame_dir}",
                file=sys.stderr,
            )
            return False

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-pattern_type", "glob",
            "-i", str(frame_dir / glob_pattern),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_file),
        ]
    else:
        image_exts = {".jpg", ".jpeg", ".png"}
        frames = sorted(
            [f for f in frame_dir.iterdir() if f.suffix.lower() in image_exts],
            key=natural_sort_key,
        )
        if not frames:
            print(
                f"  Warning: No image frames found in {frame_dir}", file=sys.stderr
            )
            return False

        list_file = frame_dir / "_filelist.txt"
        with open(list_file, "w") as f:
            for frame in frames:
                safe_name = str(frame.name).replace("'", "'\\''")
                f.write(f"file '{safe_name}'\n")
                f.write(f"duration {1 / fps}\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_file),
        ]

    print(f"  Creating video: {output_file.name} ({len(frames)} frames, {fps} fps)")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if list_file and list_file.exists():
        list_file.unlink()

    if result.returncode != 0:
        print(f"  Error: ffmpeg failed:\n{result.stderr}", file=sys.stderr)
        return False

    # Validate that the MP4 file was actually created with content.
    # ffmpeg can exit 0 but produce a zero-byte file (e.g. codec/resolution
    # incompatibility).  Returning False here prevents the caller from
    # deleting the source frames, preserving them for debugging.
    if not output_file.is_file() or output_file.stat().st_size == 0:
        print(
            f"  Error: ffmpeg produced empty or missing file: {output_file}",
            file=sys.stderr,
        )
        if output_file.is_file():
            output_file.unlink()
        return False

    print(f"  Saved: {output_file}")
    return True


def render_dataset(args):
    """Render dataset splits and convert frames to video.

    Supports a "combined" pseudo-split: when the split string includes "combined"
    (e.g. "train+test+combined"), individual splits are rendered normally, then
    their frames are concatenated into a single combined video per output.
    """
    config_path = resolve_config_path(args.experiment, args.outputs_dir)
    timestamp_dir = config_path.parent

    all_splits = args.split.split("+")
    render_splits = [s for s in all_splits if s != "combined"]
    do_combined = "combined" in all_splits

    base_output_dir = args.output_dir or (timestamp_dir / "renders" / "dataset")

    print(f"Config:  {config_path}")
    print(f"Output:  {base_output_dir}")
    print(f"Splits:  {', '.join(all_splits)}")
    print(f"Outputs: {', '.join(args.rendered_output_names)}")
    print()

    # When combining, we must keep frames through the per-split loop
    keep_frames_original = args.keep_frames
    if do_combined:
        args.keep_frames = True

    for split in render_splits:
        split_output_dir = base_output_dir / split
        print(f"--- Split: {split} ---")

        # ns-render dataset always outputs frames; we convert to video afterwards
        # Note: ns-render creates {output_path}/{split}/ internally,
        # so we pass base_output_dir (not split_output_dir) to avoid double nesting.
        cmd = [
            "ns-render", "dataset",
            "--load-config", str(config_path),
            "--output-path", str(base_output_dir),
            "--rendered-output-names",
        ] + args.rendered_output_names + [
            "--split", split,
            "--image-format", args.image_format,
            "--jpeg-quality", str(args.jpeg_quality),
            "--downscale-factor", str(args.downscale_factor),
        ]

        rc = run_command(cmd, dry_run=args.dry_run)
        if args.dry_run:
            print()
            continue
        if rc != 0:
            print(
                f"  ns-render failed for split '{split}' (exit code {rc})",
                file=sys.stderr,
            )
            continue

        # Convert each rendered output's frame directory to video
        for output_name in args.rendered_output_names:
            frame_dir = split_output_dir / output_name
            if not frame_dir.is_dir():
                print(f"  Warning: Expected frame directory not found: {frame_dir}")
                continue

            video_file = split_output_dir / f"{output_name}.mp4"
            success = frames_to_video(frame_dir, video_file, args.fps)

            # Only remove frames if video was created and we don't need them
            if success and not args.keep_frames:
                shutil.rmtree(frame_dir)
                print(f"  Removed frames: {frame_dir.name}/")

        print()

    # Restore original keep_frames setting
    args.keep_frames = keep_frames_original

    # Combined pseudo-split: concatenate frames from all rendered splits
    if do_combined and len(render_splits) > 1 and not args.dry_run:
        combined_output_dir = base_output_dir / "combined"
        combined_all_ok = True
        print("--- Combined ---")

        for output_name in args.rendered_output_names:
            # Gather frames from all splits in order
            all_frames = []
            for split in render_splits:
                frame_dir = base_output_dir / split / output_name
                if not frame_dir.is_dir():
                    continue
                image_exts = {".jpg", ".jpeg", ".png"}
                frames = sorted(
                    [f for f in frame_dir.iterdir() if f.suffix.lower() in image_exts],
                    key=natural_sort_key,
                )
                all_frames.extend(frames)

            if not all_frames:
                print(f"  Warning: No frames found for combined {output_name}")
                continue

            # Write a concat filelist referencing frames from each split dir
            combined_output_dir.mkdir(parents=True, exist_ok=True)
            list_file = combined_output_dir / f"_{output_name}_filelist.txt"
            with open(list_file, "w") as f:
                for frame in all_frames:
                    safe_path = str(frame.resolve()).replace("'", "'\\''")
                    f.write(f"file '{safe_path}'\n")
                    f.write(f"duration {1 / args.fps}\n")

            video_file = combined_output_dir / f"{output_name}.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(list_file),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(video_file),
            ]

            print(f"  Creating video: {output_name}.mp4 ({len(all_frames)} frames, {args.fps} fps)")
            result = subprocess.run(cmd, capture_output=True, text=True)
            list_file.unlink()

            if result.returncode != 0:
                print(f"  Error: ffmpeg failed:\n{result.stderr}", file=sys.stderr)
                combined_all_ok = False
            elif not video_file.is_file() or video_file.stat().st_size == 0:
                print(
                    f"  Error: ffmpeg produced empty or missing file: {video_file}",
                    file=sys.stderr,
                )
                if video_file.is_file():
                    video_file.unlink()
                combined_all_ok = False
            else:
                print(f"  Saved: {video_file}")

        # Clean up per-split frames only if ALL combined videos were created
        # successfully.  If any failed, keep frames for debugging/re-encoding.
        if not args.keep_frames and combined_all_ok:
            for split in render_splits:
                for output_name in args.rendered_output_names:
                    frame_dir = base_output_dir / split / output_name
                    if frame_dir.is_dir():
                        shutil.rmtree(frame_dir)
                        print(f"  Removed frames: {split}/{output_name}/")

        print()
    elif do_combined and args.dry_run:
        print("--- Combined ---")
        print(f"    Will combine frames from {'+'.join(render_splits)} into combined/ videos")
        print(f"    Outputs: {args.rendered_output_names}")
        print()


def render_camera_path(args):
    """Render camera path trajectory with per-output videos."""
    config_path = resolve_config_path(args.experiment, args.outputs_dir)
    timestamp_dir = config_path.parent

    camera_path = Path(args.camera_path).expanduser().resolve()
    if not args.dry_run and not camera_path.is_file():
        print(f"Error: Camera path file not found: {camera_path}", file=sys.stderr)
        sys.exit(1)

    path_name = args.camera_path_name or camera_path.stem
    base_output_dir = args.output_dir or (
        timestamp_dir / "renders" / "camera-path" / path_name
    )

    print(f"Config:      {config_path}")
    print(f"Camera path: {camera_path}")
    print(f"Output:      {base_output_dir}")
    print(f"Outputs:     {', '.join(args.rendered_output_names)}")
    print()

    # Invoke ns-render once per output name to get separate videos
    # (ns-render concatenates multiple outputs side-by-side otherwise)
    for output_name in args.rendered_output_names:
        print(f"--- Output: {output_name} ---")

        if args.keep_frames:
            # Render to images, then also create video
            frame_dir = base_output_dir / output_name
            cmd = [
                "ns-render", "camera-path",
                "--load-config", str(config_path),
                "--camera-path-filename", str(camera_path),
                "--output-path", str(frame_dir),
                "--rendered-output-names", output_name,
                "--output-format", "images",
                "--image-format", args.image_format,
                "--jpeg-quality", str(args.jpeg_quality),
                "--downscale-factor", str(args.downscale_factor),
            ]

            rc = run_command(cmd, dry_run=args.dry_run)
            if args.dry_run:
                print()
                continue
            if rc != 0:
                print(
                    f"  ns-render failed for '{output_name}' (exit code {rc})",
                    file=sys.stderr,
                )
                continue

            video_file = base_output_dir / f"{output_name}.mp4"
            glob_pattern = f"*.{args.image_format}"
            frames_to_video(frame_dir, video_file, args.fps, glob_pattern=glob_pattern)
        else:
            # Render directly to video — no frames, no ffmpeg needed
            video_file = base_output_dir / f"{output_name}.mp4"
            video_file.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                "ns-render", "camera-path",
                "--load-config", str(config_path),
                "--camera-path-filename", str(camera_path),
                "--output-path", str(video_file),
                "--rendered-output-names", output_name,
                "--output-format", "video",
                "--downscale-factor", str(args.downscale_factor),
            ]

            rc = run_command(cmd, dry_run=args.dry_run)
            if not args.dry_run and rc != 0:
                print(
                    f"  ns-render failed for '{output_name}' (exit code {rc})",
                    file=sys.stderr,
                )

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Render nerfstudio experiments and convert to video.",
    )

    # Shared arguments via parent parser
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "experiment",
        help="Path/spec to experiment (resolved via read_config.py)",
    )
    shared.add_argument(
        "--outputs-dir",
        default=None,
        help="Base outputs directory (default: $NERFSTUDIO_OUTPUTS or ./outputs)",
    )
    shared.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: <experiment>/renders/<subcommand>/)",
    )
    shared.add_argument(
        "--rendered-output-names",
        nargs="+",
        default=["rgb"],
        help="Output names to render (default: rgb)",
    )
    shared.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate (default: 30)",
    )
    shared.add_argument(
        "--keep-frames",
        action="store_true",
        help="Preserve frame images after video creation",
    )
    shared.add_argument(
        "--image-format",
        choices=["jpeg", "png"],
        default="jpeg",
        help="Image format for rendered frames (default: jpeg)",
    )
    shared.add_argument(
        "--jpeg-quality",
        type=int,
        default=100,
        help="JPEG quality 1-100 (default: 100)",
    )
    shared.add_argument(
        "--downscale-factor",
        type=int,
        default=1,
        help="Resolution downscale factor (default: 1)",
    )
    shared.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # dataset subcommand
    subparsers.add_parser(
        "dataset",
        parents=[shared],
        help="Render dataset train/test splits",
    ).add_argument(
        "--split",
        default="train+test",
        help="Split(s) to render, '+'-separated (default: train+test)",
    )

    # camera-path subcommand
    cp_parser = subparsers.add_parser(
        "camera-path",
        parents=[shared],
        help="Render camera path trajectory",
    )
    cp_parser.add_argument(
        "--camera-path",
        required=True,
        dest="camera_path",
        help="Path to camera path JSON file",
    )
    cp_parser.add_argument(
        "--camera-path-name",
        default=None,
        dest="camera_path_name",
        help="Name for output directory (default: derived from filename)",
    )

    args = parser.parse_args()
    args.outputs_dir = resolve_outputs_dir(args.outputs_dir)

    if not args.dry_run:
        need_ffmpeg = args.command == "dataset" or args.keep_frames
        check_prerequisites(need_ffmpeg=need_ffmpeg)

    if args.command == "dataset":
        render_dataset(args)
    elif args.command == "camera-path":
        render_camera_path(args)


if __name__ == "__main__":
    main()

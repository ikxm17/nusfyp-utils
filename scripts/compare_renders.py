"""Compare rendered outputs across nerfstudio experiments.

Usage:
    python scripts/compare_renders.py info <experiments...> [options]
    python scripts/compare_renders.py extract <experiments...> --frames N [options]
    python scripts/compare_renders.py compare <experiments...> --frames N [options]
    python scripts/compare_renders.py grid <experiments...> --frames N [options]

Subcommands:
    info      Show frame counts, resolution, and available output types
    extract   Extract specific frames from MP4 renders as PNGs
    compare   Cross-experiment comparison strips (same frame, same output type)
    grid      Matrix view: experiments (rows) x output types (columns) per frame

Path resolution uses read_config.py — substring matching works (e.g. "seathru8k").
"""

import argparse
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from read_config import resolve_config_path, resolve_outputs_dir


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_render_dir(spec, outputs_dir, render_type, split, camera_path_name="1"):
    """Resolve experiment spec to renders directory.

    Returns (render_dir, config_path) where render_dir contains MP4 files.
    """
    config_path = resolve_config_path(spec, outputs_dir)
    timestamp_dir = config_path.parent

    if render_type == "dataset":
        render_dir = timestamp_dir / "renders" / "dataset" / split
    else:
        render_dir = timestamp_dir / "renders" / "camera-path" / camera_path_name

    return render_dir, config_path


def derive_short_name(config_path, outputs_dir):
    """Extract short label like 'tune01_seathru8k' from full config path."""
    try:
        rel = Path(config_path).relative_to(outputs_dir)
        parts = rel.parts
        # Expected: dataset_name / experiment_name / method / timestamp / config.yml
        if len(parts) >= 2:
            # experiment_name is like "saltpond_unprocessed-tune01_seathru8k"
            experiment = parts[1]
            # Strip dataset prefix (everything up to and including the first hyphen
            # that separates dataset from experiment name)
            # e.g. "saltpond_unprocessed-tune01_seathru8k" -> "tune01_seathru8k"
            dataset = parts[0]
            prefix = dataset + "-"
            if experiment.startswith(prefix):
                return experiment[len(prefix):]
            return experiment
        return str(rel)
    except ValueError:
        return str(config_path.parent.name)


# ---------------------------------------------------------------------------
# Video / frame operations
# ---------------------------------------------------------------------------


def get_video_info(video_path):
    """Return (frame_count, width, height) for a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0, 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return count, width, height


def extract_frames(video_path, frame_indices):
    """Extract specific frames from video, return dict[int, PIL.Image]."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error: Cannot open {video_path}", file=sys.stderr)
        return {}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = {}

    for idx in sorted(frame_indices):
        if idx < 0 or idx >= total:
            print(
                f"  Warning: Frame {idx} out of range (0-{total - 1}) in {video_path.name}",
                file=sys.stderr,
            )
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results[idx] = Image.fromarray(frame_rgb)
        else:
            print(
                f"  Warning: Failed to read frame {idx} from {video_path.name}",
                file=sys.stderr,
            )

    cap.release()
    return results


# ---------------------------------------------------------------------------
# Image composition
# ---------------------------------------------------------------------------


def add_label(image, text, bar_height=32, font_size=20):
    """Add a text label bar above the image. Returns new image."""
    w, h = image.size
    labeled = Image.new("RGB", (w, h + bar_height), (30, 30, 30))
    labeled.paste(image, (0, bar_height))

    draw = ImageDraw.Draw(labeled)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (w - tw) // 2
    y = (bar_height - th) // 2
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    return labeled


def maybe_downscale(image, max_width):
    """Downscale image if wider than max_width, preserving aspect ratio."""
    if max_width is None or image.width <= max_width:
        return image
    ratio = max_width / image.width
    new_size = (max_width, int(image.height * ratio))
    return image.resize(new_size, Image.LANCZOS)


def compose_strip(images, labels, direction="vertical"):
    """Stack images vertically or horizontally with labels."""
    labeled = [add_label(img, lbl) for img, lbl in zip(images, labels)]

    if direction == "vertical":
        max_w = max(img.width for img in labeled)
        total_h = sum(img.height for img in labeled)
        result = Image.new("RGB", (max_w, total_h), (0, 0, 0))
        y = 0
        for img in labeled:
            result.paste(img, (0, y))
            y += img.height
    else:
        total_w = sum(img.width for img in labeled)
        max_h = max(img.height for img in labeled)
        result = Image.new("RGB", (total_w, max_h), (0, 0, 0))
        x = 0
        for img in labeled:
            result.paste(img, (x, 0))
            x += img.width

    return result


def compose_grid(images_2d, row_labels, col_labels, bar_height=32, font_size=20):
    """Build experiment (rows) x output_type (columns) grid with headers.

    images_2d: list of lists, images_2d[row][col] is a PIL.Image or None.
    """
    if not images_2d or not images_2d[0]:
        return None

    # Find cell dimensions from first non-None image
    cell_w, cell_h = 0, 0
    for row in images_2d:
        for img in row:
            if img is not None:
                cell_w, cell_h = img.size
                break
        if cell_w:
            break

    if not cell_w:
        return None

    n_rows = len(images_2d)
    n_cols = len(images_2d[0])

    # Layout: row labels on left, column labels on top
    label_col_width = 200  # width for row labels
    total_w = label_col_width + n_cols * cell_w
    total_h = bar_height + n_rows * (cell_h + bar_height)

    result = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    draw = ImageDraw.Draw(result)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Column headers
    for c, col_label in enumerate(col_labels):
        x = label_col_width + c * cell_w
        bbox = draw.textbbox((0, 0), col_label, font=font)
        tw = bbox[2] - bbox[0]
        tx = x + (cell_w - tw) // 2
        ty = (bar_height - (bbox[3] - bbox[1])) // 2
        draw.text((tx, ty), col_label, fill=(255, 255, 255), font=font)

    # Rows
    for r, row_label in enumerate(row_labels):
        y = bar_height + r * (cell_h + bar_height)

        # Row label
        bbox = draw.textbbox((0, 0), row_label, font=font)
        tw = bbox[2] - bbox[0]
        tx = (label_col_width - tw) // 2
        ty = y + bar_height + (cell_h - (bbox[3] - bbox[1])) // 2
        draw.text((tx, ty), row_label, fill=(255, 255, 255), font=font)

        # Row header bar label (small, above row images)
        # Already handled by row label on left

        for c in range(n_cols):
            x = label_col_width + c * cell_w
            img = images_2d[r][c]
            if img is not None:
                result.paste(img, (x, y + bar_height))

    return result


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_info(args):
    """Show frame counts, resolution, available output types for experiments."""
    outputs_dir = resolve_outputs_dir(args.outputs_dir)

    for spec in args.experiments:
        render_dir, config_path = resolve_render_dir(
            spec, outputs_dir, args.render_type, args.split, args.camera_path_name
        )
        name = derive_short_name(config_path, outputs_dir)

        print(f"\n{'=' * 60}")
        print(f"Experiment: {name}")
        print(f"Render dir: {render_dir}")

        if not render_dir.is_dir():
            print("  No renders found.")
            continue

        videos = sorted(render_dir.glob("*.mp4"))
        if not videos:
            print("  No MP4 files found.")
            continue

        print(f"  {'Output':<25} {'Frames':>8} {'Resolution':>14}")
        print(f"  {'-' * 25} {'-' * 8} {'-' * 14}")
        for v in videos:
            count, w, h = get_video_info(v)
            print(f"  {v.stem:<25} {count:>8} {w:>6}x{h:<6}")


def cmd_extract(args):
    """Extract specific frames from render videos as PNGs."""
    outputs_dir = resolve_outputs_dir(args.outputs_dir)
    output_dir = Path(args.output_dir)

    for spec in args.experiments:
        render_dir, config_path = resolve_render_dir(
            spec, outputs_dir, args.render_type, args.split, args.camera_path_name
        )
        name = derive_short_name(config_path, outputs_dir)
        print(f"\nExtracting from: {name}")

        for output_type in args.output_types:
            video_path = render_dir / f"{output_type}.mp4"
            if not video_path.is_file():
                print(f"  Warning: {video_path} not found, skipping", file=sys.stderr)
                continue

            frames = extract_frames(video_path, args.frames)
            for idx, img in frames.items():
                img = maybe_downscale(img, args.max_width)
                save_dir = output_dir / "extract" / name
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{output_type}_frame{idx:03d}.png"
                img.save(save_path)
                print(f"  Saved: {save_path}")


def cmd_compare(args):
    """Cross-experiment comparison: same frame + output type, stacked vertically."""
    outputs_dir = resolve_outputs_dir(args.outputs_dir)
    output_dir = Path(args.output_dir)

    # Resolve all experiments first
    experiments = []
    for spec in args.experiments:
        render_dir, config_path = resolve_render_dir(
            spec, outputs_dir, args.render_type, args.split, args.camera_path_name
        )
        name = derive_short_name(config_path, outputs_dir)
        experiments.append((name, render_dir))

    for output_type in args.output_types:
        for frame_idx in args.frames:
            images = []
            labels = []

            for name, render_dir in experiments:
                video_path = render_dir / f"{output_type}.mp4"
                if not video_path.is_file():
                    print(
                        f"  Warning: {output_type}.mp4 not found for {name}",
                        file=sys.stderr,
                    )
                    continue

                frames = extract_frames(video_path, [frame_idx])
                if frame_idx in frames:
                    img = maybe_downscale(frames[frame_idx], args.max_width)
                    images.append(img)
                    labels.append(name)

            if len(images) < 2:
                print(
                    f"  Skipping {output_type} frame {frame_idx}: need at least 2 experiments",
                    file=sys.stderr,
                )
                continue

            strip = compose_strip(images, labels, direction="vertical")
            save_dir = output_dir / "compare"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"compare_{output_type}_frame{frame_idx:03d}.png"
            strip.save(save_path)
            print(f"Saved: {save_path}")


def cmd_grid(args):
    """Matrix view: experiments (rows) x output types (columns) per frame."""
    outputs_dir = resolve_outputs_dir(args.outputs_dir)
    output_dir = Path(args.output_dir)

    # Resolve all experiments
    experiments = []
    for spec in args.experiments:
        render_dir, config_path = resolve_render_dir(
            spec, outputs_dir, args.render_type, args.split, args.camera_path_name
        )
        name = derive_short_name(config_path, outputs_dir)
        experiments.append((name, render_dir))

    for frame_idx in args.frames:
        images_2d = []
        row_labels = []

        for name, render_dir in experiments:
            row = []
            for output_type in args.output_types:
                video_path = render_dir / f"{output_type}.mp4"
                if not video_path.is_file():
                    row.append(None)
                    continue

                frames = extract_frames(video_path, [frame_idx])
                if frame_idx in frames:
                    img = maybe_downscale(frames[frame_idx], args.max_width)
                    row.append(img)
                else:
                    row.append(None)

            images_2d.append(row)
            row_labels.append(name)

        grid = compose_grid(images_2d, row_labels, args.output_types)
        if grid is None:
            print(f"  Skipping frame {frame_idx}: no images extracted", file=sys.stderr)
            continue

        save_dir = output_dir / "grid"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"grid_frame{frame_idx:03d}.png"
        grid.save(save_path)
        print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare rendered outputs across nerfstudio experiments.",
    )

    # Shared arguments
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "experiments",
        nargs="+",
        help="Experiment specs (substring matching via read_config.py)",
    )
    shared.add_argument(
        "--outputs-dir",
        default=None,
        help="Base outputs directory (default: $NERFSTUDIO_OUTPUTS or ./outputs)",
    )
    shared.add_argument(
        "--split",
        default="test",
        help="Dataset split (default: test)",
    )
    shared.add_argument(
        "--render-type",
        default="dataset",
        choices=["dataset", "camera-path"],
        help="Render type (default: dataset)",
    )
    shared.add_argument(
        "--camera-path-name",
        default="1",
        help="Camera path name for camera-path mode (default: 1)",
    )
    shared.add_argument(
        "--output-types",
        nargs="+",
        default=["rgb", "underwater_rgb"],
        help="Output types to process (default: rgb underwater_rgb)",
    )
    shared.add_argument(
        "--output-dir",
        default="./comparisons",
        help="Directory for saved results (default: ./comparisons)",
    )
    shared.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="Max image width before downscaling (default: no limit)",
    )

    # Frame argument — separate parser for subcommands that need it
    frame_shared = argparse.ArgumentParser(add_help=False)
    frame_shared.add_argument(
        "--frames",
        nargs="+",
        type=int,
        default=[0],
        help="Frame indices to extract (default: 0)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "info",
        parents=[shared],
        help="Show render metadata (frame counts, resolution, output types)",
    )

    subparsers.add_parser(
        "extract",
        parents=[shared, frame_shared],
        help="Extract specific frames as PNGs",
    )

    subparsers.add_parser(
        "compare",
        parents=[shared, frame_shared],
        help="Cross-experiment comparison strips",
    )

    subparsers.add_parser(
        "grid",
        parents=[shared, frame_shared],
        help="Experiment x output type matrix view",
    )

    args = parser.parse_args()

    if args.command == "info":
        cmd_info(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "grid":
        cmd_grid(args)


if __name__ == "__main__":
    main()

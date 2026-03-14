"""
Depth range statistics from COLMAP sparse reconstructions.

Computes per-camera and global depth statistics from either COLMAP binary
files (images.bin + points3D.bin) or nerfstudio transforms.json, so that
SeaSplat medium parameters can be scaled for each dataset's geometry.

Usage:
    python scripts/dataset_depth.py <colmap_sparse_dir>
    python scripts/dataset_depth.py <transforms.json>
    python scripts/dataset_depth.py <dataset_dir>
    python scripts/dataset_depth.py <path> --sort far --bins 30 -o report.txt
"""

import argparse
import json
import os
import struct
import sys

import numpy as np


# ---------------------------------------------------------------------------
# COLMAP binary readers (adapted from seasplat/utils/colmap_utils.py)
# ---------------------------------------------------------------------------

def _read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def _qvec2rotmat(qvec):
    return np.array([
        [1 - 2*qvec[2]**2 - 2*qvec[3]**2,
         2*qvec[1]*qvec[2] - 2*qvec[0]*qvec[3],
         2*qvec[3]*qvec[1] + 2*qvec[0]*qvec[2]],
        [2*qvec[1]*qvec[2] + 2*qvec[0]*qvec[3],
         1 - 2*qvec[1]**2 - 2*qvec[3]**2,
         2*qvec[2]*qvec[3] - 2*qvec[0]*qvec[1]],
        [2*qvec[3]*qvec[1] - 2*qvec[0]*qvec[2],
         2*qvec[2]*qvec[3] + 2*qvec[0]*qvec[1],
         1 - 2*qvec[1]**2 - 2*qvec[2]**2],
    ])


def _read_images_binary(path):
    """Read COLMAP images.bin → dict of {id: (name, qvec, tvec, point3D_ids)}."""
    images = {}
    with open(path, "rb") as f:
        n = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(n):
            props = _read_next_bytes(f, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            # Read null-terminated name
            name = b""
            c = _read_next_bytes(f, 1, "c")[0]
            while c != b"\x00":
                name += c
                c = _read_next_bytes(f, 1, "c")[0]
            name = name.decode("utf-8")
            # Read 2D points
            n2d = _read_next_bytes(f, 8, "Q")[0]
            if n2d > 0:
                data = _read_next_bytes(f, 24 * n2d, "ddq" * n2d)
                point3D_ids = np.array(data[2::3], dtype=np.int64)
            else:
                point3D_ids = np.array([], dtype=np.int64)
            images[image_id] = (name, qvec, tvec, point3D_ids)
    return images


def _read_points3D_binary(path):
    """Read COLMAP points3D.bin → dict of {id: xyz}."""
    points = {}
    with open(path, "rb") as f:
        n = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(n):
            props = _read_next_bytes(f, 43, "QdddBBBd")
            point_id = props[0]
            xyz = np.array(props[1:4])
            track_len = _read_next_bytes(f, 8, "Q")[0]
            if track_len > 0:
                _read_next_bytes(f, 8 * track_len, "ii" * track_len)
            points[point_id] = xyz
    return points


# ---------------------------------------------------------------------------
# ASCII PLY reader
# ---------------------------------------------------------------------------

def _read_ply_ascii(path):
    """Read ASCII PLY with float x, y, z properties → (N, 3) array."""
    vertices = []
    in_header = True
    n_verts = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if in_header:
                if line.startswith("element vertex"):
                    n_verts = int(line.split()[-1])
                elif line == "end_header":
                    in_header = False
                continue
            parts = line.split()
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if len(vertices) >= n_verts:
                break
    return np.array(vertices)


# ---------------------------------------------------------------------------
# Input auto-detection
# ---------------------------------------------------------------------------

def _detect_input(path):
    """Detect input mode from path. Returns (mode, base_dir).

    Modes: 'colmap', 'transforms'
    """
    path = os.path.abspath(path)

    # Priority 1: path is/contains images.bin + points3D.bin
    if os.path.isfile(path):
        d = os.path.dirname(path)
    else:
        d = path

    images_bin = os.path.join(d, "images.bin")
    points_bin = os.path.join(d, "points3D.bin")
    if os.path.isfile(images_bin) and os.path.isfile(points_bin):
        return "colmap", d

    # Priority 2: path is/contains transforms.json
    if os.path.isfile(path) and os.path.basename(path) == "transforms.json":
        return "transforms", os.path.dirname(path)
    tj = os.path.join(d, "transforms.json")
    if os.path.isfile(tj):
        # But check for colmap subdirectory first (priority 3)
        pass

    # Priority 3: directory with colmap/sparse/0/ subdirectory
    colmap_dir = os.path.join(d, "colmap", "sparse", "0")
    if os.path.isdir(colmap_dir):
        images_bin = os.path.join(colmap_dir, "images.bin")
        points_bin = os.path.join(colmap_dir, "points3D.bin")
        if os.path.isfile(images_bin) and os.path.isfile(points_bin):
            return "colmap", colmap_dir

    # Priority 4: directory with transforms.json
    if os.path.isfile(tj):
        return "transforms", d

    return None, path


# ---------------------------------------------------------------------------
# Depth computation
# ---------------------------------------------------------------------------

def _compute_colmap_depths(base_dir):
    """Compute per-camera depths using COLMAP binary files with track info."""
    images = _read_images_binary(os.path.join(base_dir, "images.bin"))
    points3D = _read_points3D_binary(os.path.join(base_dir, "points3D.bin"))

    print(f"  Loaded {len(images)} images, {len(points3D)} 3D points", file=sys.stderr)

    results = []
    for i, (img_id, (name, qvec, tvec, pt_ids)) in enumerate(sorted(images.items())):
        print(f"\r  Processing camera {i + 1}/{len(images)}...", end="", file=sys.stderr)
        R = _qvec2rotmat(qvec)
        cam_pos = -R.T @ tvec

        # Filter to valid point IDs (>= 0 means tracked)
        valid_ids = pt_ids[pt_ids >= 0]
        if len(valid_ids) == 0:
            results.append((name, cam_pos, np.array([]), 0))
            continue

        # Look up XYZ for each tracked point
        pts = []
        for pid in valid_ids:
            if pid in points3D:
                pts.append(points3D[pid])
        if not pts:
            results.append((name, cam_pos, np.array([]), 0))
            continue

        pts = np.array(pts)
        dists = np.linalg.norm(pts - cam_pos, axis=1)
        results.append((name, cam_pos, dists, len(pts)))

    print(file=sys.stderr)
    return results, len(images), len(points3D), "colmap"


def _compute_transforms_depths(base_dir):
    """Compute per-camera depths using transforms.json (approximate, no tracks)."""
    tj_path = os.path.join(base_dir, "transforms.json")
    with open(tj_path) as f:
        data = json.load(f)

    # Load points from PLY
    ply_name = data.get("ply_file_path", "sparse_pc.ply")
    ply_path = os.path.join(base_dir, ply_name)
    if not os.path.isfile(ply_path):
        print(f"Error: PLY file not found: {ply_path}", file=sys.stderr)
        sys.exit(1)

    print(f"  Loading point cloud from {ply_name}...", file=sys.stderr)
    points = _read_ply_ascii(ply_path)
    print(f"  Loaded {len(points)} points", file=sys.stderr)

    frames = data.get("frames", [])
    print(f"  Processing {len(frames)} cameras...", file=sys.stderr)

    results = []
    for i, frame in enumerate(frames):
        print(f"\r  Processing camera {i + 1}/{len(frames)}...", end="", file=sys.stderr)
        mat = np.array(frame["transform_matrix"])
        cam_pos = mat[:3, 3]
        name = os.path.basename(frame.get("file_path", f"frame_{i:05d}"))

        # Distance to ALL points (no track info)
        dists = np.linalg.norm(points - cam_pos, axis=1)
        results.append((name, cam_pos, dists, len(points)))

    print(file=sys.stderr)
    return results, len(frames), len(points), "transforms"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _compute_stats(results):
    """Compute global and per-camera statistics."""
    all_depths = []
    cam_stats = []

    for name, cam_pos, dists, n_pts in results:
        if len(dists) == 0:
            cam_stats.append((name, np.nan, np.nan, np.nan, n_pts, 0.0, []))
            continue
        near = float(dists.min())
        far = float(dists.max())
        med = float(np.median(dists))
        rng = far - near
        cam_stats.append((name, near, far, med, n_pts, rng, []))
        all_depths.append(dists)

    if not all_depths:
        return None, cam_stats

    all_depths = np.concatenate(all_depths)

    percentiles = {p: float(np.percentile(all_depths, p))
                   for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]}

    global_stats = {
        "min": float(all_depths.min()),
        "max": float(all_depths.max()),
        "mean": float(all_depths.mean()),
        "median": float(np.median(all_depths)),
        "std": float(all_depths.std()),
        "iqr": percentiles[75] - percentiles[25],
        "dynamic_range": float(all_depths.max() / all_depths.min()) if all_depths.min() > 0 else float("inf"),
        "percentiles": percentiles,
        "count": len(all_depths),
    }

    # Flag per-camera entries
    p5 = percentiles[5]
    p95 = percentiles[95]
    ranges = [s[5] for s in cam_stats if not np.isnan(s[1])]
    median_range = float(np.median(ranges)) if ranges else 0.0

    flagged_stats = []
    for name, near, far, med, n_pts, rng, _ in cam_stats:
        flags = []
        if not np.isnan(near):
            if near < p5:
                flags.append("SHALLOW")
            if far > p95:
                flags.append("DEEP")
            if median_range > 0:
                if rng < 0.5 * median_range:
                    flags.append("NARROW")
                if rng > 2.0 * median_range:
                    flags.append("WIDE")
        flagged_stats.append((name, near, far, med, n_pts, rng, flags))

    return global_stats, flagged_stats


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

def _text_histogram(all_depths, n_bins, width=50):
    """Build a text histogram of depth values."""
    lo, hi = all_depths.min(), all_depths.max()
    counts, edges = np.histogram(all_depths, bins=n_bins, range=(lo, hi))
    max_count = counts.max() if counts.max() > 0 else 1

    lines = []
    for i in range(n_bins):
        bar_len = int(width * counts[i] / max_count)
        bar = "█" * bar_len
        pct = 100.0 * counts[i] / len(all_depths)
        lines.append(f"  {edges[i]:8.3f} - {edges[i+1]:8.3f} | {bar:<{width}s} {counts[i]:>6d} ({pct:5.1f}%)")
    return lines


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _build_report(mode, n_cameras, n_points, global_stats, cam_stats,
                  all_depths, args):
    lines = []
    lines.append("Depth Range Report")
    lines.append("==================")
    lines.append(f"Input mode:  {mode}" + (" (track-based)" if mode == "colmap" else " (approximate — no per-camera tracks)"))
    lines.append(f"Cameras:     {n_cameras}")
    lines.append(f"3D points:   {n_points:,}")
    lines.append("")

    if global_stats is None:
        lines.append("No depth data available — no cameras have tracked points.")
        return "\n".join(lines) + "\n"

    # Global depth stats
    lines.append("Global Depth Statistics")
    lines.append("-----------------------")
    gs = global_stats
    lines.append(f"  Min:              {gs['min']:.4f}")
    lines.append(f"  Max:              {gs['max']:.4f}")
    lines.append(f"  Mean:             {gs['mean']:.4f}")
    lines.append(f"  Median:           {gs['median']:.4f}")
    lines.append(f"  Std:              {gs['std']:.4f}")
    lines.append(f"  IQR:              {gs['iqr']:.4f}")
    lines.append(f"  Dynamic range:    {gs['dynamic_range']:.2f}x")
    lines.append(f"  Total distances:  {gs['count']:,}")
    lines.append("")

    # Percentiles
    lines.append("  Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        lines.append(f"    p{p:<3d} {gs['percentiles'][p]:.4f}")
    lines.append("")

    # Histogram
    lines.append("Depth Distribution")
    lines.append("------------------")
    lines.extend(_text_histogram(all_depths, args.bins))
    lines.append("")

    # Per-camera table
    if not args.no_per_camera:
        lines.append("Per-Camera Depths")
        lines.append("-----------------")
        header = f"{'Camera':28s} {'Near':>10s} {'Far':>10s} {'Median':>10s} {'Range':>10s} {'Points':>8s}  Flags"
        lines.append(header)

        # Sort
        sort_keys = {
            "name": lambda s: s[0],
            "near": lambda s: (s[1] if not np.isnan(s[1]) else float("inf")),
            "far": lambda s: (s[2] if not np.isnan(s[2]) else float("inf")),
            "median": lambda s: (s[3] if not np.isnan(s[3]) else float("inf")),
            "range": lambda s: (s[5] if not np.isnan(s[5]) else float("inf")),
        }
        sorted_stats = sorted(cam_stats, key=sort_keys[args.sort])

        for name, near, far, med, n_pts, rng, flags in sorted_stats:
            if np.isnan(near):
                lines.append(f"{name:28s} {'—':>10s} {'—':>10s} {'—':>10s} {'—':>10s} {0:>8d}  NO_POINTS")
            else:
                flag_str = "  " + " ".join(flags) if flags else ""
                lines.append(f"{name:28s} {near:10.4f} {far:10.4f} {med:10.4f} {rng:10.4f} {n_pts:>8d}{flag_str}")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Depth range statistics from COLMAP sparse reconstructions")
    parser.add_argument("input_path",
                        help="COLMAP sparse dir, transforms.json, or parent dataset dir")
    parser.add_argument("--bins", type=int, default=20,
                        help="Histogram bin count (default: 20)")
    parser.add_argument("--sort", choices=["name", "near", "far", "median", "range"],
                        default="name", help="Per-camera table sort (default: name)")
    parser.add_argument("--no-per-camera", action="store_true",
                        help="Omit per-camera table")
    parser.add_argument("-o", "--output", help="Write report to file")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    if not os.path.exists(input_path):
        print(f"Error: path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    mode, base_dir = _detect_input(input_path)
    if mode is None:
        print(f"Error: could not detect COLMAP or transforms.json data in: {input_path}",
              file=sys.stderr)
        sys.exit(1)

    print(f"  Detected mode: {mode} ({base_dir})", file=sys.stderr)

    if mode == "colmap":
        results, n_cameras, n_points, mode_label = _compute_colmap_depths(base_dir)
    else:
        results, n_cameras, n_points, mode_label = _compute_transforms_depths(base_dir)

    global_stats, cam_stats = _compute_stats(results)

    # Collect all depths for histogram
    all_depths = np.concatenate([d for _, _, d, _ in results if len(d) > 0])

    report = _build_report(mode_label, n_cameras, n_points, global_stats,
                           cam_stats, all_depths, args)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Visualize nerfstudio camera paths and training camera poses in 3D.

Overlays camera positions from multiple sources (nerfstudio camera path JSONs
and/or transforms.json files) in an interactive 3D plot. Optionally draws
per-camera coordinate frames (X=red, Y=green, Z=blue).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
]


def load_camera_path(path):
    """Load poses from a nerfstudio camera path JSON.

    Returns (poses, keyframe_poses) where each is a list of 4x4 numpy arrays.
    """
    with open(path) as f:
        data = json.load(f)

    poses = []
    for entry in data["camera_path"]:
        matrix = np.asarray(entry["camera_to_world"], dtype=np.float64).reshape(4, 4)
        poses.append(matrix)

    keyframe_poses = []
    for kf in data.get("keyframes", []):
        matrix = np.asarray(kf["matrix"], dtype=np.float64).reshape(4, 4)
        keyframe_poses.append(matrix)

    return poses, keyframe_poses


def load_transforms(path):
    """Load poses from a nerfstudio transforms.json.

    Returns a list of 4x4 camera-to-world numpy arrays.
    """
    with open(path) as f:
        data = json.load(f)

    poses = []
    for frame in data["frames"]:
        matrix = np.asarray(frame["transform_matrix"], dtype=np.float64).reshape(4, 4)
        poses.append(matrix)

    return poses


def positions_from_poses(poses):
    """Extract Nx3 camera positions (translation column) from c2w matrices."""
    return np.array([p[:3, 3] for p in poses])


def make_path_traces(positions, name, color, show_line=True, marker_size=3):
    """Create plotly traces for camera positions and optional connecting line."""
    traces = []

    if show_line and len(positions) > 1:
        traces.append(go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode="lines",
            line=dict(color=color, width=2),
            name=f"{name} (path)",
            legendgroup=name,
            showlegend=False,
            hoverinfo="skip",
        ))

    traces.append(go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        mode="markers",
        marker=dict(size=marker_size, color=color, opacity=0.8),
        name=name,
        legendgroup=name,
    ))

    return traces


def make_keyframe_traces(keyframe_poses, name, color):
    """Create larger diamond markers for keyframe positions."""
    if not keyframe_poses:
        return []
    positions = positions_from_poses(keyframe_poses)
    return [go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        mode="markers",
        marker=dict(
            size=7, color=color, symbol="diamond",
            opacity=1.0, line=dict(color="white", width=1),
        ),
        name=f"{name} (keyframes)",
        legendgroup=name,
    )]


def make_axes_traces(poses, scale, name):
    """Create RGB coordinate frame lines at each camera position.

    X = red, Y = green, Z = blue.  Each axis is drawn as a line segment
    from the camera origin to origin + scale * axis_direction.
    """
    traces = []
    axis_colors = ["red", "green", "blue"]
    axis_labels = ["X", "Y", "Z"]

    for axis_idx in range(3):
        xs, ys, zs = [], [], []
        for pose in poses:
            origin = pose[:3, 3]
            tip = origin + scale * pose[:3, axis_idx]
            xs.extend([origin[0], tip[0], None])
            ys.extend([origin[1], tip[1], None])
            zs.extend([origin[2], tip[2], None])

        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=axis_colors[axis_idx], width=3),
            name=f"{name} axes ({axis_labels[axis_idx]})"
            if axis_idx == 0
            else axis_labels[axis_idx],
            legendgroup=f"{name}_axes",
            showlegend=(axis_idx == 0),
            hoverinfo="skip",
        ))

    return traces


def auto_axis_scale(all_positions):
    """Compute axis arrow length as 3% of scene extent."""
    if len(all_positions) == 0:
        return 0.1
    extent = np.ptp(all_positions, axis=0).max()
    return max(extent * 0.03, 1e-6)


def build_figure(sources, show_axes, axis_scale):
    """Assemble all traces into a plotly Figure."""
    fig = go.Figure()
    all_positions = []

    for i, src in enumerate(sources):
        color = COLORS[i % len(COLORS)]
        positions = src["positions"]
        all_positions.append(positions)

        show_line = src["type"] == "camera_path"
        fig.add_traces(make_path_traces(
            positions, src["name"], color,
            show_line=show_line,
            marker_size=3 if show_line else 4,
        ))

        if src.get("keyframe_poses"):
            fig.add_traces(make_keyframe_traces(
                src["keyframe_poses"], src["name"], color,
            ))

    all_pos = np.vstack(all_positions) if all_positions else np.zeros((1, 3))
    scale = axis_scale if axis_scale is not None else auto_axis_scale(all_pos)

    if show_axes:
        for i, src in enumerate(sources):
            fig.add_traces(make_axes_traces(src["poses"], scale, src["name"]))

    fig.update_layout(
        title="Camera Poses",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize nerfstudio camera paths and training poses in 3D.",
        epilog="""\
examples:
  # Compare a camera path against training cameras
  %(prog)s --camera-path datasets/saltpond/camera_paths/1.json \\
           --transforms datasets/saltpond/saltpond_unprocessed/transforms.json

  # Show coordinate frames at each camera
  %(prog)s --transforms transforms.json --show-axes

  # Subsample dense paths, save to HTML
  %(prog)s --camera-path path.json --subsample 5 -o cameras.html
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--camera-path", action="append", default=[], metavar="JSON",
        help="nerfstudio camera path JSON (repeatable)",
    )
    parser.add_argument(
        "--transforms", action="append", default=[], metavar="JSON",
        help="nerfstudio transforms.json (repeatable)",
    )
    parser.add_argument(
        "--show-axes", action="store_true",
        help="draw XYZ coordinate frames (X=red, Y=green, Z=blue)",
    )
    parser.add_argument(
        "--axis-scale", type=float, default=None, metavar="FLOAT",
        help="coordinate frame arrow length (default: 3%% of scene extent)",
    )
    parser.add_argument(
        "--subsample", type=int, default=None, metavar="N",
        help="show every Nth camera (useful for dense paths)",
    )
    parser.add_argument(
        "-o", "--output", default=None, metavar="FILE",
        help="save to file (.html interactive, .png static); default: open in browser",
    )

    args = parser.parse_args()

    if not args.camera_path and not args.transforms:
        parser.error("provide at least one --camera-path or --transforms")

    sources = []

    for path_str in args.camera_path:
        path = Path(path_str)
        if not path.is_file():
            print(f"Error: not found: {path}", file=sys.stderr)
            sys.exit(1)
        poses, keyframe_poses = load_camera_path(path)
        if args.subsample:
            poses = poses[::args.subsample]
        label = path.stem
        if path.parent.name not in (".", ""):
            label = f"{path.parent.name}/{path.stem}"
        sources.append({
            "name": label,
            "type": "camera_path",
            "poses": poses,
            "positions": positions_from_poses(poses),
            "keyframe_poses": keyframe_poses,
        })

    for path_str in args.transforms:
        path = Path(path_str)
        if not path.is_file():
            print(f"Error: not found: {path}", file=sys.stderr)
            sys.exit(1)
        poses = load_transforms(path)
        if args.subsample:
            poses = poses[::args.subsample]
        label = path.parent.name if path.parent.name not in (".", "") else path.stem
        sources.append({
            "name": label,
            "type": "transforms",
            "poses": poses,
            "positions": positions_from_poses(poses),
            "keyframe_poses": [],
        })

    for s in sources:
        n = len(s["poses"])
        kf = len(s.get("keyframe_poses", []))
        extra = f" ({kf} keyframes)" if kf else ""
        print(f"  {s['name']}: {n} cameras{extra}", file=sys.stderr)

    fig = build_figure(sources, args.show_axes, args.axis_scale)

    if args.output:
        out = Path(args.output)
        if out.suffix == ".html":
            fig.write_html(str(out))
            print(f"Saved: {out}", file=sys.stderr)
        elif out.suffix == ".png":
            fig.write_image(str(out), width=1920, height=1080)
            print(f"Saved: {out}", file=sys.stderr)
        else:
            print(f"Error: unsupported format '{out.suffix}' (use .html or .png)",
                  file=sys.stderr)
            sys.exit(1)
    else:
        fig.show()


if __name__ == "__main__":
    main()

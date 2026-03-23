"""
Underwater dataset characterization.

Computes color cast, turbidity, and visibility metrics per frame.

Usage:
    python scripts/dataset_underwater.py <image_dir>
    python scripts/dataset_underwater.py <image_dir> --sort uciqe
    python scripts/dataset_underwater.py <image_dir> --json -o analysis.json
    python scripts/dataset_underwater.py <image_dir> --temporal
    python scripts/dataset_underwater.py <image_dir> --temporal --temporal-window 10
    python scripts/dataset_underwater.py <image_dir> --depth-source <colmap_or_dataset_dir>
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Color cast metrics
# ---------------------------------------------------------------------------

def compute_color_cast(img_bgr):
    """Channel ratios, gray-world deviation, CIELAB a*/b*."""
    img_f = img_bgr.astype(np.float32) / 255.0
    mu_b, mu_g, mu_r = [img_f[:, :, c].mean() for c in range(3)]
    mu = (mu_r + mu_g + mu_b) / 3.0

    rg_ratio = mu_r / (mu_g + 1e-8)
    bg_ratio = mu_b / (mu_g + 1e-8)
    gw_dev = max(abs(mu_r - mu), abs(mu_g - mu), abs(mu_b - mu)) / (mu + 1e-8)

    lab = cv2.cvtColor(img_f, cv2.COLOR_BGR2Lab)
    # float32 input: L in [0,100], a* in [-127,127], b* in [-127,127] (no offset)
    mean_a = lab[:, :, 1].mean()
    mean_b = lab[:, :, 2].mean()

    return {
        "rg_ratio": float(rg_ratio),
        "bg_ratio": float(bg_ratio),
        "gw_deviation": float(gw_dev),
        "mean_a_star": float(mean_a),
        "mean_b_star": float(mean_b),
    }


# ---------------------------------------------------------------------------
# UCIQE
# ---------------------------------------------------------------------------

def compute_uciqe(img_bgr):
    """UCIQE = 0.4680*sigma_c + 0.2745*con_l + 0.2576*mu_s."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    s = hsv[:, :, 1] / 255.0

    sigma_c = s.std()
    mu_s = s.mean()

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    p1, p99 = np.percentile(gray, [1, 99])
    con_l = (p99 - p1) / 255.0

    uciqe = 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s  # Yang & Sowmya 2015
    return float(uciqe)


# ---------------------------------------------------------------------------
# UIQM sub-metrics (Panetta et al. 2016)
# ---------------------------------------------------------------------------

def _uicm(img_rgb_f):
    """Colorfulness metric."""
    r, g, b = img_rgb_f[:, :, 0], img_rgb_f[:, :, 1], img_rgb_f[:, :, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b

    mu_rg, mu_yb = rg.mean(), yb.mean()
    var_rg, var_yb = rg.var(), yb.var()

    return -0.0268 * np.sqrt(mu_rg ** 2 + mu_yb ** 2) + 0.1586 * np.sqrt(var_rg + var_yb)  # Panetta et al. 2016


def _uism(img_bgr_f):
    """Sharpness metric via Sobel EME on each channel."""
    block_h, block_w = 10, 10  # Panetta et al. 2016
    eme_channels = []
    for c in range(3):
        ch = img_bgr_f[:, :, c]
        sx = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.sqrt(sx ** 2 + sy ** 2)

        h, w = edge.shape
        h_trunc = (h // block_h) * block_h
        w_trunc = (w // block_w) * block_w
        edge = edge[:h_trunc, :w_trunc]

        blocks = edge.reshape(h_trunc // block_h, block_h, w_trunc // block_w, block_w)
        blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, block_h, block_w)

        bmax = blocks.max(axis=(1, 2))
        bmin = blocks.min(axis=(1, 2))
        eme = np.mean(20.0 * np.log(bmax / (bmin + 1e-8) + 1e-8))
        eme_channels.append(eme)

    return float(np.mean(eme_channels))


def _uiconm(gray_f):
    """Contrast metric via block Michelson + logAMEE."""
    block_h, block_w = 10, 10  # Panetta et al. 2016
    h, w = gray_f.shape
    h_trunc = (h // block_h) * block_h
    w_trunc = (w // block_w) * block_w
    gray_f = gray_f[:h_trunc, :w_trunc]

    blocks = gray_f.reshape(h_trunc // block_h, block_h, w_trunc // block_w, block_w)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, block_h, block_w)

    bmax = blocks.max(axis=(1, 2))
    bmin = blocks.min(axis=(1, 2))

    contrast = (bmax - bmin) / (bmax + bmin + 1e-8)
    alpha = 0.1  # Panetta et al. 2016
    log_amee = np.mean(contrast ** alpha * np.log(contrast + 1e-8))
    return float(log_amee)


def compute_uiqm(img_bgr):
    """UIQM = 0.0282*UICM + 0.2953*UISM + 3.5753*UIConM."""
    img_f = img_bgr.astype(np.float64) / 255.0
    rgb_f = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    gray_f = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

    uicm = _uicm(rgb_f)
    uism = _uism(img_f)
    uiconm = _uiconm(gray_f)

    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm  # Panetta et al. 2016
    return float(uiqm)


# ---------------------------------------------------------------------------
# Dark channel prior
# ---------------------------------------------------------------------------

def compute_dark_channel(img_bgr, patch_size=41):
    """Min-channel + erode → dark channel statistics."""
    img_f = img_bgr.astype(np.float64) / 255.0
    min_ch = img_f.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dcp = cv2.erode(min_ch, kernel)
    return {
        "dcp_mean": float(dcp.mean()),
        "dcp_std": float(dcp.std()),
        "dcp_max": float(dcp.max()),
        "dcp_p95": float(np.percentile(dcp, 95)),
    }


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------

def compute_visibility(gray, canny_low=50, canny_high=150):
    """RMS contrast and edge density."""
    gray_f = gray.astype(np.float64) / 255.0
    rms_contrast = gray_f.std() / (gray_f.mean() + 1e-8)

    edges = cv2.Canny(gray, canny_low, canny_high)
    edge_density = float(np.count_nonzero(edges)) / edges.size

    return {
        "rms_contrast": float(rms_contrast),
        "edge_density": float(edge_density),
    }


# ---------------------------------------------------------------------------
# Per-frame analysis
# ---------------------------------------------------------------------------

def analyze_frame(path, dcp_patch_size=41, canny_low=50, canny_high=150):
    """Read image and compute all underwater metrics."""
    img = cv2.imread(path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = {"frame": os.path.basename(path)}
    result.update(compute_color_cast(img))
    result["uciqe"] = compute_uciqe(img)
    result["uiqm"] = compute_uiqm(img)
    result.update(compute_dark_channel(img, dcp_patch_size))
    result.update(compute_visibility(gray, canny_low, canny_high))
    result["mean_luminance"] = float(gray.mean() / 255.0)
    return result


# ---------------------------------------------------------------------------
# Depth-color correlation
# ---------------------------------------------------------------------------

def compute_depth_per_frame(depth_source_path):
    """Compute per-camera median depth from COLMAP or transforms.json.

    Args:
        depth_source_path: path to COLMAP sparse dir, transforms.json, or
            dataset directory (auto-detected via dataset_depth._detect_input).

    Returns dict mapping image basename to median scene depth.
    """
    from dataset_depth import (
        _detect_input,
        _compute_colmap_depths,
        _compute_transforms_depths,
    )

    mode, base_dir = _detect_input(depth_source_path)
    if mode is None:
        print(f"  Warning: could not detect depth source at {depth_source_path}",
              file=sys.stderr)
        return {}

    print(f"  Loading depth data ({mode} mode)...", file=sys.stderr)
    if mode == "colmap":
        results, _, _, _ = _compute_colmap_depths(base_dir)
    else:
        results, _, _, _ = _compute_transforms_depths(base_dir)

    depth_map = {}
    for name, _cam_pos, dists, _n_pts in results:
        basename = os.path.basename(name)
        if len(dists) > 0:
            depth_map[basename] = float(np.median(dists))

    return depth_map


def compute_depth_color_correlation(results, depth_map, n_bins=5):
    """Correlate per-frame color metrics with camera depth.

    Args:
        results: list of per-frame metric dicts from analyze_frame().
        depth_map: dict mapping image basename to median depth.
        n_bins: number of equal-count depth bins.

    Returns dict with 'correlations', 'depth_bins', 'frames_matched',
    'frames_unmatched'.
    """
    from scipy.stats import pearsonr, spearmanr

    color_metrics = [
        "rg_ratio", "bg_ratio", "gw_deviation", "mean_a_star", "mean_b_star",
        "uciqe", "uiqm", "dcp_mean",
    ]

    # Match frames by basename
    matched = []
    unmatched = 0
    for r in results:
        depth = depth_map.get(r["frame"])
        if depth is not None:
            matched.append((depth, r))
        else:
            unmatched += 1

    if len(matched) < 3:
        return {
            "correlations": {},
            "depth_bins": [],
            "frames_matched": len(matched),
            "frames_unmatched": unmatched,
        }

    # Sort by depth for binning
    matched.sort(key=lambda x: x[0])
    depths = np.array([d for d, _ in matched])

    # Correlations
    correlations = {}
    for m in color_metrics:
        vals = np.array([r[m] for _, r in matched])
        if np.std(vals) == 0 or np.std(depths) == 0:
            continue
        pr, pp = pearsonr(depths, vals)
        sr, sp = spearmanr(depths, vals)
        correlations[m] = {
            "pearson_r": round(float(pr), 4),
            "pearson_p": round(float(pp), 4),
            "spearman_r": round(float(sr), 4),
            "spearman_p": round(float(sp), 4),
        }

    # Equal-count depth bins
    bin_edges = np.percentile(depths, np.linspace(0, 100, n_bins + 1))
    depth_bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            bin_items = [(d, r) for d, r in matched if lo <= d <= hi]
        else:
            bin_items = [(d, r) for d, r in matched if lo <= d < hi]

        if not bin_items:
            continue

        bin_depths = [d for d, _ in bin_items]
        bin_entry = {
            "range": f"{lo:.2f}-{hi:.2f}",
            "depth_mean": round(float(np.mean(bin_depths)), 3),
            "count": len(bin_items),
        }
        for m in color_metrics:
            vals = [r[m] for _, r in bin_items]
            bin_entry[m] = round(float(np.mean(vals)), 4)
        depth_bins.append(bin_entry)

    return {
        "correlations": correlations,
        "depth_bins": depth_bins,
        "frames_matched": len(matched),
        "frames_unmatched": unmatched,
    }


def build_depth_text_section(correlation_data):
    """Format depth-color correlation as text lines."""
    lines = []
    lines.append("Depth-Color Correlation")
    lines.append("-----------------------")

    matched = correlation_data["frames_matched"]
    unmatched = correlation_data["frames_unmatched"]
    total = matched + unmatched
    lines.append(f"Frames matched: {matched}/{total}")

    if not correlation_data["correlations"]:
        lines.append("  Insufficient matched frames for correlation analysis.")
        return lines

    # Correlation table
    lines.append("")
    lines.append("Correlations (depth vs metric):")
    hdr = f"  {'Metric':20s} {'Pearson r':>10s} {'p-value':>10s} {'Spearman r':>10s} {'p-value':>10s}"
    lines.append(hdr)
    metric_labels = {
        "rg_ratio": "R/G ratio", "bg_ratio": "B/G ratio",
        "gw_deviation": "Gray-world dev", "mean_a_star": "CIELAB a*",
        "mean_b_star": "CIELAB b*", "uciqe": "UCIQE", "uiqm": "UIQM",
        "dcp_mean": "DCP mean",
    }
    for m, corr in correlation_data["correlations"].items():
        label = metric_labels.get(m, m)
        lines.append(
            f"  {label:20s} {corr['pearson_r']:10.4f} {corr['pearson_p']:10.4f} "
            f"{corr['spearman_r']:10.4f} {corr['spearman_p']:10.4f}"
        )

    # Depth bins table
    bins = correlation_data["depth_bins"]
    if bins:
        lines.append("")
        lines.append("Per-Bin Averages:")
        bin_hdr = f"  {'Depth Range':16s} {'Count':>6s} {'R/G':>7s} {'B/G':>7s} {'GW_Dev':>7s} {'UCIQE':>7s} {'DCP':>7s}"
        lines.append(bin_hdr)
        for b in bins:
            lines.append(
                f"  {b['range']:16s} {b['count']:6d} "
                f"{b.get('rg_ratio', 0):7.3f} {b.get('bg_ratio', 0):7.3f} "
                f"{b.get('gw_deviation', 0):7.3f} {b.get('uciqe', 0):7.4f} "
                f"{b.get('dcp_mean', 0):7.4f}"
            )

    return lines


# ---------------------------------------------------------------------------
# Temporal analysis
# ---------------------------------------------------------------------------

# Outlier threshold: frames beyond this many std deviations are flagged
_OUTLIER_SIGMA = 2


def compute_temporal_stats(results, window=5):
    """Compute inter-frame appearance variance and detect outlier clusters.

    Args:
        results: list of per-frame metric dicts (must be in temporal order).
        window: rolling window size for variance computation.

    Returns dict with 'global', 'per_frame', 'outlier_clusters', 'temporal_window'.
    """
    n = len(results)
    if n < 2:
        return {
            "global": {},
            "per_frame": [],
            "outlier_clusters": [],
            "temporal_window": window,
        }

    tracked_metrics = ["mean_luminance", "rg_ratio", "bg_ratio"]
    series = {m: np.array([r[m] for r in results]) for m in tracked_metrics}

    # Global stats
    global_stats = {}
    for m in tracked_metrics:
        global_stats[f"{m}_mean"] = float(series[m].mean())
        global_stats[f"{m}_std"] = float(series[m].std())

    # Per-frame deltas and rolling std
    per_frame = []
    for i in range(n):
        entry = {"frame": results[i]["frame"]}

        # Frame-to-frame deltas
        for m in tracked_metrics:
            if i > 0:
                entry[f"delta_{m}"] = float(abs(series[m][i] - series[m][i - 1]))
            else:
                entry[f"delta_{m}"] = 0.0

        # Rolling window std
        for m in tracked_metrics:
            start = max(0, i - window + 1)
            win = series[m][start:i + 1]
            entry[f"{m}_rolling_std"] = float(win.std()) if len(win) > 1 else 0.0

        per_frame.append(entry)

    # Max deltas
    for m in tracked_metrics:
        deltas = [pf[f"delta_{m}"] for pf in per_frame[1:]]
        if deltas:
            max_idx = int(np.argmax(deltas)) + 1  # +1 because deltas start from frame 1
            global_stats[f"max_delta_{m}"] = float(max(deltas))
            global_stats[f"max_delta_{m}_frame"] = results[max_idx]["frame"]

    # Outlier detection: flag frames where any tracked metric exceeds mean +/- 2*std
    outlier_flags = [False] * n
    outlier_metrics_per_frame = [[] for _ in range(n)]

    for m in tracked_metrics:
        mean = series[m].mean()
        std = series[m].std()
        if std == 0:
            continue
        for i in range(n):
            deviation = abs(series[m][i] - mean) / std
            if deviation > _OUTLIER_SIGMA:
                outlier_flags[i] = True
                outlier_metrics_per_frame[i].append(m)

    for i in range(n):
        per_frame[i]["is_outlier"] = outlier_flags[i]
        per_frame[i]["outlier_metrics"] = outlier_metrics_per_frame[i]

    # Cluster contiguous outliers
    clusters = []
    i = 0
    while i < n:
        if outlier_flags[i]:
            start = i
            cluster_metrics = set(outlier_metrics_per_frame[i])
            max_dev = 0.0
            while i < n and outlier_flags[i]:
                cluster_metrics.update(outlier_metrics_per_frame[i])
                for m in outlier_metrics_per_frame[i]:
                    std = series[m].std()
                    if std > 0:
                        dev = abs(series[m][i] - series[m].mean()) / std
                        max_dev = max(max_dev, dev)
                i += 1
            clusters.append({
                "start_frame": results[start]["frame"],
                "end_frame": results[i - 1]["frame"],
                "start_idx": start,
                "end_idx": i - 1,
                "length": i - start,
                "metrics": sorted(cluster_metrics),
                "max_deviation_sigma": round(max_dev, 2),
            })
        else:
            i += 1

    return {
        "global": global_stats,
        "per_frame": per_frame,
        "outlier_clusters": clusters,
        "temporal_window": window,
    }


def build_temporal_text_section(temporal_data):
    """Format temporal analysis as text lines."""
    lines = []
    lines.append("Temporal Analysis")
    lines.append("-----------------")

    g = temporal_data["global"]
    window = temporal_data["temporal_window"]
    lines.append(f"Rolling window: {window} frames")
    lines.append("")

    if not g:
        lines.append("  Insufficient frames for temporal analysis.")
        return lines

    lines.append("Global Statistics:")
    for m in ["mean_luminance", "rg_ratio", "bg_ratio"]:
        mean_key = f"{m}_mean"
        std_key = f"{m}_std"
        if mean_key in g:
            label = m.replace("_", " ").title()
            lines.append(f"  {label:20s}  {g[mean_key]:.4f} +/- {g[std_key]:.4f}")

    lines.append("")
    lines.append("Max Frame-to-Frame Deltas:")
    for m in ["mean_luminance", "rg_ratio", "bg_ratio"]:
        delta_key = f"max_delta_{m}"
        frame_key = f"max_delta_{m}_frame"
        if delta_key in g:
            label = m.replace("_", " ").title()
            lines.append(f"  {label:20s}  {g[delta_key]:.4f}  ({g[frame_key]})")

    clusters = temporal_data["outlier_clusters"]
    lines.append("")
    if clusters:
        lines.append(f"Outlier Clusters ({_OUTLIER_SIGMA} sigma):")
        for c in clusters:
            if c["length"] == 1:
                lines.append(
                    f"  Frame {c['start_frame']}: "
                    f"{', '.join(c['metrics'])} deviation {c['max_deviation_sigma']:.1f} sigma"
                )
            else:
                lines.append(
                    f"  Frames {c['start_frame']} - {c['end_frame']} "
                    f"({c['length']} frames): "
                    f"{', '.join(c['metrics'])} deviation {c['max_deviation_sigma']:.1f} sigma"
                )
    else:
        lines.append("No outlier clusters detected.")

    return lines


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------

def _stat_row(label, values, width=19):
    """Format a summary stats row."""
    arr = np.array(values)
    return (
        f"{label:{width}s} {arr.mean():8.4f}   {np.median(arr):8.4f}   "
        f"{arr.std():8.4f}   {arr.min():8.4f}   {arr.max():8.4f}"
    )


def build_text_report(image_dir, resolution, results, summary,
                      temporal_data=None, depth_correlation=None):
    """Build lines-based text report."""
    lines = []
    lines.append("Underwater Dataset Analysis")
    lines.append("==========================")
    lines.append(f"Directory:  {image_dir}")
    lines.append(f"Frames:     {len(results)}")
    lines.append(f"Resolution: {resolution[0]} x {resolution[1]}")
    lines.append("")

    # Summary stats
    lines.append("Summary")
    lines.append("-------")
    hdr = f"{'':19s} {'Mean':>8s}   {'Median':>8s}   {'Std':>8s}   {'Min':>8s}   {'Max':>8s}"
    lines.append(hdr)
    for key, label in [
        ("rg_ratio", "R/G ratio"),
        ("bg_ratio", "B/G ratio"),
        ("gw_deviation", "Gray-world dev"),
        ("mean_a_star", "CIELAB a*"),
        ("mean_b_star", "CIELAB b*"),
        ("uciqe", "UCIQE"),
        ("uiqm", "UIQM"),
        ("dcp_mean", "DCP mean"),
        ("rms_contrast", "RMS contrast"),
        ("edge_density", "Edge density"),
        ("mean_luminance", "Mean luminance"),
    ]:
        vals = [r[key] for r in results]
        lines.append(_stat_row(label, vals))
    lines.append("")

    # Per-frame table
    lines.append("Per-Frame Results")
    lines.append("-----------------")
    lines.append(f"{'Frame':24s} {'R/G':>6s} {'B/G':>6s} {'GW_Dev':>6s} {'UCIQE':>6s} {'UIQM':>7s} {'DCP':>6s}")
    for r in results:
        lines.append(
            f"{r['frame']:24s} {r['rg_ratio']:6.3f} {r['bg_ratio']:6.3f} "
            f"{r['gw_deviation']:6.3f} {r['uciqe']:6.4f} {r['uiqm']:7.4f} {r['dcp_mean']:6.4f}"
        )

    if depth_correlation is not None:
        lines.append("")
        lines.extend(build_depth_text_section(depth_correlation))

    if temporal_data is not None:
        lines.append("")
        lines.extend(build_temporal_text_section(temporal_data))

    return "\n".join(lines) + "\n"


def build_json_output(image_dir, resolution, results, summary,
                      temporal_data=None, depth_correlation=None):
    """Build JSON output dict."""
    output = {
        "metadata": {
            "directory": image_dir,
            "frames": len(results),
            "resolution": {"width": resolution[0], "height": resolution[1]},
        },
        "summary": summary,
        "per_frame": results,
    }
    if depth_correlation is not None:
        output["depth_correlation"] = depth_correlation
    if temporal_data is not None:
        output["temporal"] = temporal_data
    return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Underwater dataset characterization")
    parser.add_argument("image_dir", help="Directory of images")
    parser.add_argument("--sort", choices=["name", "uciqe", "uiqm", "gw_dev", "dcp"],
                        default="name", help="Sort per-frame table (default: name)")
    parser.add_argument("--dcp-patch-size", type=int, default=41,
                        help="Dark channel patch size (default: 41)")
    parser.add_argument("--canny-low", type=int, default=50,
                        help="Canny edge detector low threshold (default: 50)")
    parser.add_argument("--canny-high", type=int, default=150,
                        help="Canny edge detector high threshold (default: 150)")
    parser.add_argument("-o", "--output", help="Write to file instead of stdout")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output JSON instead of text")
    parser.add_argument("--depth-source", default=None, dest="depth_source",
                        help="Path to COLMAP sparse dir, transforms.json, or dataset dir "
                             "for depth-color correlation analysis")
    parser.add_argument("--depth-bins", type=int, default=5, dest="depth_bins",
                        help="Number of depth bins for correlation analysis (default: 5)")
    parser.add_argument("--temporal", action="store_true",
                        help="Enable inter-frame appearance variance analysis")
    parser.add_argument("--temporal-window", type=int, default=5,
                        dest="temporal_window",
                        help="Rolling window size for temporal stats (default: 5)")
    args = parser.parse_args()

    image_dir = os.path.abspath(args.image_dir)
    if not os.path.isdir(image_dir):
        print(f"Error: not a directory: {image_dir}", file=sys.stderr)
        sys.exit(1)

    paths = sorted(
        glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.JPG"))
    )
    if not paths:
        print(f"Error: no image files in {image_dir}", file=sys.stderr)
        sys.exit(1)

    # Resolution from first image
    first = cv2.imread(paths[0])
    h, w = first.shape[:2]

    # Analyze all frames
    results = []
    for i, p in enumerate(paths):
        print(f"\r  Processing {i + 1}/{len(paths)}...", end="", file=sys.stderr)
        r = analyze_frame(p, args.dcp_patch_size, args.canny_low, args.canny_high)
        if r is None:
            print(f"\n  Warning: failed to read {p}", file=sys.stderr)
            continue
        results.append(r)
    print(file=sys.stderr)

    if not results:
        print("Error: no frames could be read", file=sys.stderr)
        sys.exit(1)

    # Sort
    sort_keys = {
        "name": "frame", "uciqe": "uciqe", "uiqm": "uiqm",
        "gw_dev": "gw_deviation", "dcp": "dcp_mean",
    }
    if args.sort != "name":
        results.sort(key=lambda r: r[sort_keys[args.sort]])

    # Aggregate summary
    metric_keys = [
        "rg_ratio", "bg_ratio", "gw_deviation", "mean_a_star", "mean_b_star",
        "uciqe", "uiqm", "dcp_mean", "dcp_std", "dcp_max", "dcp_p95",
        "rms_contrast", "edge_density", "mean_luminance",
    ]
    summary = {}
    for key in metric_keys:
        vals = np.array([r[key] for r in results])
        summary[key] = {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    # Depth-color correlation
    depth_correlation = None
    if args.depth_source:
        print("  Computing depth-color correlation...", file=sys.stderr)
        depth_map = compute_depth_per_frame(args.depth_source)
        if depth_map:
            depth_correlation = compute_depth_color_correlation(
                results, depth_map, args.depth_bins)
            print(f"  Matched {depth_correlation['frames_matched']}/"
                  f"{depth_correlation['frames_matched'] + depth_correlation['frames_unmatched']}"
                  f" frames to depth data", file=sys.stderr)
        else:
            print("  Warning: no depth data could be loaded", file=sys.stderr)

    # Temporal analysis (uses filename-sorted order regardless of --sort)
    temporal_data = None
    if args.temporal:
        temporal_results = sorted(results, key=lambda r: r["frame"])
        temporal_data = compute_temporal_stats(temporal_results, args.temporal_window)

    # Output
    if args.json_output:
        report = build_json_output(image_dir, (w, h), results, summary,
                                   temporal_data=temporal_data,
                                   depth_correlation=depth_correlation)
    else:
        report = build_text_report(image_dir, (w, h), results, summary,
                                   temporal_data=temporal_data,
                                   depth_correlation=depth_correlation)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()

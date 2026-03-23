"""
Underwater dataset characterization.

Computes color cast, turbidity, and visibility metrics per frame.

Usage:
    python scripts/dataset_underwater.py <image_dir>
    python scripts/dataset_underwater.py <image_dir> --sort uciqe
    python scripts/dataset_underwater.py <image_dir> --json -o analysis.json
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
    return result


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


def build_text_report(image_dir, resolution, results, summary):
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

    return "\n".join(lines) + "\n"


def build_json_output(image_dir, resolution, results, summary):
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
        "rms_contrast", "edge_density",
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

    # Output
    if args.json_output:
        report = build_json_output(image_dir, (w, h), results, summary)
    else:
        report = build_text_report(image_dir, (w, h), results, summary)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()

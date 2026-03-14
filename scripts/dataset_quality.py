"""
Per-frame image quality assessment for dataset directories.

Computes blur (Laplacian variance) and brightness (mean grayscale intensity)
for each frame, flags outliers, and prints a summary report.

Usage:
    python scripts/dataset_quality.py <image_dir>
    python scripts/dataset_quality.py <image_dir> --blur-threshold 80 --sort blur
    python scripts/dataset_quality.py <image_dir> -o report.txt
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np


def analyze_frame(path):
    """Return (blur_score, brightness) for a single image."""
    img = cv2.imread(path)
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    return blur, brightness


def main():
    parser = argparse.ArgumentParser(description="Per-frame image quality assessment")
    parser.add_argument("image_dir", help="Directory of images")
    parser.add_argument("--blur-threshold", type=float, default=100.0,
                        help="Laplacian variance below this = blurry (default: 100.0)")
    parser.add_argument("--bright-low", type=float, default=40.0,
                        help="Mean brightness below this = underexposed (default: 40.0)")
    parser.add_argument("--bright-high", type=float, default=220.0,
                        help="Mean brightness above this = overexposed (default: 220.0)")
    parser.add_argument("--sort", choices=["name", "blur", "brightness"], default="name",
                        help="Sort order for per-frame table (default: name)")
    parser.add_argument("-o", "--output", help="Write output to file instead of stdout")
    args = parser.parse_args()

    image_dir = os.path.abspath(args.image_dir)
    if not os.path.isdir(image_dir):
        print(f"Error: not a directory: {image_dir}", file=sys.stderr)
        sys.exit(1)

    paths = sorted(
        glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.jpg"))
    )
    if not paths:
        print(f"Error: no .png or .jpg files in {image_dir}", file=sys.stderr)
        sys.exit(1)

    # Read first image for resolution
    first = cv2.imread(paths[0])
    h, w = first.shape[:2]

    # Analyze all frames
    results = []
    for i, p in enumerate(paths):
        print(f"\r  Processing {i + 1}/{len(paths)}...", end="", file=sys.stderr)
        blur, brightness = analyze_frame(p)
        if blur is None:
            print(f"\n  Warning: failed to read {p}", file=sys.stderr)
            continue
        flags = []
        if blur < args.blur_threshold:
            flags.append("BLUR")
        if brightness < args.bright_low:
            flags.append("DARK")
        if brightness > args.bright_high:
            flags.append("BRIGHT")
        results.append((os.path.basename(p), blur, brightness, flags))
    print(file=sys.stderr)

    if not results:
        print("Error: no frames could be read", file=sys.stderr)
        sys.exit(1)

    # Sort
    if args.sort == "blur":
        results.sort(key=lambda r: r[1])
    elif args.sort == "brightness":
        results.sort(key=lambda r: r[2])

    # Stats
    blurs = np.array([r[1] for r in results])
    brights = np.array([r[2] for r in results])
    n_outliers = sum(1 for r in results if r[3])
    n_blur = sum(1 for r in results if "BLUR" in r[3])
    n_dark = sum(1 for r in results if "DARK" in r[3])
    n_bright = sum(1 for r in results if "BRIGHT" in r[3])

    # Build report
    lines = []
    lines.append("Dataset Quality Report")
    lines.append("======================")
    lines.append(f"Directory:  {image_dir}")
    lines.append(f"Frames:     {len(results)}")
    lines.append(f"Resolution: {w} x {h}")
    lines.append("")
    lines.append("Summary")
    lines.append("-------")
    lines.append(f"{'':19s} {'Mean':>8s}   {'Median':>8s}   {'Std':>8s}   {'Min':>8s}   {'Max':>8s}")
    lines.append(f"{'Blur (Lap var)':19s} {blurs.mean():8.2f}   {np.median(blurs):8.2f}   {blurs.std():8.2f}   {blurs.min():8.2f}   {blurs.max():8.2f}")
    lines.append(f"{'Brightness':19s} {brights.mean():8.2f}   {np.median(brights):8.2f}   {brights.std():8.2f}   {brights.min():8.2f}   {brights.max():8.2f}")
    lines.append("")
    pct = 100 * n_outliers / len(results)
    lines.append(f"Outliers: {n_outliers} of {len(results)} ({pct:.1f}%)")
    lines.append(f"  Blurry (<{args.blur_threshold}):      {n_blur}")
    lines.append(f"  Underexposed (<{args.bright_low}): {n_dark}")
    lines.append(f"  Overexposed (>{args.bright_high}): {n_bright}")
    lines.append("")
    lines.append("Per-Frame Results")
    lines.append("-----------------")
    lines.append(f"{'Frame':24s} {'Blur':>8s}   {'Brightness':>10s}  Flags")
    for name, blur, brightness, flags in results:
        flag_str = "  " + " ".join(flags) if flags else ""
        lines.append(f"{name:24s} {blur:8.2f}   {brightness:10.2f}{flag_str}")

    report = "\n".join(lines) + "\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()

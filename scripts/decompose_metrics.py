"""
Decompose SSIM and LPIPS metrics into per-component scores.

Computes SSIM luminance/contrast/structure components (Wang 2004) and
LPIPS per-layer distances from rendered test frames vs ground truth.
Can run retroactively on any existing experiment with renders.

Usage:
    python scripts/decompose_metrics.py <experiment_spec>
    python scripts/decompose_metrics.py <spec1> <spec2> --outputs-dir <path>
    python scripts/decompose_metrics.py <spec> --json --dataset-dir <path>
    python scripts/decompose_metrics.py <spec> --device cpu

Examples:
    python scripts/decompose_metrics.py tune04_seathru6k \\
        --outputs-dir ../fyp-playground/outputs

    python scripts/decompose_metrics.py saltpond_unprocessed/saltpond_unprocessed-tune10_gw05 \\
        --json --outputs-dir ../fyp-playground/outputs
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from eval_experiments import resolve_runs
from read_config import load_config, resolve_config_path, resolve_outputs_dir


# ---------------------------------------------------------------------------
# SSIM component decomposition (Wang et al. 2004)
# ---------------------------------------------------------------------------

# Default parameters matching pytorch_msssim
_SSIM_WINDOW_SIZE = 11
_SSIM_SIGMA = 1.5
_SSIM_K1 = 0.01
_SSIM_K2 = 0.03


def _gaussian_window(size, sigma, channels):
    """Create a Gaussian window for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) @ g.unsqueeze(0)  # outer product
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return window.expand(channels, 1, size, size).contiguous()


def compute_ssim_components(img1, img2, data_range=1.0,
                            window_size=_SSIM_WINDOW_SIZE,
                            sigma=_SSIM_SIGMA,
                            k1=_SSIM_K1, k2=_SSIM_K2):
    """Compute SSIM luminance, contrast, structure components.

    Args:
        img1, img2: [1, C, H, W] tensors in [0, data_range].

    Returns dict with ssim, luminance, contrast, structure (all float).
    """
    C = img1.shape[1]
    window = _gaussian_window(window_size, sigma, C).to(img1.device, img1.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2

    # Clamp to avoid negative variance from numerical errors
    sigma1_sq = sigma1_sq.clamp(min=0)
    sigma2_sq = sigma2_sq.clamp(min=0)

    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    C3 = C2 / 2

    sigma1 = torch.sqrt(sigma1_sq)
    sigma2 = torch.sqrt(sigma2_sq)

    # Luminance
    luminance_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)

    # Contrast
    contrast_map = (2 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)

    # Structure
    structure_map = (sigma12 + C3) / (sigma1 * sigma2 + C3)

    # Full SSIM
    ssim_map = luminance_map * contrast_map * structure_map

    return {
        "ssim": float(ssim_map.mean()),
        "luminance": float(luminance_map.mean()),
        "contrast": float(contrast_map.mean()),
        "structure": float(structure_map.mean()),
    }


# ---------------------------------------------------------------------------
# LPIPS per-layer extraction
# ---------------------------------------------------------------------------

def load_lpips_model(net="alex", device="cpu"):
    """Load LPIPS model with per-layer access.

    Uses torchmetrics internal _LPIPS which supports retperlayer.
    """
    from torchmetrics.functional.image.lpips import _LPIPS as LPIPSNet

    model = LPIPSNet(pretrained=True, net=net).to(device)
    model.eval()
    return model


def compute_lpips_layers(model, img1, img2, normalize=True):
    """Compute LPIPS total and per-layer distances.

    Args:
        model: loaded LPIPS model from load_lpips_model().
        img1, img2: [1, C, H, W] tensors in [0, 1].
        normalize: if True, inputs are [0, 1]; model rescales to [-1, 1].

    Returns dict with total lpips and per-layer list.
    """
    with torch.no_grad():
        total, per_layer = model(img1, img2, retperlayer=True, normalize=normalize)

    return {
        "lpips": float(total.item()),
        "layers": [float(l.item()) for l in per_layer],
    }


# ---------------------------------------------------------------------------
# Eval split resolution
# ---------------------------------------------------------------------------

def resolve_eval_indices(config, n_frames):
    """Determine eval frame indices from nerfstudio config.

    Args:
        config: loaded config object from config.yml.
        n_frames: total number of frames in the dataset.

    Returns sorted array of eval frame indices.
    """
    # Extract dataparser config
    try:
        dp = config.pipeline.datamanager.dataparser
    except AttributeError:
        # Fallback: assume fraction mode with default split
        dp = None

    eval_mode = getattr(dp, "eval_mode", "fraction") if dp else "fraction"
    all_indices = np.arange(n_frames)

    if eval_mode == "interval":
        eval_interval = getattr(dp, "eval_interval", 8)
        eval_indices = all_indices[all_indices % eval_interval == 0]

    elif eval_mode == "fraction":
        train_fraction = getattr(dp, "train_split_fraction", 0.9)
        n_train = math.ceil(n_frames * train_fraction)
        train_indices = np.linspace(0, n_frames - 1, n_train, dtype=int)
        eval_indices = np.setdiff1d(all_indices, train_indices)

    elif eval_mode == "all":
        eval_indices = all_indices

    else:
        # Fallback: every 8th frame
        print(f"  Warning: unknown eval_mode '{eval_mode}', using interval=8",
              file=sys.stderr)
        eval_indices = all_indices[all_indices % 8 == 0]

    return np.sort(eval_indices)


def resolve_dataset_dir(config, override_dir=None):
    """Get the dataset directory from config or override.

    Returns (dataset_dir, downscale_factor) or (None, None) if not found.
    """
    if override_dir:
        ds_dir = Path(override_dir)
        # Try to get downscale from config
        try:
            dp = config.pipeline.datamanager.dataparser
            downscale = getattr(dp, "downscale_factor", None)
        except AttributeError:
            downscale = None
        return ds_dir, downscale

    try:
        dp = config.pipeline.datamanager.dataparser
        data_path = Path(getattr(dp, "data", ""))
        downscale = getattr(dp, "downscale_factor", None)
        if data_path.is_dir():
            return data_path, downscale
    except AttributeError:
        pass

    return None, None


def get_gt_image_paths(dataset_dir, downscale_factor, n_frames=None):
    """Get sorted list of ground truth image paths.

    Uses images_N/ for downscaled, or images/ for full resolution.
    """
    if downscale_factor and downscale_factor > 1:
        img_dir = dataset_dir / f"images_{int(downscale_factor)}"
        if not img_dir.is_dir():
            img_dir = dataset_dir / "images"
    else:
        img_dir = dataset_dir / "images"

    if not img_dir.is_dir():
        return []

    paths = sorted(
        list(img_dir.glob("*.png"))
        + list(img_dir.glob("*.jpg"))
        + list(img_dir.glob("*.JPG"))
    )
    return paths


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

def load_video_frames(video_path):
    """Load all frames from an MP4 as numpy arrays (RGB, uint8).

    Returns list of np.ndarray [H, W, 3].
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error: cannot open {video_path}", file=sys.stderr)
        return []

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def load_image(path):
    """Load an image as RGB numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_tensor(img_np, device="cpu"):
    """Convert HWC uint8 numpy to [1, C, H, W] float tensor in [0, 1]."""
    t = torch.from_numpy(img_np).float() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    return t.to(device)


# ---------------------------------------------------------------------------
# Per-experiment processing
# ---------------------------------------------------------------------------

def process_experiment(run_dir, outputs_dir, dataset_dir_override, device,
                       lpips_model):
    """Process a single experiment: decompose SSIM and LPIPS.

    Returns (label, result_dict) or (label, None) on failure.
    """
    run_dir = Path(run_dir)
    timestamp_dir = run_dir

    # Derive label
    try:
        rel = run_dir.resolve().relative_to(Path(outputs_dir).resolve())
        parts = rel.parts
        if len(parts) >= 2:
            label = parts[1]
            # Strip dataset prefix
            dataset_prefix = parts[0] + "-"
            if label.startswith(dataset_prefix):
                label = label[len(dataset_prefix):]
        else:
            label = str(rel)
    except ValueError:
        label = run_dir.name

    print(f"\n=== {label} ===", file=sys.stderr)

    # Load config
    config_path = timestamp_dir / "config.yml"
    if not config_path.is_file():
        print(f"  Error: config.yml not found at {config_path}", file=sys.stderr)
        return label, None

    config = load_config(config_path)
    if config is None:
        print(f"  Error: failed to load config", file=sys.stderr)
        return label, None

    # Resolve dataset directory
    ds_dir, downscale = resolve_dataset_dir(config, dataset_dir_override)
    if ds_dir is None or not ds_dir.is_dir():
        print(f"  Error: dataset directory not found. Use --dataset-dir to specify.",
              file=sys.stderr)
        if ds_dir:
            print(f"  Tried: {ds_dir}", file=sys.stderr)
        return label, None

    # Get GT image paths
    gt_paths = get_gt_image_paths(ds_dir, downscale)
    if not gt_paths:
        print(f"  Error: no images found in {ds_dir}", file=sys.stderr)
        return label, None

    n_total = len(gt_paths)
    print(f"  Dataset: {ds_dir.name}, {n_total} total frames, downscale={downscale}",
          file=sys.stderr)

    # Determine eval indices
    eval_indices = resolve_eval_indices(config, n_total)
    n_eval = len(eval_indices)
    print(f"  Eval split: {n_eval} frames", file=sys.stderr)

    # Find render video
    render_dir = timestamp_dir / "renders" / "dataset"
    # Try "test" first, then "val"
    for split_name in ("test", "val"):
        video_path = render_dir / split_name / "medium_rgb.mp4"
        if video_path.is_file():
            break
    else:
        print(f"  Error: no medium_rgb.mp4 in {render_dir}/test/ or val/",
              file=sys.stderr)
        return label, None

    print(f"  Loading renders from {split_name}/medium_rgb.mp4...", file=sys.stderr)
    rendered_frames = load_video_frames(video_path)

    if not rendered_frames:
        print(f"  Error: no frames extracted from {video_path}", file=sys.stderr)
        return label, None

    n_rendered = len(rendered_frames)

    # Verify frame count match
    if n_rendered != n_eval:
        print(f"  Warning: rendered frames ({n_rendered}) != eval split ({n_eval}). "
              f"Using min({n_rendered}, {n_eval}).", file=sys.stderr)
        n_use = min(n_rendered, n_eval)
        eval_indices = eval_indices[:n_use]
        rendered_frames = rendered_frames[:n_use]
    else:
        n_use = n_eval

    print(f"  Processing {n_use} frames...", file=sys.stderr)

    # Process each frame
    per_frame = []
    for i in range(n_use):
        frame_idx = int(eval_indices[i])
        gt_path = gt_paths[frame_idx]
        gt_img = load_image(gt_path)

        if gt_img is None:
            print(f"  Warning: failed to load GT image {gt_path}", file=sys.stderr)
            continue

        rendered_img = rendered_frames[i]

        # Resize if needed (rendered may differ from GT due to rounding)
        gt_h, gt_w = gt_img.shape[:2]
        rd_h, rd_w = rendered_img.shape[:2]
        if (gt_h, gt_w) != (rd_h, rd_w):
            rendered_img = cv2.resize(rendered_img, (gt_w, gt_h),
                                      interpolation=cv2.INTER_LINEAR)

        gt_t = to_tensor(gt_img, device)
        rd_t = to_tensor(rendered_img, device)

        # SSIM components
        ssim_result = compute_ssim_components(gt_t, rd_t)

        # LPIPS per-layer
        lpips_result = compute_lpips_layers(lpips_model, gt_t, rd_t)

        per_frame.append({
            "frame_idx": frame_idx,
            "gt_file": gt_path.name,
            "ssim": ssim_result["ssim"],
            "ssim_luminance": ssim_result["luminance"],
            "ssim_contrast": ssim_result["contrast"],
            "ssim_structure": ssim_result["structure"],
            "lpips": lpips_result["lpips"],
            "lpips_layers": lpips_result["layers"],
        })

        print(f"\r  Frame {i + 1}/{n_use}: SSIM={ssim_result['ssim']:.4f} "
              f"(l={ssim_result['luminance']:.3f} c={ssim_result['contrast']:.3f} "
              f"s={ssim_result['structure']:.3f})  LPIPS={lpips_result['lpips']:.4f}",
              end="", file=sys.stderr)

    print(file=sys.stderr)

    if not per_frame:
        print(f"  Error: no frames processed successfully", file=sys.stderr)
        return label, None

    # Aggregate
    aggregate = {
        "ssim": float(np.mean([f["ssim"] for f in per_frame])),
        "ssim_luminance": float(np.mean([f["ssim_luminance"] for f in per_frame])),
        "ssim_contrast": float(np.mean([f["ssim_contrast"] for f in per_frame])),
        "ssim_structure": float(np.mean([f["ssim_structure"] for f in per_frame])),
        "lpips": float(np.mean([f["lpips"] for f in per_frame])),
        "lpips_layers": [
            float(np.mean([f["lpips_layers"][j] for f in per_frame]))
            for j in range(len(per_frame[0]["lpips_layers"]))
        ],
    }

    return label, {
        "num_frames": len(per_frame),
        "aggregate": aggregate,
        "per_frame": per_frame,
    }


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def format_text(results):
    """Human-readable text output."""
    lines = []

    for label, data in results.items():
        lines.append(f"=== {label} ({data['num_frames']} frames) ===")
        lines.append("")

        agg = data["aggregate"]

        # SSIM components
        lines.append("  SSIM Decomposition:")
        lines.append(f"    SSIM (composite):  {agg['ssim']:.4f}")
        lines.append(f"    Luminance:         {agg['ssim_luminance']:.4f}")
        lines.append(f"    Contrast:          {agg['ssim_contrast']:.4f}")
        lines.append(f"    Structure:         {agg['ssim_structure']:.4f}")
        product = agg["ssim_luminance"] * agg["ssim_contrast"] * agg["ssim_structure"]
        lines.append(f"    L*C*S check:       {product:.4f}")
        lines.append("")

        # LPIPS layers
        lines.append("  LPIPS Per-Layer:")
        lines.append(f"    Total:             {agg['lpips']:.4f}")
        for j, v in enumerate(agg["lpips_layers"]):
            pct = v / agg["lpips"] * 100 if agg["lpips"] > 0 else 0
            lines.append(f"    Layer {j + 1}:           {v:.4f}  ({pct:.1f}%)")
        layer_sum = sum(agg["lpips_layers"])
        lines.append(f"    Sum check:         {layer_sum:.4f}")
        lines.append("")

        # Identify bottleneck
        components = [
            ("luminance", agg["ssim_luminance"]),
            ("contrast", agg["ssim_contrast"]),
            ("structure", agg["ssim_structure"]),
        ]
        bottleneck = min(components, key=lambda x: x[1])
        lines.append(f"  SSIM bottleneck: {bottleneck[0]} ({bottleneck[1]:.4f})")

        max_layer_idx = int(np.argmax(agg["lpips_layers"]))
        lines.append(f"  LPIPS dominant layer: {max_layer_idx + 1} "
                      f"({agg['lpips_layers'][max_layer_idx]:.4f})")
        lines.append("")

        # Per-frame table
        lines.append("  Per-Frame Results:")
        hdr = (f"    {'GT File':24s} {'SSIM':>6s} {'Lum':>6s} "
               f"{'Con':>6s} {'Str':>6s} {'LPIPS':>6s}")
        lines.append(hdr)
        for f in data["per_frame"]:
            lines.append(
                f"    {f['gt_file']:24s} {f['ssim']:6.4f} "
                f"{f['ssim_luminance']:6.4f} {f['ssim_contrast']:6.4f} "
                f"{f['ssim_structure']:6.4f} {f['lpips']:6.4f}"
            )
        lines.append("")

    return "\n".join(lines)


def format_json(results):
    """JSON output."""
    return json.dumps({"experiments": results}, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Decompose SSIM and LPIPS into per-component scores.",
        epilog="""\
examples:
  %(prog)s tune04_seathru6k --outputs-dir ../fyp-playground/outputs
  %(prog)s tune10_gw05 tune10_gw15 --json --outputs-dir ../fyp-playground/outputs
  %(prog)s tune04_seathru6k --dataset-dir ../fyp-playground/datasets/saltpond/saltpond_unprocessed
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "specs", nargs="+",
        help="Experiment path specs (substring, method dir, or timestamp dir)",
    )
    parser.add_argument(
        "--outputs-dir", default=None,
        help="Base outputs directory (default: $NERFSTUDIO_OUTPUTS or ./outputs)",
    )
    parser.add_argument(
        "--dataset-dir", default=None,
        help="Override ground truth dataset directory (use when config.yml "
             "contains stale paths from a remote machine)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output JSON instead of human-readable text",
    )
    parser.add_argument(
        "--device", default=None,
        help="Compute device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"],
        dest="lpips_net",
        help="LPIPS backbone network (default: alex)",
    )

    args = parser.parse_args()

    # Resolve device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", file=sys.stderr)

    # Resolve outputs dir
    outputs_dir = resolve_outputs_dir(args.outputs_dir)

    # Load LPIPS model once
    print("Loading LPIPS model...", file=sys.stderr)
    lpips_model = load_lpips_model(net=args.lpips_net, device=device)

    # Process experiments
    all_results = {}
    for spec in args.specs:
        runs = resolve_runs(spec, outputs_dir)
        for run_dir in runs:
            label, data = process_experiment(
                run_dir, outputs_dir, args.dataset_dir, device, lpips_model)
            if data is not None:
                all_results[label] = data

    if not all_results:
        print("No experiments processed successfully.", file=sys.stderr)
        sys.exit(1)

    # Output
    if args.json:
        print(format_json(all_results))
    else:
        print(format_text(all_results))


if __name__ == "__main__":
    main()

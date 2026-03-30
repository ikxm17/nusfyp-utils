# Scripts

Utility scripts for managing nerfstudio experiment workflows — from running batch training to comparing configs across runs.

## Overview

| Script | Purpose |
|--------|---------|
| `experiments/run_experiments.py` | Batch-run nerfstudio training experiments |
| `read_config.py` | Read and diff nerfstudio experiment configs |
| `log_experiments.py` | Generate experiment logs with config diffs against a baseline |
| `eval_experiments.py` | Batch-run `ns-eval` and save metrics to experiment directories |
| `render.py` | Render experiments to video (wraps `ns-render` + ffmpeg) |
| `render_experiments.py` | Batch-render multiple experiments (wraps `render.py` for all runs) |
| `compare_renders.py` | Extract frames from render videos, compare across experiments visually |
| `change_config_path.py` | Rewrite hardcoded paths in nerfstudio configs for cross-machine use |
| `read_tb.py` | Read and analyze TensorBoard training curves from experiment runs |
| `dataset_quality.py` | Per-frame image quality assessment (blur, brightness, outlier detection) |
| `dataset_depth.py` | Depth range statistics from COLMAP sparse reconstructions |
| `dataset_underwater.py` | Underwater dataset characterization (color cast, turbidity, depth-color correlation, temporal variance) |
| `decompose_metrics.py` | Decompose SSIM (luminance/contrast/structure) and LPIPS (per-layer) from experiment renders |
| `paper_figures.py` | Generate publication-quality figures from TensorBoard training data (9 figure types) |

### Agent scripts (`agents/`)

| Script | Purpose |
|--------|---------|
| `agents/analyze_batch.py` | Gather all quantitative analysis for a batch of experiments into structured JSON |

### Script relationships

```
config/local_config.py ──> config/experiment_config.py ──> run_experiments.py
                           (local settings)                (config defines experiments, runner executes them)

read_config.py ──> log_experiments.py       (log generator uses reader to load/compare configs)
read_config.py ──> eval_experiments.py      (evaluator uses reader + log_experiments for path resolution)
log_experiments.py ──> eval_experiments.py
config/experiment_config.py ──> eval_experiments.py  (config mode uses experiment config for run discovery)
read_config.py ──> render.py               (renderer uses reader for path resolution)
render.py ──> render_experiments.py         (batch renderer imports render.py functions)
eval_experiments.py ──> render_experiments.py  (batch renderer reuses run resolution logic)
config/experiment_config.py ──> render_experiments.py  (config mode uses experiment config for run discovery)
read_config.py ──> compare_renders.py      (visual comparison uses reader for path resolution)
read_config.py ──> read_tb.py              (TB reader uses reader for path resolution + config loading)
log_experiments.py ──> read_tb.py          (uses find_runs for run discovery)
eval_experiments.py ──> read_tb.py         (uses resolve_runs for flexible path specs)
change_config_path.py                       (standalone — used manually when moving between machines)

read_tb.py ──> agents/analyze_batch.py     (batch analyzer calls TB comparison)
compare_renders.py ──> agents/analyze_batch.py  (batch analyzer calls grid + extract)
dataset_underwater.py ──> agents/analyze_batch.py (batch analyzer calls color analysis)
```

---

## experiments/run_experiments.py

Automates batch `ns-train` invocations so you don't have to manually run each training combination. Loads experiment definitions from a config module and executes them sequentially, capturing output to log files prefixed with `SUCCESS_` or `FAILED_`. Validates that `extra_args` flags are valid ns-train flags before execution.

### Usage

```bash
# Preview all commands without executing
python scripts/experiments/run_experiments.py --dry-run

# Run all experiments
python scripts/experiments/run_experiments.py

# Run only experiments whose name contains "torpedo"
python scripts/experiments/run_experiments.py --filter torpedo

# Use a custom config file
python scripts/experiments/run_experiments.py --config /path/to/my_config.py

# Add experiment index prefix to log filenames
python scripts/experiments/run_experiments.py --log-index
```

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--dry-run` | Preview commands without executing | off |
| `--config <path>` | Path to config `.py` file or module name | `config/experiment_config.py` |
| `--filter <substring>` | Only run experiments whose name contains this substring | none |
| `--log-index` | Prefix `expt_{index}_` to log filenames | off |

### Config module requirements

The config module must export:
- `EXPERIMENTS` — list of experiment dicts (see `config/experiment_config.py` for the schema)
- `LOG_DIR` — path to directory for log files

### Dependencies

- `ns-train` must be on `PATH` (nerfstudio installed)
- Python standard library only

---

> **Note:** Experiment configuration files (`experiment_config.py`, `local_config.py`, `local_config.example.py`) now live in the top-level `config/` directory. See those files directly for schema and customization details.

---

## read_config.py

CLI tool for reading and comparing nerfstudio experiment configurations. When running many experiments with different hyperparameters, this helps identify exactly what changed between runs without manually inspecting YAML files.

### Usage

```bash
# Read model config for an experiment
python scripts/read_config.py read <path> --section model

# Read a specific parameter
python scripts/read_config.py read <path> --param learn-background

# Diff two configs
python scripts/read_config.py diff <path-a> <path-b>

# Diff with custom labels
python scripts/read_config.py diff <path-a> <path-b> --name-a "baseline" --name-b "no-seathru"

# Use a custom outputs directory
python scripts/read_config.py read <path> --outputs-dir /path/to/outputs
```

### Path resolution

Paths can be specified as:
1. Full path to `config.yml` or its parent directory
2. Relative path from outputs dir (e.g. `saltpond_unprocessed/saltpond_unprocessed-a_exploration`)
3. Substring match on dataset/experiment names (e.g. `a_exploration`)

The script auto-descends the nerfstudio directory hierarchy (`dataset/experiment/method/timestamp/config.yml`), picking the latest timestamp when multiple exist.

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--section {model,optimizers,all}` | Config section to display/diff | `model` |
| `--param <name>` | Look up a specific parameter | none |
| `--outputs-dir <path>` | Base outputs directory | `$NERFSTUDIO_OUTPUTS` or `./outputs` |
| `--name-a`, `--name-b` | Labels for diff output | auto-derived from path |

### Dependencies

- `pyyaml` — uses `yaml.unsafe_load()` to handle nerfstudio's serialized Python dataclasses

---

## log_experiments.py

Generates a formatted report comparing all runs in an experiment directory against a baseline. Useful for tracking how hyperparameter changes across repeated runs affect the configuration, making it easier to correlate config differences with result differences.

### Usage

```bash
# Log all runs, using the earliest as baseline
python scripts/log_experiments.py /path/to/method-dir

# Pick a specific baseline by timestamp substring
python scripts/log_experiments.py /path/to/method-dir --baseline 024717

# Include runs from other directories
python scripts/log_experiments.py /path/to/method-dir --extra /other/method-dir

# Write output to a file
python scripts/log_experiments.py /path/to/method-dir -o experiment_log.txt

# Diff optimizers instead of model config
python scripts/log_experiments.py /path/to/method-dir --section optimizers
```

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `experiment_dir` | Path to method-level directory (or resolvable spec) | required |
| `--baseline <substring>` | Select baseline by timestamp substring | earliest run |
| `--extra <paths...>` | Additional directories or run paths to include | none |
| `--extra-labels <labels...>` | Custom labels for extra runs | none |
| `--section {model,optimizers,all}` | Config section to compare | `model` |
| `-o, --output <file>` | Write to file instead of stdout | stdout |
| `--outputs-dir <path>` | Base outputs directory for path resolution | `$NERFSTUDIO_OUTPUTS` or `./outputs` |

### Output format

The report includes:
- Full baseline config listing
- Per-run diff showing changed parameters, fields only in baseline, and fields only in that run

### Dependencies

- `read_config.py` (imported directly)
- `pyyaml`

---

## eval_experiments.py

Batch-evaluates nerfstudio experiment runs using `ns-eval`. Supports two modes: **config mode** (uses `experiment_config.py` to resolve runs, with `--filter` support) and **path mode** (resolves explicit path specs). Saves `metrics.json` directly into each run's timestamp directory, and prints computed metrics (PSNR, SSIM, LPIPS) to stdout after each run and in a summary table. Optionally saves rendered evaluation images.

### Usage

```bash
# Config mode — evaluate all experiments defined in experiment_config
python scripts/eval_experiments.py --dry-run

# Config mode — filter by experiment name substring
python scripts/eval_experiments.py --filter torpedo --dry-run

# Path mode — evaluate all runs under a method directory
python scripts/eval_experiments.py ../fyp-playground/outputs/saltpond_unprocessed/saltpond_unprocessed-a_exploration/sea-splatfacto

# Path mode — evaluate a single run by timestamp directory
python scripts/eval_experiments.py ../fyp-playground/outputs/.../2026-03-08_015758

# Path mode — evaluate multiple path specs
python scripts/eval_experiments.py a_exploration b_exploration --outputs-dir ../fyp-playground/outputs

# Skip runs that already have metrics
python scripts/eval_experiments.py <path> --skip-existing

# Also save rendered eval images
python scripts/eval_experiments.py <path> --render-images
```

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `paths` (positional) | Experiment path specs (timestamp dir, method dir, or substring). If omitted, uses config mode. | none |
| `--config <path>` | Path to config `.py` file or module name (config mode) | `config/experiment_config.py` |
| `--filter <substring>` | Only evaluate experiments whose name contains this substring (config mode only) | none |
| `--outputs-dir <path>` | Base outputs directory for resolution (path mode) | `$NERFSTUDIO_OUTPUTS` or `./outputs` |
| `--output-name <name>` | Metrics JSON filename | `metrics.json` |
| `--render-images` | Also save rendered eval images | off |
| `--render-dir-name <name>` | Subdirectory for rendered images | `eval_renders` |
| `--skip-existing` | Skip runs that already have a metrics file | off |
| `--dry-run` | Print commands without executing | off |

### Output placement

```
<timestamp>/
  config.yml
  nerfstudio_models/
  metrics.json          <-- ns-eval JSON output
  eval_renders/         <-- optional (--render-images)
```

### Dependencies

- `ns-eval` must be on `PATH` (nerfstudio installed)
- `read_config.py` (path resolution utilities)
- `log_experiments.py` (`find_runs`, `resolve_experiment_dir`)
- `experiments/run_experiments.py` (`load_config`, for config mode)
- `config/experiment_config.py` + `config/local_config.py` (config mode)

---

## render.py

Renders nerfstudio experiments to video in one command. Wraps `ns-render` with automatic ffmpeg video conversion and frame cleanup. Supports two modes: **dataset** (ground truth train/test splits for evaluation) and **camera-path** (smooth trajectories from camera path JSONs for visualisation).

### Usage

```bash
# Render train+test splits (default) for an experiment
python scripts/render.py dataset a_exploration --outputs-dir ../fyp-playground/outputs

# Render only the test split with rgb and depth outputs
python scripts/render.py dataset <experiment> --split test --rendered-output-names clean_rgb depth

# Render a camera path trajectory
python scripts/render.py camera-path <experiment> --camera-path /path/to/trajectory.json

# Render camera path with custom output name and keep frames
python scripts/render.py camera-path <experiment> --camera-path traj.json --camera-path-name flythrough --keep-frames

# Preview ns-render commands without executing
python scripts/render.py dataset <experiment> --dry-run
python scripts/render.py camera-path <experiment> --camera-path traj.json --dry-run
```

### Arguments

**Shared (both subcommands):**

| Flag | Description | Default |
|------|-------------|---------|
| `experiment` (positional) | Path/spec to experiment (resolved via `read_config.py`) | required |
| `--outputs-dir <path>` | Base outputs directory | `$NERFSTUDIO_OUTPUTS` or `./outputs` |
| `--output-dir <path>` | Override output directory | `<experiment>/renders/<subcommand>/` |
| `--rendered-output-names <names...>` | Output names to render | `clean_rgb` |
| `--fps <n>` | Video frame rate | `30` |
| `--keep-frames` | Preserve frame images after video creation | off |
| `--image-format {jpeg,png}` | Image format for frames | `jpeg` |
| `--jpeg-quality <n>` | JPEG quality 1-100 | `100` |
| `--downscale-factor <n>` | Resolution downscale factor | `1` |
| `--dry-run` | Print commands without executing | off |

**`dataset` subcommand:**

| Flag | Description | Default |
|------|-------------|---------|
| `--split <splits>` | Split(s) to render, `+`-separated | `train+test` |

**`camera-path` subcommand:**

| Flag | Description | Default |
|------|-------------|---------|
| `--camera-path <file>` | Path to camera path JSON file | required |
| `--camera-path-name <name>` | Name for output directory | derived from filename |

### Output structure

```
<timestamp>/renders/
├── dataset/
│   ├── test/
│   │   ├── clean_rgb.mp4
│   │   └── depth.mp4
│   └── train/
│       └── clean_rgb.mp4
└── camera-path/
    └── {path-name}/
        ├── clean_rgb.mp4
        └── depth.mp4
```

With `--keep-frames`, original frame directories are preserved alongside the videos.

### How it works

- **Dataset mode**: Runs `ns-render dataset` (always outputs frames), then converts each `{split}/{output_name}/` frame directory to video via ffmpeg concat demuxer
- **Camera-path mode**: Invokes `ns-render camera-path` once per output name (to avoid side-by-side concatenation). Without `--keep-frames`, renders directly to video. With `--keep-frames`, renders to images then converts via ffmpeg

Safety: frames are never deleted if ffmpeg fails, regardless of `--keep-frames`.

### Dependencies

- `ns-render` must be on `PATH` (nerfstudio installed)
- `ffmpeg` must be on `PATH` (needed for dataset mode; for camera-path only with `--keep-frames`)
- `read_config.py` (path resolution utilities)

---

## render_experiments.py

Batch-renders multiple nerfstudio experiments in one invocation. Wraps `render.py` to process all runs matching a path spec or experiment config. Follows the same dual-mode pattern as `eval_experiments.py`: **config mode** (uses `experiment_config.py` to resolve runs, with `--filter`) and **path mode** (explicit path specs). Supports both dataset and camera-path render types, with skip-existing detection so re-running after partial completions only renders what's missing.

### Usage

```bash
# Config mode — preview renders for all experiments
python scripts/render_experiments.py --dry-run

# Config mode — filter by experiment name
python scripts/render_experiments.py --filter saltpond --dry-run

# Path mode — render all runs under a method directory
python scripts/render_experiments.py ../fyp-playground/outputs/saltpond_unprocessed/saltpond_unprocessed-a_exploration/sea-splatfacto \
  --outputs-dir ../fyp-playground/outputs

# Path mode — render multiple experiment specs
python scripts/render_experiments.py a_exploration b_exploration --outputs-dir ../fyp-playground/outputs

# Render camera paths (auto-discovers from datasets/{group}/camera_paths/)
python scripts/render_experiments.py a_exploration --render-type camera-path --outputs-dir ../fyp-playground/outputs

# Render camera paths with explicit path
python scripts/render_experiments.py a_exploration --render-type camera-path \
  --camera-path ../fyp-playground/datasets/saltpond/camera_paths/1.json --outputs-dir ../fyp-playground/outputs

# Render both dataset + camera-path
python scripts/render_experiments.py a_exploration --render-type all --outputs-dir ../fyp-playground/outputs

# Skip runs that already have render videos
python scripts/render_experiments.py a_exploration --skip-existing --outputs-dir ../fyp-playground/outputs

# Render with depth output and downscaled resolution
python scripts/render_experiments.py a_exploration --rendered-output-names clean_rgb depth --downscale-factor 2 \
  --outputs-dir ../fyp-playground/outputs
```

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `paths` (positional) | Experiment path specs. If omitted, uses config mode. | none |
| `--render-type {dataset,camera-path,all}` | Which render type(s) to run | `dataset` |
| `--outputs-dir <path>` | Base outputs directory | `$NERFSTUDIO_OUTPUTS` or `./outputs` |
| `--config <path>` | Config module for config mode | `config/experiment_config.py` |
| `--filter <substring>` | Filter experiments (config mode only) | none |
| `--camera-path <file>` | Explicit camera path JSON (used for all runs) | none |
| `--camera-paths-dir <dir>` | Directory of camera path JSONs (all rendered per run) | none |
| `--rendered-output-names <names...>` | Output names to render | `clean_rgb` |
| `--split <splits>` | Dataset splits, `+`-separated | `train+test` |
| `--fps <n>` | Video frame rate | `30` |
| `--keep-frames` | Preserve frame images after video creation | off |
| `--image-format {jpeg,png}` | Frame format | `jpeg` |
| `--jpeg-quality <n>` | JPEG quality 1-100 | `100` |
| `--downscale-factor <n>` | Resolution downscale factor | `1` |
| `--skip-existing` | Skip runs whose render videos already exist | off |
| `--dry-run` | Preview without executing | off |

### Camera path resolution

When `--render-type` is `camera-path` or `all`, camera paths are resolved in this order:
1. **`--camera-path`**: explicit single file (used for all runs)
2. **`--camera-paths-dir`**: all `*.json` in the directory (rendered per run)
3. **Auto-discovery**: reads the run's `config.yml` to find the dataset path, then looks for `camera_paths/*.json` in the dataset group directory (e.g. `datasets/saltpond/camera_paths/`)

### Integration with cluster pipeline

Rendering runs on the cluster via `cluster/jobs/render.pbs`, either as part of the full pipeline or standalone:

```bash
# Full pipeline: train → eval → render
./cluster/scripts/submit.sh --render

# Standalone render job (e.g. with render-specific args)
qsub -v EXTRA_ARGS="--render-type all --filter torpedo" cluster/jobs/render.pbs
```

Rendered videos sync automatically with `./cluster/scripts/sync_results.sh` (no extra flags needed).

### Dependencies

- `render.py` (imported directly — `render_dataset`, `render_camera_path`, `check_prerequisites`)
- `eval_experiments.py` (`resolve_runs`, `resolve_runs_from_config`, `validate_run`)
- `read_config.py` (`resolve_outputs_dir`)
- `experiments/run_experiments.py` (`load_config`, for config mode)
- `ns-render` must be on `PATH` (nerfstudio installed)
- `ffmpeg` must be on `PATH` (for dataset mode or `--keep-frames`)

---

## compare_renders.py

Extracts frames from rendered MP4 videos and produces visual comparisons across experiments. Four subcommands support the inspection workflow: check what's available (`info`), pull out frames (`extract`), compare the same frame across experiments (`compare`), and build a full experiment×output matrix (`grid`).

### Usage

```bash
# Show available renders for an experiment
python scripts/compare_renders.py info seathru8k --outputs-dir ../fyp-playground/outputs

# Extract specific frames as PNGs
python scripts/compare_renders.py extract seathru8k --outputs-dir ../fyp-playground/outputs \
  --frames 0 12 --output-types clean_rgb medium_rgb

# Cross-experiment comparison strips (vertical stacking)
python scripts/compare_renders.py compare seathru8k baseline seathru5k \
  --outputs-dir ../fyp-playground/outputs --frames 0 12 --output-types clean_rgb medium_rgb

# Full matrix: experiments (rows) x output types (columns)
python scripts/compare_renders.py grid seathru8k baseline \
  --outputs-dir ../fyp-playground/outputs --frames 0 --output-types clean_rgb medium_rgb depth

# Camera-path renders instead of dataset
python scripts/compare_renders.py info seathru8k --render-type camera-path --camera-path-name 1
```

### Arguments

**Shared (all subcommands):**

| Flag | Description | Default |
|------|-------------|---------|
| `experiments` (positional) | Experiment specs (substring matching) | required |
| `--outputs-dir <path>` | Base outputs directory | `$NERFSTUDIO_OUTPUTS` or `./outputs` |
| `--split <split>` | Dataset split | `test` |
| `--render-type {dataset,camera-path}` | Render type | `dataset` |
| `--camera-path-name <name>` | Camera path name (camera-path mode) | `1` |
| `--output-types <types...>` | Output types to process | `clean_rgb medium_rgb` |
| `--output-dir <path>` | Where to save results | `./comparisons` |
| `--max-width <pixels>` | Max image width before downscaling | no limit |

**`extract`, `compare`, `grid`:**

| Flag | Description | Default |
|------|-------------|---------|
| `--frames <indices...>` | Frame indices to extract | `0` |

### Output structure

```
comparisons/
├── extract/{experiment}/{output_type}_frame{NNN}.png
├── compare/compare_{output_type}_frame{NNN}.png
└── grid/grid_frame{NNN}.png
```

### Dependencies

- `cv2` (OpenCV) — frame extraction from MP4
- `PIL` (Pillow) — image composition and text labels
- `read_config.py` — path resolution

---

## change_config_path.py

Rewrites hardcoded absolute paths in nerfstudio `config.yml` files. Nerfstudio serializes `pathlib.PosixPath` objects in its configs, which break when moving outputs between machines. This script replaces path prefixes so configs can be loaded on a different machine.

### Usage

```bash
# Replace /home/saber with /home/alice
python scripts/change_config_path.py config.yml --old-base /home/saber --new-base /home/alice

# Auto-detect new base as $HOME
python scripts/change_config_path.py config.yml --old-base /home/saber

# Also remap data paths separately
python scripts/change_config_path.py config.yml --old-base /home/saber \
    --old-data /home/islabella/workspaces/irwin_ws/fyp-playground/datasets \
    --new-data /home/alice/datasets

# Create a .bak backup before editing
python scripts/change_config_path.py config.yml --old-base /home/saber --backup
```

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `config` | Path to the nerfstudio `config.yml` | required |
| `--old-base <path>` | Old base path prefix to replace | required |
| `--new-base <path>` | New base path prefix | `$HOME` |
| `--old-data <path>` | Old data path prefix (optional separate mapping) | none |
| `--new-data <path>` | New data path prefix | none |
| `--backup` | Create a `.bak` backup before editing | off |

### How it works

Nerfstudio serializes paths as YAML sequences:
```yaml
!!python/object/apply:pathlib.PosixPath
- /
- home
- saber
- workspaces
```

The script matches these `- component` line sequences and replaces them with the new path components.

### Dependencies

- Python standard library only

---

## read_tb.py

Reads and analyzes TensorBoard event files from nerfstudio experiment runs. Extracts per-step training data (~3000 scalar data points per tag at 30K steps logged every 10) to diagnose training dynamics: loss component dominance, PSNR trajectory, SeaThru/gray world activation stability, convergence assessment, medium parameter evolution, and per-phase training health.

### Usage

```bash
# Per-experiment training summary (includes per-phase assessment)
python scripts/read_tb.py summary tune02_bg003 --outputs-dir ../fyp-playground/outputs

# Summary in JSON format
python scripts/read_tb.py summary tune02_bg003 --json --outputs-dir ../fyp-playground/outputs

# Compact comparison (default — human-readable, eval metrics + key signals)
python scripts/read_tb.py compare tune02_bg003 tune02_bg005 tune02_gw005 tune02_gw020 \
  --outputs-dir ../fyp-playground/outputs

# Compact comparison with observations (threshold-based textual analysis)
python scripts/read_tb.py compare tune02_bg003 tune02_bg005 --describe \
  --outputs-dir ../fyp-playground/outputs

# Full verbose table (all ~50 rows, for agent/script consumption)
python scripts/read_tb.py compare tune02_bg003 tune02_bg005 --verbose \
  --outputs-dir ../fyp-playground/outputs

# Verbose + observations combined
python scripts/read_tb.py compare tune02_bg003 tune02_bg005 --verbose --describe

# Comparison in JSON format
python scripts/read_tb.py compare tune02_bg003 tune02_bg005 --json

# Use a larger averaging window (2000 steps instead of default 1000)
python scripts/read_tb.py compare tune02_bg003 tune02_bg005 --window 2000

# Custom convergence threshold (stricter: 2% instead of default 5%)
python scripts/read_tb.py summary tune02_bg003 --converge-threshold 0.02

# Custom recovery factor (tighter: 5% instead of default 10%)
python scripts/read_tb.py summary tune02_bg003 --recovery-factor 1.05

# Export raw scalars to CSV
python scripts/read_tb.py export tune02_bg003 --outputs-dir ../fyp-playground/outputs --format csv

# Export specific tags to JSON
python scripts/read_tb.py export tune02_bg003 --format json --tags "psnr" "main_loss"
```

### Subcommands

**`summary`** — Per-experiment training summary including:
- Loss components (final window average)
- PSNR trajectory (final + peak)
- Phase transitions (SeaThru spike magnitude + recovery step)
- Convergence assessment (CONVERGED / STILL_IMPROVING / DIVERGING)
- Medium parameter values (B_inf, learned_bg)
- **Per-phase assessment**: convergence, PSNR, and loss at each training phase checkpoint (Phase 1: Vanilla 3DGS → Phase 2: Transition → Phase 3: Joint Optimization)
- **Phase 3 per-component convergence**: which individual losses are still improving or diverging

**`compare`** — Side-by-side comparison across experiments with three output modes:
- **Compact** (default): Human-readable format showing eval metrics (from `metrics.json`), training summary, top-3 loss budget, and medium parameters. Includes right-aligned annotations flagging concerning values (slow recovery, declining PSNR, dominant losses, implausible B_inf).
- **Verbose** (`--verbose`): Full ~50-row table with all metrics grouped by category (training, PSNR, loss components, medium, phase transitions, config, per-phase). Used by `analyze_batch.py` and other scripts.
- **Describe** (`--describe`): Appends threshold-based textual observations after either compact or verbose output. Flags: STILL_IMPROVING experiments worth extending, slow Phase 2 recovery, declining Phase 3 PSNR, dominant loss components (>50%), implausible B_inf channels (>0.5), concerning/critical spikes.

Loads one experiment at a time to manage memory.

**`export`** — Dump raw scalar time-series to CSV or JSON for external tools. Supports tag filtering by substring.

### Arguments

**Shared (all subcommands):**

| Flag | Description | Default |
|------|-------------|---------|
| `--outputs-dir <path>` | Base outputs directory | `$NERFSTUDIO_OUTPUTS` or `./outputs` |
| `--window <N>` | Number of final steps to average over | `1000` |
| `--converge-threshold <float>` | Relative slope threshold for convergence classification: > threshold = DIVERGING, < -threshold = STILL_IMPROVING | `0.05` |
| `--recovery-factor <float>` | Factor of pre-activation loss baseline that defines recovery (1.1 = within 10%) | `1.1` |
| `--transition-estimate <int>` | Fallback Phase 2 duration estimate (iters) when recovery step is not detected | `3000` |

**`summary`:**

| Flag | Description | Default |
|------|-------------|---------|
| `paths` (positional) | Experiment path specs (timestamp dir, method dir, or substring) | required |
| `--json` | Output JSON instead of human-readable text | off |

**`compare`:**

| Flag | Description | Default |
|------|-------------|---------|
| `paths` (positional) | Experiment path specs to compare | required |
| `--json` | Output JSON instead of human-readable table | off |
| `--verbose` | Show full comparison table (all metrics, all rows) | off (compact) |
| `--describe` | Append textual observations (combinable with default or `--verbose`) | off |

**`export`:**

| Flag | Description | Default |
|------|-------------|---------|
| `path` (positional) | Single experiment path spec | required |
| `--format {csv,json}` | Output format | `csv` |
| `--tags <substrings...>` | Filter to tags containing these substrings | all tags |

### Per-phase assessment

Training is divided into three phases based on config and data:
- **Phase 1 (Vanilla 3DGS)**: `[0, seathru_from_iter)` — base geometry before medium activation
- **Phase 2 (Transition)**: `[seathru_from_iter, recovery_step)` — medium warm-up and GS adaptation (boundary is data-driven via recovery detection, fallback: `--transition-estimate`)
- **Phase 3 (Joint)**: `[recovery_step, max_num_iterations]` — full system convergence

Each phase reports: convergence status, PSNR start->end (peak), loss start->end. Phase 2 additionally reports spike ratio and recovery duration. Phase 3 reports per-component loss convergence.

### Output format

**Summary** output includes sections for training overview, PSNR, loss components, medium parameters, phase transitions, config phases, and per-phase assessment.

**Compare (compact, default):**
```
                         tune10_gw05  tune10_dcp02_35k
Eval Metrics
  PSNR                        29.01             25.79
  SSIM                        0.846             0.829
  ...
Training Summary
  Convergence       STILL_IMPROVING         CONVERGED
  Phase 2 spike               2.67x             2.81x    healthy
  Phase 3 PSNR trend       +2.25 dB          -1.11 dB    tune10_dcp02_35k: declining
  ...
Loss Budget (Phase 3 final)
  main_loss             48% (0.028)       49% (0.088)
  dcp                   27% (0.016)       42% (0.074)
  ...
```

**Compare (verbose, `--verbose`):** Column-oriented table grouped by category with ~50 rows:
```
Metric                       exp_a/ts1    exp_b/ts2
--------------------------------------------------
  [Training]
  total_steps                   30,000       30,000
  ...
  [Per-Phase Assessment]
  phase3_joint/convergence   CONVERGED  STILL_IMPROVING
  ...
```

**Observations (`--describe`):** Appended after either format:
```
Observations:
  - tune10_gw05: STILL_IMPROVING at 29,990 — Phase 3 gaining +2.25 dB, consider extending
  - tune10_dcp02_35k: Phase 3 PSNR declining (-1.11 dB) despite CONVERGED status
```

**JSON** (`--json`): list of `{"label": "...", "summary": {...}}` objects with all computed metrics.

**Export** CSV format: `tag,step,value` rows for all scalar events.

### Dependencies

- `tensorboard` (`EventAccumulator` for parsing event files)
- `numpy` (windowed averages, linear regression)
- `read_config.py` (`resolve_outputs_dir`, `load_config`)
- `eval_experiments.py` (`resolve_runs`)

---

## dataset_quality.py

Per-frame image quality assessment for dataset directories. Computes blur (Laplacian variance) and brightness (mean grayscale intensity) for every frame, flags outliers against configurable thresholds, and prints a summary report with per-frame details.

### Usage

```bash
# Basic quality check on a dataset
python scripts/dataset_quality.py /path/to/images/

# Custom thresholds
python scripts/dataset_quality.py /path/to/images/ --blur-threshold 80 --bright-low 30 --bright-high 230

# Sort by blur score to find worst frames
python scripts/dataset_quality.py /path/to/images/ --sort blur

# Save report to file
python scripts/dataset_quality.py /path/to/images/ -o quality_report.txt
```

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `image_dir` (positional) | Directory containing `.png` / `.jpg` images | required |
| `--blur-threshold <float>` | Laplacian variance below this = blurry | `100.0` |
| `--bright-low <float>` | Mean brightness below this = underexposed | `40.0` |
| `--bright-high <float>` | Mean brightness above this = overexposed | `220.0` |
| `--sort {name,blur,brightness}` | Sort order for per-frame table | `name` |
| `-o, --output <file>` | Write report to file instead of stdout | stdout |

### Output

Report includes:
- Directory path, frame count, resolution
- Summary statistics (mean, median, std, min, max) for blur and brightness
- Outlier counts by category
- Per-frame table with blur score, brightness, and flags (`BLUR`, `DARK`, `BRIGHT`)

### Dependencies

- `cv2` (OpenCV) — image loading, grayscale conversion, Laplacian
- `numpy` — summary statistics

---

## dataset_depth.py

Depth range statistics from COLMAP sparse reconstructions. Computes per-camera and global depth ranges from 3D point clouds. Supports two input modes: COLMAP binary (track-based, accurate per-camera depths) and nerfstudio transforms.json (approximate, distances to all points). Auto-detects input format from the given path.

### Usage

```bash
# COLMAP binary mode (track-based — preferred)
python scripts/dataset_depth.py /path/to/colmap/sparse/0/

# transforms.json mode (approximate)
python scripts/dataset_depth.py /path/to/transforms.json

# Auto-detect from dataset directory
python scripts/dataset_depth.py /path/to/dataset/

# Custom sort and histogram bins
python scripts/dataset_depth.py /path/to/data --sort far --bins 30

# Compact output (no per-camera table)
python scripts/dataset_depth.py /path/to/data --no-per-camera

# Save report to file
python scripts/dataset_depth.py /path/to/data -o depth_report.txt
```

### Input auto-detection

The script auto-detects the input format from the path:

| Priority | Condition | Mode |
|----------|-----------|------|
| 1 | Path contains `images.bin` + `points3D.bin` | COLMAP binary (track-based) |
| 2 | Path is `transforms.json` | transforms.json (approximate) |
| 3 | Directory with `colmap/sparse/0/` subdirectory | COLMAP binary (walk) |
| 4 | Directory with `transforms.json` | transforms.json (walk) |

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `input_path` (positional) | COLMAP sparse dir, transforms.json, or parent dataset dir | required |
| `--bins <int>` | Histogram bin count | `20` |
| `--sort {name,near,far,median,range}` | Per-camera table sort order | `name` |
| `--no-per-camera` | Omit per-camera table | off |
| `-o, --output <file>` | Write report to file instead of stdout | stdout |
| `--shallow-pct <int>` | Percentile threshold for SHALLOW flag | `5` |
| `--deep-pct <int>` | Percentile threshold for DEEP flag | `95` |
| `--narrow-factor <float>` | Factor of median range below which = NARROW | `0.5` |
| `--wide-factor <float>` | Factor of median range above which = WIDE | `2.0` |

### Output

Report includes:
- Scene overview (camera count, point count, input mode)
- Global depth statistics (min, max, mean, median, std, IQR, dynamic range, percentiles)
- Flag thresholds (computed values for each flag, derived from the data and threshold parameters)
- Text histogram of depth distribution
- Per-camera table with near/far/median/range and flags (`SHALLOW`, `DEEP`, `NARROW`, `WIDE`)

### Dependencies

- `numpy` — distance computation, statistics
- Python standard library only (inline COLMAP binary and PLY readers)

---

## dataset_underwater.py

Underwater dataset characterization tool. Computes per-frame color cast, turbidity, and visibility metrics for underwater image datasets. Optionally correlates metrics with COLMAP depth and detects temporal appearance outliers.

### Usage

```bash
# Basic analysis
python scripts/dataset_underwater.py /path/to/images/

# Sort by UCIQE to find worst-quality frames
python scripts/dataset_underwater.py /path/to/images/ --sort uciqe

# JSON output to file
python scripts/dataset_underwater.py /path/to/images/ --json -o analysis.json

# Depth-color correlation (requires COLMAP data or transforms.json)
python scripts/dataset_underwater.py /path/to/images/ --depth-source /path/to/dataset/

# Inter-frame temporal variance analysis
python scripts/dataset_underwater.py /path/to/images/ --temporal --temporal-window 10

# Full analysis with all extensions
python scripts/dataset_underwater.py /path/to/images/ \
    --depth-source /path/to/dataset/ --temporal --json
```

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `image_dir` (positional) | Directory containing `.png` / `.jpg` images | required |
| `--sort {name,uciqe,uiqm,gw_dev,dcp}` | Sort order for per-frame table | `name` |
| `--dcp-patch-size <int>` | Dark channel prior patch size | `41` |
| `--canny-low <int>` | Canny edge detector low threshold | `50` |
| `--canny-high <int>` | Canny edge detector high threshold | `150` |
| `-o, --output <file>` | Write output to file instead of stdout | stdout |
| `--json` | Output JSON instead of text | off |
| `--depth-source <path>` | Path to COLMAP sparse dir, transforms.json, or dataset dir for depth-color correlation | off |
| `--depth-bins <int>` | Number of equal-count depth bins | `5` |
| `--temporal` | Enable inter-frame appearance variance analysis | off |
| `--temporal-window <int>` | Rolling window size for temporal stats | `5` |

### Metrics

**Color cast**:
- R/G and B/G channel ratios
- Gray-world deviation (max channel deviation from mean)
- CIELAB a\*/b\* (color direction: blue, green, red)

**Turbidity**:
- UCIQE — underwater color image quality (saturation std + luminance contrast + saturation mean)
- UIQM — underwater image quality measure (Panetta et al. 2016: colorfulness + sharpness + contrast)
- Dark channel prior — min-channel + erosion statistics

**Visibility**:
- RMS contrast — grayscale standard deviation normalized by mean
- Edge density — fraction of Canny edge pixels
- Mean luminance — average grayscale brightness [0, 1]

**Depth-color correlation** (with `--depth-source`):
- Pearson and Spearman correlations between camera depth and each color metric
- Per-bin average color metrics across equal-count depth bins
- Predicts whether the medium model's depth-uniform assumptions hold

**Temporal variance** (with `--temporal`):
- Frame-to-frame deltas for luminance, R/G ratio, B/G ratio
- Rolling window std over configurable window
- Outlier cluster detection (contiguous frames beyond 2 sigma)
- Predicts Phase 3 difficulty (high variance → ROV lighting → expected drift)

### Output

**Text** (default): Summary statistics → compact per-frame table → optional depth-color correlation → optional temporal analysis.

**JSON** (`--json`): `metadata` + `summary` + `per_frame` + optional `depth_correlation` + optional `temporal`.

### Dependencies

- `cv2` (OpenCV) — color conversion, Sobel, Canny, erode
- `numpy` — statistics, block operations
- `scipy` (conditional) — Pearson/Spearman correlations (only when `--depth-source` used)
- `dataset_depth.py` (import) — COLMAP/transforms.json depth loading (only when `--depth-source` used)

---

## decompose_metrics.py

Post-hoc decomposition of SSIM and LPIPS metrics into per-component scores. Computes SSIM luminance/contrast/structure (Wang 2004) and LPIPS per-layer distances from rendered test frames vs ground truth.

### Usage

```bash
# Decompose metrics for an experiment
python scripts/decompose_metrics.py <experiment_spec> --outputs-dir ../fyp-playground/outputs

# Compare decompositions across experiments
python scripts/decompose_metrics.py tune10_gw05 tune10_gw15 --outputs-dir ../fyp-playground/outputs

# JSON output with explicit dataset directory
python scripts/decompose_metrics.py <experiment_spec> --json \
    --outputs-dir ../fyp-playground/outputs \
    --dataset-dir ../fyp-playground/datasets/saltpond/saltpond_unprocessed

# Run on CPU
python scripts/decompose_metrics.py <experiment_spec> --device cpu --outputs-dir ../fyp-playground/outputs
```

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `specs` (positional) | Experiment path specs (substring, method dir, or timestamp dir) | required |
| `--outputs-dir <path>` | Base outputs directory | `$NERFSTUDIO_OUTPUTS` or `./outputs` |
| `--dataset-dir <path>` | Override GT dataset directory (when config.yml has stale remote paths) | from config.yml |
| `--json` | Output JSON instead of text | off |
| `--device <device>` | Compute device | `cuda` if available, else `cpu` |
| `--lpips-net {alex,vgg,squeeze}` | LPIPS backbone network | `alex` |

### What it does

1. Resolve experiment run directory and load `config.yml`
2. Determine test split indices from dataparser config (fraction/interval mode)
3. Load GT images from the dataset directory (respects downscale factor)
4. Extract rendered frames from `renders/dataset/test/medium_rgb.mp4`
5. Compute per-frame SSIM components (luminance, contrast, structure) and LPIPS per-layer distances
6. Aggregate and output results

### Notes / Caveats

- SSIM uses the Wang 2004 three-component formula (l*c*s), which differs slightly (~0.003) from pytorch_msssim's simplified formula (l*cs). Both are standard; the decomposition requires the three-component form.
- LPIPS per-layer distances sum exactly to the total LPIPS score.
- Requires `--dataset-dir` when config.yml contains paths from a remote machine (e.g., Vanda).
- LPIPS requires a neural network forward pass; CPU is slower but works without GPU.

### Dependencies

- `torch`, `pytorch_msssim` — SSIM computation
- `torchmetrics` — LPIPS per-layer extraction (`_LPIPS` internal class)
- `cv2` (OpenCV) — MP4 frame extraction, image loading
- `read_config.py`, `eval_experiments.py` — path resolution

---

## agents/analyze_batch.py

Agent infrastructure script — called by `/auto-experiment` and `/auto-analyze` workflows, not intended for direct human use.

Gathers all quantitative analysis for a batch of experiments into a single structured JSON report. Replaces ~25 individual tool calls during auto-analyze with one script invocation. Orchestrates existing scripts (`read_tb.py`, `compare_renders.py`, `dataset_underwater.py`) via subprocess calls and combines their outputs with `metrics.json` data and dataset input analysis.

### Usage

```bash
# Full analysis for a batch
python scripts/agents/analyze_batch.py tune10 \
    --outputs-dir ../fyp-playground/outputs \
    --analysis-dir /tmp/batch-analysis \
    --dataset-analysis ../fyp-playground/datasets/saltpond/analysis.md \
    --num-frames 3 \
    --output-types clean_rgb medium_rgb depth \
    --max-width 480

# Minimal (just metrics + TB, skip visual analysis)
python scripts/agents/analyze_batch.py tune10 \
    --outputs-dir ../fyp-playground/outputs \
    --analysis-dir /tmp/batch-analysis \
    --num-frames 0

# Default output types (clean_rgb, medium_rgb, depth, accumulation, backscatter, attenuation_map)
python scripts/agents/analyze_batch.py tune10 --outputs-dir ../fyp-playground/outputs
```

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `batch_prefix` (positional) | Batch prefix to match experiments (e.g. `tune10` matches `tune10_*`) | required |
| `--outputs-dir <path>` | Base outputs directory | `$NERFSTUDIO_OUTPUTS` or `./outputs` |
| `--analysis-dir <path>` | Directory for analysis artifacts (grids, renders, report) | `/tmp/batch-analysis` |
| `--dataset-analysis <path>` | Path to dataset `analysis.md` for input color metrics comparison | none |
| `--num-frames <N>` | Number of representative frames to extract and analyze | `3` |
| `--output-types <types...>` | Output types for comparison grids | `clean_rgb medium_rgb depth accumulation backscatter attenuation_map` |
| `--max-width <pixels>` | Max image width for renders | `480` |
| `--cleanup-tb` | Delete local tfevents files after TB data has been extracted | `false` |

### What it does (in order)

1. **Find experiments**: Glob for `<batch_prefix>_*` in the outputs directory
2. **Read metrics**: Load `metrics.json` (PSNR, SSIM, LPIPS, clean_psnr, etc.)
3. **TB analysis**: Run `read_tb.py compare` across all experiments
3b. **Cleanup TB** (if `--cleanup-tb`): Delete tfevents files now that metrics are in the report
4. **Render info**: Get total frame count via `compare_renders.py info`
5. **Pick frames**: Evenly space `--num-frames` frames across the render
6. **Comparison grids**: Generate experiment x output type matrices via `compare_renders.py grid`
7. **Extract renders**: Pull render frames for each experiment via `compare_renders.py extract`
8. **Color analysis**: Run `dataset_underwater.py --json` on extracted renders
9. **Dataset input metrics**: Parse `analysis.md` for input color cast baselines

### Output

Writes `report.json` to `--analysis-dir` with:
- `batch_prefix`, `experiments` (list of names)
- `metrics` — per-experiment eval metrics (PSNR, SSIM, LPIPS, clean_psnr, etc.)
- `tb_analysis` — structured TensorBoard summaries (list of `{"label", "summary"}` dicts from `read_tb.py compare --json`)
- `render_info` — total frame count and selected frame indices
- `grid_images` — paths to generated comparison grid PNGs
- `color_analysis` — per-experiment color metrics (R/G ratio, gray-world deviation, CIELAB, DCP)
- `dataset_input_metrics` — input dataset color metrics (if `--dataset-analysis` provided)

Also generates artifacts in `--analysis-dir`:
```
<analysis-dir>/
├── report.json
├── grids/grid/grid_frame{NNN}.png
└── renders/extract/{experiment}/clean_rgb_frame{NNN}.png
```

Prints progress to stderr and the report path to stdout (last line).

### Notes / Caveats

- Handles missing data gracefully — if TB files, renders, or metrics are missing, the corresponding fields are set to error messages, empty lists, or null
- TensorBoard loading is the main bottleneck (~30-60s for 4 experiments)
- Set `--num-frames 0` to skip all visual analysis (grids, extracts, color analysis)

### Dependencies

- `read_tb.py` (subprocess: TensorBoard comparison)
- `compare_renders.py` (subprocess: frame extraction and grid composition)
- `dataset_underwater.py` (subprocess: color analysis on rendered frames)
- Python standard library only (no direct third-party imports)

---

## paper_figures.py

Generate publication-quality figures from TensorBoard training data. Designed for paper preparation — produces consistent, reproducible figures with phase annotations, peak markers, and cross-experiment comparisons.

### Purpose

Fills the visualization gap between raw TensorBoard data and paper-ready figures. Wraps `read_tb.py` for data loading, adds matplotlib styling, and provides 9 figure types covering the full analysis needs of the underwater decomposition paper.

### Usage

```bash
# Single experiment — all standard figures
python scripts/paper_figures.py all <experiment> --outputs-dir <path>

# Specific figure type
python scripts/paper_figures.py psnr <experiment>
python scripts/paper_figures.py medium-params <experiment>
python scripts/paper_figures.py loss-components <experiment> --budget

# Cross-experiment comparison
python scripts/paper_figures.py cross-compare <exp1> <exp2> ... --metric psnr
python scripts/paper_figures.py psnr-gap <exp1> <exp2> ...
python scripts/paper_figures.py early-stopping <exp1> <exp2> ...
```

### Figure types

| Subcommand | Mode | What it shows |
|------------|------|---------------|
| `psnr` | Single | PSNR trajectory with EMA smoothing, phase boundaries, peak + decline |
| `medium-params` | Single | β_D + B_inf per channel (2-panel, plausibility band) |
| `loss-components` | Single | Per-loss curves; `--budget` for fraction-of-total view |
| `gaussian-count` | Single | Gaussian count with saturation point |
| `phase2-spike` | Single | Zoomed loss at medium onset (spike ratio, recovery) |
| `medium-activity` | Single | medium_contribution + attenuation/backscatter magnitude |
| `cross-compare` | Multi | Same metric overlaid for N experiments |
| `psnr-gap` | Multi | Bar chart: PSNR vs clean_PSNR gap (decomposition proxy) |
| `early-stopping` | Multi | Scatter of peak step vs peak PSNR (decline colormap) |
| `all` | Single | Generates all single-experiment figures + loss budget |

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--outputs-dir` | auto-detect | Base outputs directory |
| `--output-dir` | `../fyp-playground/paper/figures/` | Figure output directory |
| `--format` | `both` | Output format: `pdf`, `png`, or `both` |
| `--smooth` | `100` | EMA smoothing window (samples); `0` to disable |
| `--width` | `single` | Figure width: `single` (3.5") or `double` (7.0") |
| `--no-phase` | off | Suppress phase boundary annotations |
| `--label` | auto | Custom labels (comma-separated) |
| `--metric` | `psnr` | For `cross-compare`: which metric to overlay |
| `--budget` | off | For `loss-components`: show as fraction of total |

### Execution model

Designed to run on Vanda (where TB data lives) with artifacts synced back:
```bash
# On Vanda
python scripts/paper_figures.py all <experiment> \
    --outputs-dir /scratch/$USER/fyp-playground/outputs \
    --output-dir /scratch/$USER/fyp-playground/paper/figures/

# Sync back
scp -r vanda:/scratch/$USER/fyp-playground/paper/figures/ fyp-playground/paper/figures/
```

### Dependencies

- `matplotlib` — figure rendering
- `numpy` — data manipulation
- `read_tb.py`, `eval_experiments.py`, `read_config.py` — data loading and path resolution

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
| `dataset_underwater.py` | Underwater dataset characterization |

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
python scripts/render.py dataset <experiment> --split test --rendered-output-names rgb depth

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
| `--rendered-output-names <names...>` | Output names to render | `rgb` |
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
│   │   ├── rgb.mp4
│   │   └── depth.mp4
│   └── train/
│       └── rgb.mp4
└── camera-path/
    └── {path-name}/
        ├── rgb.mp4
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
python scripts/render_experiments.py a_exploration --rendered-output-names rgb depth --downscale-factor 2 \
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
| `--rendered-output-names <names...>` | Output names to render | `rgb` |
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
  --frames 0 12 --output-types rgb underwater_rgb

# Cross-experiment comparison strips (vertical stacking)
python scripts/compare_renders.py compare seathru8k baseline seathru5k \
  --outputs-dir ../fyp-playground/outputs --frames 0 12 --output-types rgb underwater_rgb

# Full matrix: experiments (rows) x output types (columns)
python scripts/compare_renders.py grid seathru8k baseline \
  --outputs-dir ../fyp-playground/outputs --frames 0 --output-types rgb underwater_rgb depth

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
| `--output-types <types...>` | Output types to process | `rgb underwater_rgb` |
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

# Side-by-side comparison table across experiments
python scripts/read_tb.py compare tune02_bg003 tune02_bg005 tune02_gw005 tune02_gw020 \
  --outputs-dir ../fyp-playground/outputs

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

**`compare`** — Side-by-side comparison table with one column per experiment and rows for each metric. Includes per-phase assessment rows. Designed for wave-level analysis. Loads one experiment at a time to manage memory.

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

**`summary`, `compare`:**

| Flag | Description | Default |
|------|-------------|---------|
| `paths` (positional) | Experiment path specs (timestamp dir, method dir, or substring) | required |
| `--json` | Output JSON instead of human-readable text/table | off |

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

Each phase reports: convergence status, PSNR start→end (peak), loss start→end. Phase 2 additionally reports spike ratio and recovery duration. Phase 3 reports per-component loss convergence.

### Output format

**Summary** output includes sections for training overview, PSNR, loss components, medium parameters, phase transitions, config phases, and per-phase assessment.

**Compare** output is a column-oriented table grouped by category (including per-phase rows):
```
Metric                       exp_a/ts1    exp_b/ts2
--------------------------------------------------
  [Training]
  total_steps                   30,000       30,000
  convergence                CONVERGED  STILL_IMPROVING

  [Per-Phase Assessment]
  phase1_vanilla/convergence CONVERGED    CONVERGED
  phase1_vanilla/psnr_end        27.8         27.5
  phase3_joint/convergence   CONVERGED  STILL_IMPROVING
  phase3_joint/psnr_peak         28.0         27.2
  ...
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

Underwater dataset characterization tool. Computes per-frame color cast, turbidity, and visibility metrics for underwater image datasets.

### Usage

```bash
# Basic analysis
python scripts/dataset_underwater.py /path/to/images/

# Sort by UCIQE to find worst-quality frames
python scripts/dataset_underwater.py /path/to/images/ --sort uciqe

# JSON output to file
python scripts/dataset_underwater.py /path/to/images/ --json -o analysis.json

# Custom dark channel patch size
python scripts/dataset_underwater.py /path/to/images/ --dcp-patch-size 21
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

### Output

**Text** (default): Summary statistics → compact per-frame table (R/G, B/G, GW_Dev, UCIQE, UIQM, DCP_Mean).

**JSON** (`--json`): `metadata` + `summary` (all metrics with stats) + `per_frame` (all metrics per frame).

### Dependencies

- `cv2` (OpenCV) — color conversion, Sobel, Canny, erode
- `numpy` — statistics, block operations

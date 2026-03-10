# Scripts

Utility scripts for managing nerfstudio experiment workflows — from running batch training to comparing configs across runs.

## Overview

| Script | Purpose |
|--------|---------|
| `experiments/run_experiments.py` | Batch-run nerfstudio training experiments |
| `experiments/experiment_config.py` | Define experiment matrices (datasets x templates x models x repeats) |
| `experiments/local_config.example.py` | Template for machine-specific settings (copy to `local_config.py`) |
| `read_config.py` | Read and diff nerfstudio experiment configs |
| `log_experiments.py` | Generate experiment logs with config diffs against a baseline |
| `eval_experiments.py` | Batch-run `ns-eval` and save metrics to experiment directories |
| `render.py` | Render experiments to video (wraps `ns-render` + ffmpeg) |
| `change_config_path.py` | Rewrite hardcoded paths in nerfstudio configs for cross-machine use |

### Script relationships

```
local_config.py ──> experiment_config.py ──> run_experiments.py
                    (local settings)         (config defines experiments, runner executes them)

read_config.py ──> log_experiments.py       (log generator uses reader to load/compare configs)
read_config.py ──> eval_experiments.py      (evaluator uses reader + log_experiments for path resolution)
log_experiments.py ──> eval_experiments.py
experiment_config.py ──> eval_experiments.py  (config mode uses experiment config for run discovery)
read_config.py ──> render.py               (renderer uses reader for path resolution)
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
| `--config <path>` | Path to config `.py` file or module name | `experiment_config` |
| `--filter <substring>` | Only run experiments whose name contains this substring | none |
| `--log-index` | Prefix `expt_{index}_` to log filenames | off |

### Config module requirements

The config module must export:
- `EXPERIMENTS` — list of experiment dicts (see `experiment_config.py` for the schema)
- `LOG_DIR` — path to directory for log files

### Dependencies

- `ns-train` must be on `PATH` (nerfstudio installed)
- Python standard library only

---

## experiments/experiment_config.py

Defines the experiment matrix for `run_experiments.py`. Configures datasets, experiment templates (hyperparameter variations), models, and repeat counts, then generates the full `EXPERIMENTS` list as a cartesian product. Machine-specific settings are imported from `local_config.py`.

### Structure

```python
# Required from local_config.py:
WORKSPACE_DIR          # Base path to fyp-playground
DATASETS               # Dict mapping dataset names to paths
EXPERIMENT_TEMPLATES   # List of {suffix, extra_args} dicts

# Optional (overridable in local_config.py):
OUTPUT_DIR             # Where nerfstudio writes outputs (default: WORKSPACE_DIR/outputs)
LOG_DIR                # Where runner writes logs (default: WORKSPACE_DIR/logs)
MODELS                 # List of model names (default: ["sea-splatfacto"])
NUMBER_OF_REPEATS      # How many times to repeat each combination (default: 1)

# Auto-generated:
EXPERIMENTS            # datasets x templates x repeats x models
```

### Customization

- Add datasets to `DATASETS` in `local_config.py`
- Add experiment variants to `EXPERIMENT_TEMPLATES` with custom `extra_args` (these become `ns-train` CLI flags)
- Override `OUTPUT_DIR`, `LOG_DIR`, `MODELS`, or `NUMBER_OF_REPEATS` in `local_config.py`
- Use `--filter` on the runner to select subsets at runtime

---

## experiments/local_config.example.py

Template for machine-specific settings. Copy to `local_config.py` and edit:

```bash
cp scripts/experiments/local_config.example.py scripts/experiments/local_config.py
```

### Settings

| Variable | Required | Description |
|----------|----------|-------------|
| `WORKSPACE_DIR` | yes | Base path to `fyp-playground` |
| `DATASETS` | yes | Dict mapping dataset names to paths |
| `EXPERIMENT_TEMPLATES` | yes | List of `{suffix, extra_args}` dicts |
| `OUTPUT_DIR` | no | Override default output directory (`WORKSPACE_DIR/outputs`) |
| `LOG_DIR` | no | Override default log directory (`WORKSPACE_DIR/logs`) |
| `MODELS` | no | Override default model list (`["sea-splatfacto"]`) |
| `NUMBER_OF_REPEATS` | no | Override default repeat count (`1`) |

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
| `--config <path>` | Path to config `.py` file or module name (config mode) | `experiment_config` |
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
- `experiments/experiment_config.py` + `experiments/local_config.py` (config mode)

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

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
| `change_config_path.py` | Rewrite hardcoded paths in nerfstudio configs for cross-machine use |

### Script relationships

```
local_config.py ──> experiment_config.py ──> run_experiments.py
                    (local settings)         (config defines experiments, runner executes them)

read_config.py ──> log_experiments.py       (log generator uses reader to load/compare configs)
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

# fyp-utils

Utility scripts and environment configs for nerfstudio-based experiments. Handles the workflow around training runs — from environment setup and batch execution to config comparison and output visualization.

## Structure

```
fyp-utils/
├── config/           # Experiment configuration (machine-specific settings)
├── scripts/          # CLI tools for running and analyzing experiments
├── environments/     # Environment setup (conda)
│   └── nerfstudio/
└── cluster/          # HPC deployment (Apptainer, PBS jobs, sync)
```

See each directory's README for detailed usage:

- **[scripts/README.md](scripts/README.md)** — experiment runner, config reader/differ, experiment logger, path fixer, render pipeline
- **[environments/nerfstudio/README.md](environments/nerfstudio/README.md)** — conda environment setup
- **[cluster/README.md](cluster/README.md)** — Vanda HPC deployment (Apptainer container, PBS jobs, result sync)

## Nerfstudio Fork

This project uses a patched nerfstudio fork at [`ikxm17/nerfstudio`](https://github.com/ikxm17/nerfstudio.git) (branch `fix/fork-patches`). The fork carries patches that fix compatibility issues not yet merged upstream:

- **Lazy exporter imports** — heavy top-level imports (open3d, pymeshlab, gsplat, etc.) moved to method bodies so CLI completions don't require all optional packages
- **COLMAP version guard** — `--SiftExtraction.use_gpu` / `--SiftMatching.use_gpu` flags only passed for COLMAP < 3.9 (removed in 3.9+); vocab tree path only passed for COLMAP < 3.11 (uses faiss format in 3.11+)

`setup_env.sh` handles cloning and updating this fork automatically. To clone manually:

```bash
git clone --branch fix/fork-patches https://github.com/ikxm17/nerfstudio.git
```

## Configuration

Experiment configuration uses a two-layer system:

- **`config/experiment_config.py`** — shared orchestration layer. Imports machine-specific settings from `local_config.py`, applies defaults, and builds the `EXPERIMENTS` list as a cartesian product of datasets × templates × repeats × models.
- **`config/local_config.py`** — machine-specific settings (git-ignored). Each machine maintains its own copy.

Two example templates are provided:

| Template | Use case |
|----------|----------|
| `config/local_config.example.py` | Local / remote GPU machines |
| `config/local_config.cluster.example.py` | Vanda HPC cluster |

### Required settings

| Setting | Description |
|---------|-------------|
| `WORKSPACE_DIR` | Path to the `fyp-playground` directory |
| `DATASETS` | Dict mapping dataset names to paths |
| `EXPERIMENT_TEMPLATES` | List of template dicts with `suffix` and `extra_args` |

### Optional overrides

| Setting | Default |
|---------|---------|
| `OUTPUT_DIR` | `WORKSPACE_DIR + "/outputs"` |
| `LOG_DIR` | `WORKSPACE_DIR + "/logs"` |
| `MODELS` | `["sea-splatfacto"]` |
| `NUMBER_OF_REPEATS` | `1` |
| `VIS` | `"viewer+tensorboard"` |
| `VIEWER` | `False` |
| `RENDER_OUTPUT_NAMES` | `["rgb"]` |
| `RENDER_TYPE` | `"dataset"` |
| `RENDER_SPLIT` | `"train+test"` |

## Recommended Directory Structure

For non-cluster experiment runs (local or remote GPU machines), place all repos and data under a common parent:

```
~/workspace/fyp/
├── fyp-utils/                     # This repo (scripts + config)
├── fyp-playground/                # Experiment workspace (WORKSPACE_DIR in local_config.py)
│   ├── datasets/                  # Training data
│   │   ├── torpedo/
│   │   │   └── torpedo_unprocessed/
│   │   └── saltpond/
│   │       ├── saltpond_unprocessed/
│   │       └── camera_paths/      # For camera-path renders
│   ├── outputs/                   # Training outputs (auto-created by ns-train)
│   └── logs/                      # Training logs
├── sea-splatfacto/                # sea-splatfacto source (installed editable)
└── nerfstudio/                    # Nerfstudio fork (auto-detected by setup_env.sh)
```

- `setup_env.sh` auto-detects `nerfstudio` and `sea-splatfacto` at `<project-root>/../nerfstudio/` and `<project-root>/../sea-splatfacto/` respectively; override with `--nerfstudio <path>` or `--sea-splatfacto <path>`
- `WORKSPACE_DIR` in `local_config.py` should point to the `fyp-playground/` directory

## Quick Start

```bash
# 1. Clone the nerfstudio fork (if not using setup_env.sh to clone automatically)
git clone --branch fix/fork-patches https://github.com/ikxm17/nerfstudio.git ../nerfstudio

# 2. Set up the environment
./environments/nerfstudio/setup_env.sh --platform cpu|cu118|cu121

# 3. Activate
conda activate nerfstudio

# 4. Configure experiments
cp config/local_config.example.py config/local_config.py
#    Edit config/local_config.py with your WORKSPACE_DIR

# 5. Preview experiment commands
python scripts/experiments/run_experiments.py --dry-run

# 6. Run experiments
python scripts/experiments/run_experiments.py

# 7. Compare configs across runs
python scripts/read_config.py diff <run-a> <run-b>

# 8. Generate experiment log
python scripts/log_experiments.py /path/to/method-dir -o log.txt
```

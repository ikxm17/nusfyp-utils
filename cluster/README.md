# cluster/

HPC deployment files for running nerfstudio training on the Vanda PBS cluster (NUS, `vanda.nus.edu.sg`).

## Directory Structure

| File | Purpose |
|------|---------|
| `nerfstudio.def` | Apptainer container definition (CUDA 11.8 + nerfstudio + sea-splatfacto) |
| `jobs/train.pbs` | PBS job script for training (`run_experiments.py`) |
| `jobs/train_array.pbs` | PBS array job script — one experiment per sub-job |
| `jobs/eval.pbs` | PBS job script for evaluation + checkpoint cleanup |
| `jobs/render.pbs` | PBS job script for rendering experiments to video |
| `jobs/analyze.pbs` | PBS job script for batch analysis (CPU-only, no GPU) |
| `scripts/submit.sh` | Convenience wrapper to submit train→eval→render→analyze dependency chain |
| `scripts/sync_results.sh` | **Runs locally** — rsync results + logs from cluster, rewrite config paths |
| `scripts/sync_analysis.sh` | **Runs locally** — rsync only analysis artifacts (report.json + grids) |

## Workspace Layout

The PBS job scripts hardcode paths that assume the directory structure below.

### Cluster layout (assumed by PBS jobs)

```
$HOME/workspace/fyp/
├── fyp-utils/                     # This repo
├── sea-splatfacto/                # Bind-mounted into container
├── nerfstudio/                    # Bind-mounted into container
└── containers/
    └── nerfstudio.sif -> /scratch/$USER/fyp-playground/containers/nerfstudio.sif

/scratch/$USER/fyp-playground/
├── containers/
│   └── nerfstudio.sif             # Actual container image (~8.4 GB)
├── datasets/                      # Training data
├── outputs/                       # Training outputs
└── logs/                          # Job + training logs
```

### Local layout (assumed by `sync_results.sh`)

```
~/workspace/fyp/
├── fyp-utils/                     # This repo
└── fyp-playground/                # Sibling directory — sync target
    ├── datasets/
    ├── outputs/                   # Synced from cluster
    └── logs/                      # Synced from cluster
```

## Setup

### 1. Build the container

Build from the parent workspace directory so relative `%files` paths resolve:

```bash
cd ~/workspace/fyp
mkdir -p containers
apptainer build containers/nerfstudio.sif fyp-utils/cluster/nerfstudio.def
```

Takes ~15-30 min (CUDA kernel compilation). No GPU required — `nvcc` comes from the base image.

### 2. Transfer datasets

```bash
# Run on the cluster
rsync -avz local_machine:~/workspace/fyp/fyp-playground/datasets/ /scratch/$USER/fyp-playground/datasets/
```

### 3. Configure local_config

```bash
cp config/local_config.cluster.example.py config/local_config.py
# Edit if needed (datasets, templates, etc.)
```

Each machine maintains its own `config/local_config.py` in its git checkout.

## Usage

### Submit training + eval (sequential)

**IMPORTANT**: Always use `--dataset` to scope jobs to the intended dataset. Without it, experiments run across ALL datasets in `local_config.py`. Use `--filter` to further narrow by experiment name within that dataset.

```bash
cd ~/workspace/fyp/fyp-utils

# Submit for a specific dataset (recommended: always use --dataset and --render)
./cluster/scripts/submit.sh --render --dataset redsea_unprocessed

# Dataset + filter to run a single experiment
./cluster/scripts/submit.sh --render --dataset redsea_unprocessed --filter tune00_dcp_low_st10k

# Training only
./cluster/scripts/submit.sh --train-only --dataset redsea_unprocessed

# Check job status
qstat -u $USER
```

### Paid queue / walltime override

Vanda charges GPU-hours based on **requested walltime**, not actual elapsed time. All jobs default to the `auto_free` queue to prevent accidental allocation burn. Use `--paid` to explicitly use the paid queue (higher priority, consumes allocation).

```bash
# Training on paid queue (consumes GPU-hour allocation, higher priority)
./cluster/scripts/submit.sh --paid

# Override walltime (e.g. after profiling actual runtime)
./cluster/scripts/submit.sh --paid --walltime 2:00:00

# Free tier is the default — no flag needed
./cluster/scripts/submit.sh --render --filter torpedo
```

### Submit training in parallel (array jobs)

Each experiment runs as a separate PBS sub-job with its own GPU:

```bash
# Submit all experiments in parallel (1 GPU per experiment)
./cluster/scripts/submit.sh --parallel

# Parallel + filter
./cluster/scripts/submit.sh --parallel --filter torpedo

# Parallel, training only
./cluster/scripts/submit.sh --parallel --train-only
```

PBS auto-schedules sub-jobs within the cluster's concurrent job limit (max 4 concurrent jobs).

### Submit render job standalone

For render-specific options (e.g. `--render-type all`), submit `render.pbs` directly:

```bash
# Render all types (dataset + camera-path) for a specific dataset
qsub -v EXTRA_ARGS="--render-type all --filter torpedo" cluster/jobs/render.pbs
```

### Sync results locally

Run on your **local machine**. Rendered videos from the cluster are synced automatically under `outputs/**/renders/`.

```bash
cd ~/workspace/fyp/fyp-utils

# Sync outputs + logs (excludes checkpoints + tensorboard events)
./cluster/scripts/sync_results.sh

# Include checkpoints (for inspecting models or ad-hoc local renders)
./cluster/scripts/sync_results.sh --include-checkpoints

# Delete synced files from remote after successful sync
./cluster/scripts/sync_results.sh --cleanup
```

## How Editability Works

The container bakes compiled dependencies (PyTorch, CUDA, tiny-cuda-nn) and editable-install metadata (entry points, `.dist-info`). At runtime, PBS scripts bind-mount the live source trees:

```
--bind ~/workspace/fyp/sea-splatfacto:/opt/sea-splatfacto
--bind ~/workspace/fyp/nerfstudio:/opt/nerfstudio
```

Python resolves source through the editable install pointers to `/opt/sea-splatfacto` and `/opt/nerfstudio`, which now point to the host's live code. Code changes take effect immediately — only structural changes (new entry points, packages, or dependencies) require a container rebuild.

`config/local_config.py` is in `$HOME`, which Apptainer auto-binds.

## Resource Requests

| Job | GPUs | CPUs | Memory | Walltime |
|-----|------|------|--------|----------|
| `train.pbs` / `train_array.pbs` | 1x A40 | 36 | 128GB | 12h |
| `eval.pbs` | 1x A40 | 2 | 16GB | 4h |
| `render.pbs` | 1x A40 | 2 | 16GB | 4h |
| `analyze.pbs` | none | 4 | 8GB | 1h |

Queue limits: max walltime 12h, max 2x A40 GPUs per job, max 72 CPUs per node, max 4 concurrent jobs.

> **Tip:** Training defaults to `auto_free` (no GPU-hour charge). Use `--paid --walltime <HH:MM:SS>` for production runs after profiling actual runtime.

## Troubleshooting

### `qsub: copy of script to tmp failed on close: No space left on device`

The management node's root filesystem (`/dev/sda3`) is full. `qsub` can't write temp files to `/tmp`. `submit.sh` automatically falls back to `TMPDIR=/scratch/$USER/tmp` when this happens. If submitting directly via `qsub`, set it manually:

```bash
TMPDIR=/scratch/$USER/tmp mkdir -p /scratch/$USER/tmp && TMPDIR=/scratch/$USER/tmp qsub ...
```

### Container image location

The container image (`nerfstudio.sif`, ~8.4 GB) lives on scratch to avoid consuming home directory quota on the root partition. A symlink at `$HOME/workspace/fyp/containers/nerfstudio.sif` points to `/scratch/$USER/fyp-playground/containers/nerfstudio.sif`. PBS scripts resolve the symlink transparently.

## Dependencies

- **Builds on**: `environments/nerfstudio/setup_env.sh` (exact same install logic)
- **Uses**: `scripts/experiments/run_experiments.py`, `scripts/eval_experiments.py`, `scripts/render_experiments.py`, `config/experiment_config.py`
- **Sync uses**: `scripts/change_config_path.py` for path rewriting

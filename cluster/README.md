# cluster/

HPC deployment files for running nerfstudio training on the Vanda PBS cluster (NUS, `vanda.nus.edu.sg`).

## Directory Structure

| File | Purpose |
|------|---------|
| `nerfstudio.def` | Apptainer container definition (CUDA 11.8 + nerfstudio + sea-splatfacto) |
| `local_config.example.py` | Template for cluster's `local_config.py` (gitignored) |
| `jobs/train.pbs` | PBS job script for training (`run_experiments.py`) |
| `jobs/train_array.pbs` | PBS array job script — one experiment per sub-job |
| `jobs/eval.pbs` | PBS job script for evaluation + checkpoint cleanup |
| `scripts/submit.sh` | Convenience wrapper to submit train→eval dependency chain |
| `scripts/sync_results.sh` | **Runs locally** — rsync results + logs from cluster, rewrite config paths |

## Workspace Layout

The PBS job scripts hardcode paths that assume the directory structure below.

### Cluster layout (assumed by PBS jobs)

```
$HOME/workspace/fyp/
├── fyp-utils/                     # This repo
├── sea-splatfacto/                # Bind-mounted into container
├── nerfstudio/                    # Bind-mounted into container
└── containers/
    └── nerfstudio.sif             # Built container image

/scratch/$USER/fyp-playground/
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
cp cluster/local_config.example.py cluster/local_config.py
# Edit if needed (datasets, templates, etc.)
```

PBS scripts automatically copy `cluster/local_config.py` into `scripts/experiments/` at job start.

## Usage

### Submit training + eval (sequential)

```bash
cd ~/workspace/fyp/fyp-utils

# Submit both jobs (eval depends on training success)
./cluster/scripts/submit.sh

# Training only
./cluster/scripts/submit.sh --train-only

# Filter specific datasets
./cluster/scripts/submit.sh --filter torpedo

# Check job status
qstat -u $USER
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

### Sync results locally

Run on your **local machine**:

```bash
cd ~/workspace/fyp/fyp-utils

# Sync outputs + logs (excludes checkpoints + tensorboard events)
./cluster/scripts/sync_results.sh

# Include checkpoints (for runs you want to render)
./cluster/scripts/sync_results.sh --include-checkpoints

# Sync checkpoints + auto-render after sync
./cluster/scripts/sync_results.sh --include-checkpoints --render
```

## How Editability Works

The container bakes compiled dependencies (PyTorch, CUDA, tiny-cuda-nn) and editable-install metadata (entry points, `.dist-info`). At runtime, PBS scripts bind-mount the live source trees:

```
--bind ~/workspace/fyp/sea-splatfacto:/opt/sea-splatfacto
--bind ~/workspace/fyp/nerfstudio:/opt/nerfstudio
```

Python resolves source through the editable install pointers to `/opt/sea-splatfacto` and `/opt/nerfstudio`, which now point to the host's live code. Code changes take effect immediately — only structural changes (new entry points, packages, or dependencies) require a container rebuild.

`local_config.py` is in `$HOME`, which Apptainer auto-binds.

## Resource Requests

| Job | GPUs | CPUs | Memory | Walltime |
|-----|------|------|--------|----------|
| `train.pbs` / `train_array.pbs` | 1x A40 | 36 | 128GB | 12h |
| `eval.pbs` | 1x A40 | 2 | 16GB | 4h |

Queue limits: max walltime 12h, max 2x A40 GPUs per job, max 72 CPUs per node, max 4 concurrent jobs.

## Dependencies

- **Builds on**: `environments/nerfstudio/setup_env.sh` (exact same install logic)
- **Uses**: `scripts/experiments/run_experiments.py`, `scripts/eval_experiments.py`
- **Sync uses**: `scripts/change_config_path.py` for path rewriting

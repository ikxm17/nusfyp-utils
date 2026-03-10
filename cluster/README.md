# cluster/

HPC deployment files for running nerfstudio training on the Vanda PBS cluster (NUS, `vanda.nus.edu.sg`).

## Directory Structure

| File | Purpose |
|------|---------|
| `nerfstudio.def` | Apptainer container definition (CUDA 11.8 + nerfstudio + sea-splatfacto) |
| `train.pbs` | PBS job script for training (`run_experiments.py`) |
| `eval.pbs` | PBS job script for evaluation + checkpoint cleanup |
| `submit.sh` | Convenience wrapper to submit train→eval dependency chain |
| `sync_results.sh` | **Runs locally** — rsync results from cluster + rewrite config paths |
| `local_config.cluster.py` | Template for cluster's `local_config.py` |

## Cluster Layout

```
/home/svu/e0908336/workspace/fyp/
├── fyp-utils/                     # This repo
├── sea-splatfacto/                # Bind-mounted into container at runtime
├── nerfstudio/                    # Bind-mounted into container at runtime
└── containers/nerfstudio.sif      # Built container image (~8-10GB)

/scratch/e0908336/fyp-playground/
├── datasets/                      # Training data (rsync from local)
├── outputs/                       # Training outputs
└── logs/                          # PBS job logs
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
cp cluster/local_config.cluster.py scripts/experiments/local_config.py
# Edit if needed (datasets, templates, etc.)
```

## Usage

### Submit training + eval

```bash
cd ~/workspace/fyp/fyp-utils

# Submit both jobs (eval depends on training success)
./cluster/submit.sh

# Training only
./cluster/submit.sh --train-only

# Filter specific datasets
./cluster/submit.sh --filter torpedo

# Check job status
qstat -u $USER
```

### Sync results locally

Run on your **local machine**:

```bash
cd ~/workspace/fyp/fyp-utils

# Sync metrics and configs (excludes checkpoints + tensorboard events)
./cluster/sync_results.sh

# Include checkpoints (for runs you want to render)
./cluster/sync_results.sh --include-checkpoints
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
| `train.pbs` | 1x A40 | 4 | 32GB | 4h |
| `eval.pbs` | 1x A40 | 2 | 16GB | 1h |

## Dependencies

- **Builds on**: `environments/nerfstudio/setup_env.sh` (exact same install logic)
- **Uses**: `scripts/experiments/run_experiments.py`, `scripts/eval_experiments.py`
- **Sync uses**: `scripts/change_config_path.py` for path rewriting

# Nerfstudio Environment

Setup files for reproducible nerfstudio environments across machines, supporting both native conda and Docker workflows.

## Files

| File | Purpose |
|------|---------|
| `setup_env.sh` | Create/update the nerfstudio conda environment |
| `environment.yml` | Conda environment specification |
| `container.sh` | Docker container lifecycle management |
| `compose.yml` | Docker Compose configuration |
| `Dockerfile` | Docker image definition |
| `.env` | Docker environment variables |

---

## Conda Setup (setup_env.sh)

Creates a `nerfstudio` conda environment with PyTorch and installs nerfstudio + sea-splatfacto from source. Supports CPU-only and CUDA platforms for use across local (no GPU) and remote (GPU) machines.

### Usage

```bash
# CPU-only, default paths
./setup_env.sh

# CUDA 11.8
./setup_env.sh --platform cu118

# CUDA 12.1 with custom nerfstudio path
./setup_env.sh --platform cu121 --nerfstudio /opt/nerfstudio
```

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--platform {cpu,cu118,cu121}` | Compute platform | `cpu` |
| `--nerfstudio <path>` | Path to nerfstudio source | `~/opt/nerfstudio` |
| `--sea-splatfacto <path>` | Path to sea-splatfacto source | auto-detected from project root |

### What it installs

1. Conda environment from `environment.yml`
2. PyTorch 2.1.2 + torchvision (CPU or CUDA variant)
3. CUDA toolkit + tiny-cuda-nn (GPU platforms only)
4. Nerfstudio from source (editable mode)
5. Sea-splatfacto from source (editable mode)
6. CLI tab completions (`ns-install-cli`)

---

## Docker (container.sh)

Simple wrapper for Docker Compose container lifecycle.

### Usage

```bash
# Start services and enter container shell
./container.sh start

# Rebuild images and enter container shell
./container.sh rebuild

# Stop the container
./container.sh stop
```

# Nerfstudio Environment

Setup files for a reproducible nerfstudio conda environment across machines. COLMAP is included in the conda environment, so both `ns-process-data` and `colmap gui` are available after setup.

## Files

| File | Purpose |
|------|---------|
| `setup_env.sh` | Create/update the nerfstudio conda environment |
| `environment.yml` | Conda environment specification (includes COLMAP) |

---

## Conda Setup (setup_env.sh)

Creates a `nerfstudio` conda environment with PyTorch, COLMAP, and installs nerfstudio + sea-splatfacto from source. Supports CPU-only and CUDA platforms for use across local (no GPU) and remote (GPU) machines.

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
| `--platform {cpu,cu118,cu121}` | Compute platform | `cu118` |
| `--nerfstudio <path>` | Path to nerfstudio source | `~/opt/nerfstudio` |
| `--sea-splatfacto <path>` | Path to sea-splatfacto source | auto-detected from project root |

### What it installs

1. Conda environment from `environment.yml`
2. PyTorch 2.1.2 + torchvision (CPU or CUDA variant)
3. CUDA toolkit + tiny-cuda-nn (GPU platforms only)
4. Nerfstudio from source (editable mode)
5. Sea-splatfacto from source (editable mode)
6. CLI tab completions (`ns-install-cli`)
7. COLMAP (via conda-forge, included in `environment.yml`)

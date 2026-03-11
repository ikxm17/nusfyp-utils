#!/usr/bin/env bash
#
# Setup nerfstudio conda environment with flexible compute platform support.
#
# Usage:
#   ./setup_env.sh [OPTIONS]
#
# Options:
#   --platform cpu|cu118|cu121   Compute platform (default: cpu)
#   --nerfstudio PATH            Path to nerfstudio source (default: ~/opt/nerfstudio)
#   --sea-splatfacto PATH        Path to sea-splatfacto source (default: auto-detected from project root)
#   --help                       Show this help message
#
# Examples:
#   ./setup_env.sh                                    # CPU-only, default paths
#   ./setup_env.sh --platform cu118                   # CUDA 11.8
#   ./setup_env.sh --platform cu121 --nerfstudio /opt/nerfstudio
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

COMPUTE_PLATFORM="cpu"
NERFSTUDIO_PATH="$HOME/opt/nerfstudio"
SEA_SPLATFACTO_PATH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform)   COMPUTE_PLATFORM="$2"; shift 2 ;;
    --nerfstudio) NERFSTUDIO_PATH="$2"; shift 2 ;;
    --sea-splatfacto) SEA_SPLATFACTO_PATH="$2"; shift 2 ;;
    --help)
      sed -n '2,/^$/s/^# \?//p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1 (see --help)"
      exit 1
      ;;
  esac
done

# Auto-detect sea-splatfacto relative to project root (fyp-utils/../sea-splatfacto)
if [ -z "$SEA_SPLATFACTO_PATH" ]; then
  SEA_SPLATFACTO_PATH="$(cd "$PROJECT_ROOT/.." && pwd)/sea-splatfacto"
fi

TORCH_VERSION="2.1.2"
TORCHVISION_VERSION="0.16.2"

case "$COMPUTE_PLATFORM" in
  cpu|cu118|cu121) ;;
  *)
    echo "Invalid platform: $COMPUTE_PLATFORM (choose: cpu, cu118, cu121)"
    exit 1
    ;;
esac

echo "==> Compute platform:  $COMPUTE_PLATFORM"
echo "==> Nerfstudio path:   $NERFSTUDIO_PATH"
echo "==> Sea-splatfacto:    $SEA_SPLATFACTO_PATH"

# Create or update conda env
if conda env list | grep -q "^nerfstudio "; then
  echo "==> Updating existing nerfstudio environment"
  conda env update -f "$SCRIPT_DIR/environment.yml" --prune
else
  echo "==> Creating nerfstudio environment"
  conda env create -f "$SCRIPT_DIR/environment.yml"
fi

# Activate (disable nounset — conda/nerfstudio activate hooks reference unset vars)
eval "$(conda shell.bash hook)"
set +u
conda activate nerfstudio
set -u

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch
echo "==> Installing PyTorch ${TORCH_VERSION} (${COMPUTE_PLATFORM})"
if [ "$COMPUTE_PLATFORM" = "cpu" ]; then
  pip install "torch==${TORCH_VERSION}+cpu" "torchvision==${TORCHVISION_VERSION}+cpu" \
    --extra-index-url https://download.pytorch.org/whl/cpu
else
  pip install "torch==${TORCH_VERSION}+${COMPUTE_PLATFORM}" "torchvision==${TORCHVISION_VERSION}+${COMPUTE_PLATFORM}" \
    --extra-index-url "https://download.pytorch.org/whl/${COMPUTE_PLATFORM}"
fi

# CUDA-specific packages
if [ "$COMPUTE_PLATFORM" != "cpu" ]; then
  case "$COMPUTE_PLATFORM" in
    cu118) CUDA_LABEL="cuda-11.8.0" ;;
    cu121) CUDA_LABEL="cuda-12.1.0" ;;
  esac

  echo "==> Installing CUDA toolkit (${CUDA_LABEL})"
  conda install -c "nvidia/label/${CUDA_LABEL}" cuda-toolkit -y

  echo "==> Installing tiny-cuda-nn"
  # If TORCH_CUDA_ARCH_LIST is set but TCNN_CUDA_ARCHITECTURES is not (e.g. container
  # builds without a GPU), derive it so tiny-cuda-nn doesn't try to auto-detect the GPU.
  # TORCH_CUDA_ARCH_LIST uses "8.6" format; TCNN_CUDA_ARCHITECTURES uses "86" format.
  if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ] && [ -z "${TCNN_CUDA_ARCHITECTURES:-}" ]; then
    export TCNN_CUDA_ARCHITECTURES
    TCNN_CUDA_ARCHITECTURES=$(echo "$TORCH_CUDA_ARCH_LIST" | tr -d '.' | tr ' ' ';')
  fi
  pip install setuptools wheel ninja
  pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
fi

# Install nerfstudio (editable + compat mode for IDE/Pylance support)
echo "==> Installing nerfstudio from $NERFSTUDIO_PATH"
pip install -e "$NERFSTUDIO_PATH" --config-settings editable_mode=compat

# Install sea-splatfacto (editable, depends on nerfstudio)
echo "==> Installing sea-splatfacto from $SEA_SPLATFACTO_PATH"
pip install -e "$SEA_SPLATFACTO_PATH" --config-settings editable_mode=compat

# CLI tab completions (non-fatal — some commands may fail completion generation)
echo "==> Installing CLI completions"
ns-install-cli || echo "Warning: ns-install-cli had errors (completions may be incomplete)"

echo ""
echo "Done! Activate with: conda activate nerfstudio"

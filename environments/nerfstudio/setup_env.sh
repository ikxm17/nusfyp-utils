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
NERFSTUDIO_REPO="https://github.com/ikxm17/nerfstudio.git"
NERFSTUDIO_BRANCH="fix/lazy-exporter-imports"
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

  # Expose conda-installed CUDA to the linker and build tools — conda's
  # compiler_compat/ld doesn't search $CONDA_PREFIX/lib by default.
  export CUDA_HOME="$CONDA_PREFIX"

  # Build LIBRARY_PATH from known conda CUDA lib locations + system fallback.
  # Conda packages may place libraries in different subdirectories depending
  # on the package version and channel.
  _cuda_lib_path="$CONDA_PREFIX/lib"
  [ -d "$CONDA_PREFIX/lib/stubs" ] && _cuda_lib_path="$_cuda_lib_path:$CONDA_PREFIX/lib/stubs"
  [ -d "$CONDA_PREFIX/targets/x86_64-linux/lib" ] && _cuda_lib_path="$_cuda_lib_path:$CONDA_PREFIX/targets/x86_64-linux/lib"
  # System CUDA fallback — if the machine has /usr/local/cuda, include its
  # lib64 so the linker can find runtime libs even if conda's copies are
  # missing or lack development symlinks.
  [ -d "/usr/local/cuda/lib64" ] && _cuda_lib_path="$_cuda_lib_path:/usr/local/cuda/lib64"
  export LIBRARY_PATH="$_cuda_lib_path:${LIBRARY_PATH:-}"
  echo "==> CUDA_HOME=$CUDA_HOME"
  echo "==> LIBRARY_PATH=$LIBRARY_PATH"

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

# Clone or update nerfstudio from fork
if [ ! -d "$NERFSTUDIO_PATH/.git" ]; then
  echo "==> Cloning nerfstudio from $NERFSTUDIO_REPO (branch: $NERFSTUDIO_BRANCH)"
  git clone --branch "$NERFSTUDIO_BRANCH" "$NERFSTUDIO_REPO" "$NERFSTUDIO_PATH"
else
  echo "==> Updating nerfstudio checkout"
  # Ensure fork remote exists and checkout the right branch
  (cd "$NERFSTUDIO_PATH" && {
    if ! git remote get-url origin 2>/dev/null | grep -q "ikxm17"; then
      git remote rename origin upstream 2>/dev/null || true
      git remote add origin "$NERFSTUDIO_REPO"
    fi
    git fetch origin
    git checkout "$NERFSTUDIO_BRANCH" 2>/dev/null || git checkout -b "$NERFSTUDIO_BRANCH" "origin/$NERFSTUDIO_BRANCH"
    git pull --ff-only origin "$NERFSTUDIO_BRANCH" || true
  })
fi

# Install nerfstudio (editable + compat mode for IDE/Pylance support)
# Pin numpy<2 inline — PyTorch 2.1.2 has ABI incompatibility with numpy 2.x.
# Pinning inline ensures pip resolves the constraint in the same invocation.
echo "==> Installing nerfstudio from $NERFSTUDIO_PATH"
pip install "numpy<2" -e "$NERFSTUDIO_PATH" --config-settings editable_mode=compat

# Install sea-splatfacto (editable, depends on nerfstudio)
echo "==> Installing sea-splatfacto from $SEA_SPLATFACTO_PATH"
pip install "numpy<2" -e "$SEA_SPLATFACTO_PATH" --config-settings editable_mode=compat

# CLI tab completions (non-fatal — some commands may fail completion generation)
echo "==> Installing CLI completions"
ns-install-cli || echo "Warning: ns-install-cli had errors (completions may be incomplete)"

echo ""
echo "Done! Activate with: conda activate nerfstudio"

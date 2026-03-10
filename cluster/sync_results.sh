#!/bin/bash
# Sync experiment results from Vanda HPC cluster to local machine.
#
# Runs LOCALLY — downloads outputs (excluding large checkpoints and
# tensorboard event files), then rewrites config.yml paths for local use.
#
# Usage:
#   ./cluster/sync_results.sh                        # Sync metrics + configs only
#   ./cluster/sync_results.sh --include-checkpoints  # Also sync checkpoint files
#
# Prerequisites:
#   - SSH key or password access to vanda.nus.edu.sg as e0908336
#   - fyp-playground directory exists locally
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Cluster connection
CLUSTER_USER="e0908336"
CLUSTER_HOST="vanda.nus.edu.sg"
CLUSTER_REMOTE="${CLUSTER_USER}@${CLUSTER_HOST}"

# Paths
REMOTE_OUTPUTS="/scratch/${CLUSTER_USER}/fyp-playground/outputs/"
LOCAL_PLAYGROUND="$(cd "$PROJECT_ROOT/../fyp-playground" && pwd)"
LOCAL_OUTPUTS="${LOCAL_PLAYGROUND}/outputs/"

# Path rewrite settings (cluster → local)
OLD_BASE="/home/svu/${CLUSTER_USER}"
NEW_BASE="$HOME"
OLD_DATA="/scratch/${CLUSTER_USER}/fyp-playground/datasets"
NEW_DATA="${LOCAL_PLAYGROUND}/datasets"

INCLUDE_CHECKPOINTS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --include-checkpoints) INCLUDE_CHECKPOINTS=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Build rsync exclude list
EXCLUDES=(
    --exclude "events.out.tfevents.*"
)
if [ "$INCLUDE_CHECKPOINTS" = false ]; then
    EXCLUDES+=(--exclude "*.ckpt")
fi

echo "==> Syncing outputs from ${CLUSTER_REMOTE}:${REMOTE_OUTPUTS}"
echo "    to ${LOCAL_OUTPUTS}"
echo "    Checkpoints: $([ "$INCLUDE_CHECKPOINTS" = true ] && echo "included" || echo "excluded")"

mkdir -p "$LOCAL_OUTPUTS"

rsync -avz --progress \
    "${EXCLUDES[@]}" \
    "${CLUSTER_REMOTE}:${REMOTE_OUTPUTS}" \
    "$LOCAL_OUTPUTS"

echo ""
echo "==> Rewriting paths in config.yml files..."

find "$LOCAL_OUTPUTS" -name "config.yml" | while read config; do
    python "$PROJECT_ROOT/scripts/change_config_path.py" "$config" \
        --old-base "$OLD_BASE" \
        --new-base "$NEW_BASE" \
        --old-data "$OLD_DATA" \
        --new-data "$NEW_DATA"
done

echo ""
echo "==> Sync complete."

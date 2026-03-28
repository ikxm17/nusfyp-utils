#!/bin/bash
# Sync experiment results from Vanda HPC cluster to local machine.
#
# Runs LOCALLY — downloads outputs (excluding large checkpoints and
# tensorboard event files), then rewrites config.yml paths for local use.
#
# Usage:
#   ./cluster/scripts/sync_results.sh                        # Sync metrics + configs only
#   ./cluster/scripts/sync_results.sh --include-renders       # Also sync per-frame render PNGs
#   ./cluster/scripts/sync_results.sh --include-checkpoints  # Also sync checkpoint files
#   ./cluster/scripts/sync_results.sh --include-tb --tb-filter "tune18_*"  # Sync TB for matching experiments only
#   ./cluster/scripts/sync_results.sh --dataset redsea_unprocessed        # Sync only this dataset's outputs
#   ./cluster/scripts/sync_results.sh --cleanup              # Sync, then delete outputs/logs from scratch
#
# Prerequisites:
#   - SSH key or password access to vanda.nus.edu.sg as e0908336
#   - fyp-playground directory exists locally
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

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
INCLUDE_TB=false
INCLUDE_RENDERS=false
TB_FILTER=""
DATASET=""
CLEANUP=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --include-checkpoints) INCLUDE_CHECKPOINTS=true; shift ;;
        --include-tb) INCLUDE_TB=true; shift ;;
        --include-renders) INCLUDE_RENDERS=true; shift ;;
        --tb-filter) TB_FILTER="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --cleanup) CLEANUP=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate: --tb-filter requires --include-tb
if [ -n "$TB_FILTER" ] && [ "$INCLUDE_TB" = false ]; then
    echo "Error: --tb-filter requires --include-tb"
    exit 1
fi

# Build rsync exclude list
EXCLUDES=()
if [ "$INCLUDE_TB" = false ]; then
    # Exclude all tfevents
    EXCLUDES+=(--exclude "events.out.tfevents.*")
elif [ -n "$TB_FILTER" ]; then
    # Include tfevents only for matching experiments, exclude the rest
    EXCLUDES+=(--include "**/*-${TB_FILTER}*/**/events.out.tfevents.*")
    EXCLUDES+=(--exclude "events.out.tfevents.*")
fi
# When --include-tb without --tb-filter: no exclude rule → all tfevents synced (backward compat)
if [ "$INCLUDE_CHECKPOINTS" = false ]; then
    EXCLUDES+=(--exclude "*.ckpt")
fi
# Exclude render frame directories unless --include-renders is set.
# Without the flag, only MP4 videos sync locally (frames stay on Vanda).
if [ "$INCLUDE_RENDERS" = false ]; then
    EXCLUDES+=(--exclude "**/renders/dataset/*/*/")
    EXCLUDES+=(--exclude "**/renders/camera-path/*/*/*/")
fi
EXCLUDES+=(--exclude "_*_filelist.txt")

# Scope source paths when --dataset is specified
RSYNC_REMOTE_OUTPUTS="${REMOTE_OUTPUTS}"
RSYNC_LOCAL_OUTPUTS="${LOCAL_OUTPUTS}"
if [ -n "$DATASET" ]; then
    RSYNC_REMOTE_OUTPUTS="${REMOTE_OUTPUTS}${DATASET}/"
    RSYNC_LOCAL_OUTPUTS="${LOCAL_OUTPUTS}${DATASET}/"
fi

echo "==> Syncing outputs from ${CLUSTER_REMOTE}:${RSYNC_REMOTE_OUTPUTS}"
echo "    to ${RSYNC_LOCAL_OUTPUTS}"
[ -n "$DATASET" ] && echo "    Dataset: ${DATASET}"
echo "    Renders: $([ "$INCLUDE_RENDERS" = true ] && echo "included" || echo "excluded")"
echo "    Checkpoints: $([ "$INCLUDE_CHECKPOINTS" = true ] && echo "included" || echo "excluded")"
if [ "$INCLUDE_TB" = true ] && [ -n "$TB_FILTER" ]; then
    echo "    TensorBoard: included (filter: *-${TB_FILTER}*)"
else
    echo "    TensorBoard: $([ "$INCLUDE_TB" = true ] && echo "included" || echo "excluded")"
fi

mkdir -p "$RSYNC_LOCAL_OUTPUTS"

rsync -avz --progress \
    "${EXCLUDES[@]}" \
    "${CLUSTER_REMOTE}:${RSYNC_REMOTE_OUTPUTS}" \
    "$RSYNC_LOCAL_OUTPUTS"

# Sync training logs (SUCCESS_*.log, FAILED_*.log)
REMOTE_LOGS="/scratch/${CLUSTER_USER}/fyp-playground/logs/"
LOCAL_LOGS="${LOCAL_PLAYGROUND}/logs/"

echo ""
echo "==> Syncing logs from ${CLUSTER_REMOTE}:${REMOTE_LOGS}"
echo "    to ${LOCAL_LOGS}"

mkdir -p "$LOCAL_LOGS"

rsync -avz --progress "${CLUSTER_REMOTE}:${REMOTE_LOGS}" "$LOCAL_LOGS"

echo ""
echo "==> Rewriting paths in config.yml files..."

find "$RSYNC_LOCAL_OUTPUTS" -name "config.yml" | while read config; do
    python3 "$PROJECT_ROOT/scripts/change_config_path.py" "$config" \
        --old-base "$OLD_BASE" \
        --new-base "$NEW_BASE" \
        --old-data "$OLD_DATA" \
        --new-data "$NEW_DATA"
done

# Post-sync cleanup: remove synced outputs and logs from scratch
if [ "$CLEANUP" = true ]; then
    echo ""
    echo "==> Cleaning up remote scratch..."
    ssh "${CLUSTER_REMOTE}" "rm -rf ${REMOTE_OUTPUTS}* ${REMOTE_LOGS}*"
    echo "    Removed contents of ${REMOTE_OUTPUTS} and ${REMOTE_LOGS}"
fi

echo ""
echo "==> Sync complete."

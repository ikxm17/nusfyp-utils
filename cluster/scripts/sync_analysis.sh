#!/bin/bash
# Sync analysis artifacts from Vanda HPC to local machine.
#
# Pulls ONLY the lightweight analysis outputs (report.json, grids, extracted
# renders) — NOT the full experiment outputs, checkpoints, or TB files.
# Typically ~10-50 MB vs 1 GB+ for full sync_results.sh.
#
# Runs LOCALLY.
#
# Usage:
#   ./cluster/scripts/sync_analysis.sh <batch_prefix>
#   ./cluster/scripts/sync_analysis.sh <batch_prefix> --local-dir /tmp/my-analysis
#
# Prerequisites:
#   - SSH key or password access to vanda.nus.edu.sg as e0908336
set -euo pipefail

CLUSTER_USER="e0908336"
CLUSTER_HOST="vanda.nus.edu.sg"
CLUSTER_REMOTE="${CLUSTER_USER}@${CLUSTER_HOST}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <batch_prefix> [--local-dir <path>]"
    exit 1
fi

BATCH_PREFIX="$1"
shift

LOCAL_DIR="/tmp/batch-analysis"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local-dir) LOCAL_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

REMOTE_ANALYSIS="/scratch/${CLUSTER_USER}/fyp-playground/analysis/${BATCH_PREFIX}/"

# Verify remote directory exists before syncing
if ! ssh "${CLUSTER_REMOTE}" "test -d ${REMOTE_ANALYSIS}"; then
    echo "Error: Remote analysis directory not found: ${REMOTE_ANALYSIS}"
    echo "Has the analyze job completed?"
    exit 1
fi

echo "==> Syncing analysis artifacts from ${CLUSTER_REMOTE}:${REMOTE_ANALYSIS}"
echo "    to ${LOCAL_DIR}/"
echo "    Batch prefix: ${BATCH_PREFIX}"

mkdir -p "$LOCAL_DIR"

rsync -avz --progress \
    "${CLUSTER_REMOTE}:${REMOTE_ANALYSIS}" \
    "$LOCAL_DIR/"

echo ""
echo "==> Sync complete."
echo "    Report: ${LOCAL_DIR}/report.json"
echo "    Grids:  ${LOCAL_DIR}/grids/"

# Cluster connection and remote-path defaults shared across cluster/scripts/*.
#
# Source from another script with:
#   source "$(dirname "${BASH_SOURCE[0]}")/_cluster_env.sh"
#
# Each variable can be overridden by the caller's environment — e.g. set
# VANDA_USER=otheruser before invoking the sync scripts to use a different
# account without editing this file.

CLUSTER_USER="${VANDA_USER:-e0908336}"
CLUSTER_HOST="${VANDA_HOST:-vanda.nus.edu.sg}"
CLUSTER_REMOTE="${CLUSTER_USER}@${CLUSTER_HOST}"

# Remote scratch layout — these mirror the conventions in
# fyp-utils/cluster/README.md and the PBS jobs.
REMOTE_BASE="/scratch/${CLUSTER_USER}/fyp-playground"
REMOTE_OUTPUTS="${REMOTE_BASE}/outputs/"
REMOTE_LOGS="${REMOTE_BASE}/logs/"
REMOTE_ANALYSIS_BASE="${REMOTE_BASE}/analysis"

# Local default for batch-analysis artifacts (must match analyze_batch.py
# default to keep the sync target aligned with the analyser's output).
LOCAL_BATCH_ANALYSIS_DIR="${LOCAL_BATCH_ANALYSIS_DIR:-/tmp/batch-analysis}"

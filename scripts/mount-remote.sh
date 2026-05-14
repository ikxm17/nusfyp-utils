#!/bin/bash
# Mount Vanda's /scratch outputs tree (or a subpath) locally via SSHFS.
#
# Usage:
#   ./scripts/mount-remote.sh                       # whole outputs/ tree
#   ./scripts/mount-remote.sh curasao               # just outputs/curasao/
#   ./scripts/mount-remote.sh curasao/tune17_xxx    # one specific run
#
# Environment:
#   VANDA_HOST  (default vanda)         SSH host alias
#   VANDA_USER  (default e0908336)      remote username on /scratch
#
# The mountpoint sits next to the local synced outputs:
#   ~/workspace/fyp/fyp-playground/outputs-remote/vanda/
# so any tool that can read local outputs (TensorBoard, mpv, read_tb.py)
# can read live remote tfevents/renders without copying bytes locally.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REMOTE_HOST="${VANDA_HOST:-vanda}"
REMOTE_USER="${VANDA_USER:-e0908336}"
REMOTE_BASE="/scratch/${REMOTE_USER}/fyp-playground/outputs"
LOCAL_BASE="${PROJECT_ROOT}/fyp-playground/outputs-remote/vanda"

SUBPATH="${1:-}"
# Strip leading/trailing slashes so concatenation stays clean
SUBPATH="${SUBPATH#/}"
SUBPATH="${SUBPATH%/}"

if [[ "$SUBPATH" == "-h" || "$SUBPATH" == "--help" ]]; then
    awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
    exit 0
fi

REMOTE_PATH="${REMOTE_BASE}${SUBPATH:+/$SUBPATH}"
LOCAL_PATH="${LOCAL_BASE}${SUBPATH:+/$SUBPATH}"

mkdir -p "$LOCAL_PATH"

if mountpoint -q "$LOCAL_PATH"; then
    echo "Already mounted: $LOCAL_PATH"
    exit 0
fi

echo "==> sshfs ${REMOTE_HOST}:${REMOTE_PATH} -> ${LOCAL_PATH}"
sshfs "${REMOTE_HOST}:${REMOTE_PATH}" "$LOCAL_PATH" \
    -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,follow_symlinks,cache=yes

echo "Mounted. Browse with: ls \"$LOCAL_PATH\""
echo "Unmount with:        ./scripts/unmount-remote.sh${SUBPATH:+ $SUBPATH}"

#!/bin/bash
# Unmount the Vanda SSHFS mount (or a specific subpath).
#
# Usage:
#   ./scripts/unmount-remote.sh                  # unmount everything under outputs-remote/vanda/
#   ./scripts/unmount-remote.sh curasao          # unmount just that subpath
#
# Tries a clean unmount first (fusermount -u); falls back to lazy unmount
# (fusermount -uz) for hung mounts (e.g. after VPN drops).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

LOCAL_BASE="${PROJECT_ROOT}/fyp-playground/outputs-remote/vanda"

SUBPATH="${1:-}"
SUBPATH="${SUBPATH#/}"
SUBPATH="${SUBPATH%/}"

if [[ "$SUBPATH" == "-h" || "$SUBPATH" == "--help" ]]; then
    awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
    exit 0
fi

unmount_one() {
    local target="$1"
    # Clean unmount can fail with EBUSY if the kernel still holds cached inodes
    # from a recent ls/stat. Retry once after a brief pause, then fall back to
    # lazy unmount (functionally identical for read-only viewing).
    if fusermount -u "$target" 2>/dev/null; then
        echo "Unmounted $target"
    else
        sleep 0.3
        if fusermount -u "$target" 2>/dev/null; then
            echo "Unmounted $target"
        elif fusermount -uz "$target" 2>/dev/null; then
            echo "Unmounted $target (lazy)"
        else
            echo "ERROR: failed to unmount $target" >&2
            return 1
        fi
    fi
}

if [ -z "$SUBPATH" ]; then
    # Unmount any SSHFS/FUSE mount under the base, deepest-first so nested mounts release cleanly.
    mapfile -t MOUNTS < <(mount | awk -v base="$LOCAL_BASE" '$3 ~ base && ($1 ~ /sshfs/ || $5 ~ /fuse/) {print $3}' | awk '{ print length, $0 }' | sort -rn | cut -d' ' -f2-)
    if [ "${#MOUNTS[@]}" -eq 0 ]; then
        echo "No SSHFS mounts under $LOCAL_BASE"
        exit 0
    fi
    for m in "${MOUNTS[@]}"; do
        unmount_one "$m"
    done
else
    LOCAL_PATH="${LOCAL_BASE}/${SUBPATH}"
    if ! mountpoint -q "$LOCAL_PATH"; then
        echo "Not mounted: $LOCAL_PATH"
        exit 0
    fi
    unmount_one "$LOCAL_PATH"
fi

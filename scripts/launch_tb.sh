#!/bin/bash
# Launch TensorBoard against an FYP experiment (or the whole outputs tree).
#
# Usage:
#   ./scripts/launch_tb.sh                         # logdir = full outputs/ tree
#   ./scripts/launch_tb.sh 008v00_control          # single experiment by substring
#   ./scripts/launch_tb.sh saltpond_unprocessed    # whole dataset (multi-run overlay)
#   ./scripts/launch_tb.sh /full/path/to/run       # explicit path (skips substring search)
#
# Environment:
#   PORT  (default 6006)  TensorBoard port
#
# Substring search scans two roots:
#   - fyp-playground/outputs/                (locally synced)
#   - fyp-playground/outputs-remote/vanda/   (live SSHFS mount, if mounted)
# If a substring matches the same run name in both roots, the mount wins (live wins
# over stale). The path is the disambiguator — pass an explicit path to override.
#
# TB event files are NOT synced from Vanda by default. If the resolved logdir
# has no events.out.tfevents.* files, the script refuses to launch and prints
# either the sync command (local root) or the mount command (remote root).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUTPUTS_LOCAL="${PROJECT_ROOT}/fyp-playground/outputs"
OUTPUTS_REMOTE="${PROJECT_ROOT}/fyp-playground/outputs-remote/vanda"

# Build the list of roots that actually exist on disk. The remote root only
# exists once mount-remote.sh has been run, and the mountpoint dir persists
# even when unmounted (just shows as empty), so existence + a populated check
# is enough — we don't need to probe `mountpoint -q` here.
OUTPUTS_ROOTS=()
[ -d "$OUTPUTS_LOCAL" ] && OUTPUTS_ROOTS+=("$OUTPUTS_LOCAL")
[ -d "$OUTPUTS_REMOTE" ] && OUTPUTS_ROOTS+=("$OUTPUTS_REMOTE")

if [ "${#OUTPUTS_ROOTS[@]}" -eq 0 ]; then
    echo "Error: neither $OUTPUTS_LOCAL nor $OUTPUTS_REMOTE exists" >&2
    exit 1
fi

PORT="${PORT:-6006}"
FILTER="${1:-}"

if [[ "$FILTER" == "-h" || "$FILTER" == "--help" ]]; then
    awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
    exit 0
fi

# Explicit-path mode: if the argument is an existing directory (absolute or
# relative to CWD), use it directly and skip substring resolution.
LOGDIR=""
if [ -n "$FILTER" ]; then
    # Expand leading ~ to $HOME so users can pass shell-style paths
    CANDIDATE="${FILTER/#\~/$HOME}"
    if [ -d "$CANDIDATE" ]; then
        LOGDIR="$(cd "$CANDIDATE" && pwd)"
        echo "==> Logdir: $LOGDIR (explicit path)"
    fi
fi

# Substring resolution across all available roots
if [ -z "$LOGDIR" ]; then
    if [ -z "$FILTER" ]; then
        LOGDIR="${OUTPUTS_ROOTS[0]}"
        echo "==> Logdir: $LOGDIR (full outputs tree)"
        if [ "${#OUTPUTS_ROOTS[@]}" -gt 1 ]; then
            echo "    (other root available: ${OUTPUTS_ROOTS[1]} — pass a substring or path to switch)"
        fi
    else
        # Dataset dirs sit at depth 1, experiment dirs at depth 2 under each root.
        MATCHES=()
        for root in "${OUTPUTS_ROOTS[@]}"; do
            while IFS= read -r line; do
                [ -n "$line" ] && MATCHES+=("$line")
            done < <(find "$root" -mindepth 1 -maxdepth 2 -type d -name "*${FILTER}*" 2>/dev/null | sort)
        done

        if [ "${#MATCHES[@]}" -eq 0 ]; then
            echo "Error: no dataset or experiment directory matched '*${FILTER}*' under any root:" >&2
            for r in "${OUTPUTS_ROOTS[@]}"; do
                echo "    $r" >&2
            done
            exit 1
        elif [ "${#MATCHES[@]}" -eq 1 ]; then
            LOGDIR="${MATCHES[0]}"
            echo "==> Logdir: $LOGDIR"
        else
            # Mirror case: same run name in both local + remote. Prefer remote (live).
            if [ "${#MATCHES[@]}" -eq 2 ] && \
               [ "$(basename "${MATCHES[0]}")" = "$(basename "${MATCHES[1]}")" ] && \
               [[ "${MATCHES[0]}" == "$OUTPUTS_LOCAL"* ]] && \
               [[ "${MATCHES[1]}" == "$OUTPUTS_REMOTE"* ]]; then
                LOGDIR="${MATCHES[1]}"
                echo "==> Logdir: $LOGDIR"
                echo "    (preferring live mount over local copy at ${MATCHES[0]})"
            else
                echo "Error: '${FILTER}' matched ${#MATCHES[@]} directories. Use a more specific filter:" >&2
                printf '    %s\n' "${MATCHES[@]:0:20}" >&2
                if [ "${#MATCHES[@]}" -gt 20 ]; then
                    echo "    ... and $(( ${#MATCHES[@]} - 20 )) more" >&2
                fi
                exit 1
            fi
        fi
    fi
fi

# Refuse to launch if no TB event files exist under the resolved logdir.
if ! find "$LOGDIR" -name "events.out.tfevents.*" -type f -print -quit | grep -q .; then
    echo "Error: no events.out.tfevents.* files found under $LOGDIR" >&2
    echo >&2
    if [[ "$LOGDIR" == "$OUTPUTS_REMOTE"* ]]; then
        echo "Logdir is under the Vanda SSHFS mount. Possible causes:" >&2
        echo "  - mount is not active:  ./fyp-utils/scripts/mount-remote.sh" >&2
        echo "  - NUS VPN disconnected" >&2
        echo "  - the run hasn't produced tfevents yet" >&2
    else
        echo "TB event files are not synced from Vanda by default. Pull them with:" >&2
        if [ -n "$FILTER" ]; then
            echo "  ./fyp-utils/cluster/scripts/sync_results.sh --include-tb --tb-filter \"${FILTER}\"" >&2
        else
            echo "  ./fyp-utils/cluster/scripts/sync_results.sh --include-tb --tb-filter \"<glob>\"" >&2
        fi
        echo "Or mount Vanda live (no sync needed):" >&2
        echo "  ./fyp-utils/scripts/mount-remote.sh" >&2
    fi
    exit 1
fi

echo "==> tensorboard --logdir <above> --port $PORT --bind_all"
exec conda run -n nerfstudio --no-capture-output \
    tensorboard --logdir "$LOGDIR" --port "$PORT" --bind_all

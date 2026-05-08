#!/bin/bash
# Submit train→eval→render dependency chain to PBS.
#
# Usage:
#   ./cluster/scripts/submit.sh                        # Submit train + eval
#   ./cluster/scripts/submit.sh --render               # Submit train + eval + render
#   ./cluster/scripts/submit.sh --train-only           # Submit training only
#   ./cluster/scripts/submit.sh --parallel             # Submit as PBS array job (1 experiment per sub-job)
#   ./cluster/scripts/submit.sh --filter torpedo       # Only experiments whose name contains "torpedo"
#   ./cluster/scripts/submit.sh --dataset redsea_unprocessed  # Only experiments for this dataset
#   ./cluster/scripts/submit.sh --paid                 # Use paid queue (consumes GPU-hour allocation)
#   ./cluster/scripts/submit.sh --paid --walltime 2:00:00  # Paid queue + custom walltime
#   ./cluster/scripts/submit.sh --render --analyze --batch-prefix tune20 --dataset torpedo  # Full chain with analysis
#
# Run from the fyp-utils/ directory.
set -euo pipefail

# Verify TMPDIR is writable; fall back to scratch otherwise.
# PBS qsub needs temp space to copy the job script before submission.
_probe=""
if ! _probe=$(mktemp "${TMPDIR:-/tmp}/.submit_probe.XXXXXX" 2>/dev/null); then
    export TMPDIR="/scratch/${USER}/tmp"
    mkdir -p "$TMPDIR"
fi
[ -n "$_probe" ] && rm -f "$_probe"
unset _probe

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS_DIR="$SCRIPT_DIR/../jobs"

TRAIN_ONLY=false
PARALLEL=false
RENDER=false
ANALYZE=false
BATCH_PREFIX=""
PAID=false
EXTRA_ARGS=()
QSUB_OPTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-only)
            TRAIN_ONLY=true
            shift
            ;;
        --render)
            RENDER=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --analyze)
            ANALYZE=true
            shift
            ;;
        --batch-prefix)
            BATCH_PREFIX="$2"
            shift 2
            ;;
        --paid)
            PAID=true
            shift
            ;;
        --walltime)
            QSUB_OPTS+=(-l "walltime=$2")
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Default to free tier queue unless --paid is specified
if [ "$PAID" = true ]; then
    echo "==> Using PAID queue. Jobs will consume GPU-hour allocation."
else
    QSUB_OPTS+=(-q auto_free)
    echo "==> Using free tier queue (auto_free). Use --paid to consume GPU-hour allocation."
fi

# Build the -v EXTRA_ARGS flag for qsub (joined as a single env var, preserving
# current round-trip semantics: arrays for local quoting, string for PBS).
# Args containing spaces are not supported across the qsub boundary.
QSUB_V=()
if (( ${#EXTRA_ARGS[@]} > 0 )); then
    QSUB_V=(-v "EXTRA_ARGS=${EXTRA_ARGS[*]}")
fi

submit_sequential() {
    # Submit single training job that runs all experiments sequentially
    TRAIN_JOB=$(qsub "${QSUB_OPTS[@]}" "${QSUB_V[@]}" "$JOBS_DIR/train.pbs")
    echo "Training job submitted: $TRAIN_JOB"
    echo "$TRAIN_JOB"
}

submit_parallel() {
    # Query experiment count from run_experiments.py
    CONTAINER="$HOME/workspace/fyp/containers/nerfstudio.sif"
    FYP_UTILS="$HOME/workspace/fyp/fyp-utils"

    COUNT=$(apptainer exec \
        --bind "$HOME/workspace/fyp/sea-splatfacto:/opt/sea-splatfacto" \
        --bind "$HOME/workspace/fyp/nerfstudio:/opt/nerfstudio" \
        "$CONTAINER" \
        python "$FYP_UTILS/scripts/experiments/run_experiments.py" \
            --config "$FYP_UTILS/config/experiment_config.py" \
            --count \
            "${EXTRA_ARGS[@]}")

    if ! [[ "$COUNT" =~ ^[0-9]+$ ]]; then
        echo "Error: experiment count is not numeric (got: '$COUNT'). Apptainer/run_experiments may have errored." >&2
        exit 1
    fi
    if [ "$COUNT" -le 0 ]; then
        echo "Error: no experiments to run (count=$COUNT)" >&2
        exit 1
    fi

    LAST_INDEX=$((COUNT - 1))
    echo "Submitting array job with $COUNT experiments (indices 0-$LAST_INDEX)"

    TRAIN_JOB=$(qsub "${QSUB_OPTS[@]}" -J "0-$LAST_INDEX" "${QSUB_V[@]}" "$JOBS_DIR/train_array.pbs")
    echo "Array training job submitted: $TRAIN_JOB"
    echo "$TRAIN_JOB"
}

# Submit training
if [ "$PARALLEL" = true ]; then
    TRAIN_JOB=$(submit_parallel | tail -1)
else
    TRAIN_JOB=$(submit_sequential | tail -1)
fi

if [ "$TRAIN_ONLY" = true ]; then
    exit 0
fi

# Submit eval job — runs only after training succeeds
EVAL_JOB=$(qsub "${QSUB_OPTS[@]}" -W depend=afterok:"$TRAIN_JOB" "${QSUB_V[@]}" "$JOBS_DIR/eval.pbs")
echo "Eval job submitted: $EVAL_JOB (depends on $TRAIN_JOB)"

if [ "$RENDER" = true ]; then
    # Submit render job — runs only after eval succeeds
    RENDER_JOB=$(qsub "${QSUB_OPTS[@]}" -W depend=afterok:"$EVAL_JOB" "${QSUB_V[@]}" "$JOBS_DIR/render.pbs")
    echo "Render job submitted: $RENDER_JOB (depends on $EVAL_JOB)"
fi

if [ "$ANALYZE" = true ]; then
    # Validate: --analyze requires --render (analysis depends on render output)
    if [ "$RENDER" = false ]; then
        echo "Error: --analyze requires --render (analysis depends on render output)" >&2
        exit 1
    fi
    if [ -z "$BATCH_PREFIX" ]; then
        echo "Error: --analyze requires --batch-prefix <prefix>" >&2
        exit 1
    fi

    # Extract --dataset value from EXTRA_ARGS (positionally — matches the
    # token immediately following --dataset; --dataset=<value> form not supported).
    DATASET_FOR_ANALYZE=""
    for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
        if [[ "${EXTRA_ARGS[i]}" == "--dataset" ]] && (( i + 1 < ${#EXTRA_ARGS[@]} )); then
            DATASET_FOR_ANALYZE="${EXTRA_ARGS[i + 1]}"
            break
        fi
    done
    if [ -z "$DATASET_FOR_ANALYZE" ]; then
        echo "Error: --analyze requires --dataset in the arguments" >&2
        exit 1
    fi

    # Analyze is CPU-only — strip GPU queue (`-q <name>`) from QSUB_OPTS so the
    # analyze.pbs script picks up its own queue/resource defaults.
    ANALYZE_QSUB_OPTS=()
    skip_next=false
    for opt in "${QSUB_OPTS[@]}"; do
        if [ "$skip_next" = true ]; then
            skip_next=false
            continue
        fi
        if [[ "$opt" == "-q" ]]; then
            skip_next=true
            continue
        fi
        ANALYZE_QSUB_OPTS+=("$opt")
    done

    ANALYZE_JOB=$(qsub "${ANALYZE_QSUB_OPTS[@]}" \
        -W depend=afterok:"$RENDER_JOB" \
        -v BATCH_PREFIX="$BATCH_PREFIX",DATASET="$DATASET_FOR_ANALYZE" \
        "$JOBS_DIR/analyze.pbs")
    echo "Analyze job submitted: $ANALYZE_JOB (depends on $RENDER_JOB)"
fi

#!/bin/bash
# Submit train→eval→render dependency chain to PBS.
#
# Usage:
#   ./cluster/scripts/submit.sh                        # Submit train + eval
#   ./cluster/scripts/submit.sh --render               # Submit train + eval + render
#   ./cluster/scripts/submit.sh --train-only           # Submit training only
#   ./cluster/scripts/submit.sh --parallel             # Submit as PBS array job (1 experiment per sub-job)
#   ./cluster/scripts/submit.sh --filter torpedo       # Pass extra args to all jobs
#   ./cluster/scripts/submit.sh --paid                 # Use paid queue (consumes GPU-hour allocation)
#   ./cluster/scripts/submit.sh --paid --walltime 2:00:00  # Paid queue + custom walltime
#
# Run from the fyp-utils/ directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS_DIR="$SCRIPT_DIR/../jobs"

TRAIN_ONLY=false
PARALLEL=false
RENDER=false
PAID=false
FILTER=""
EXTRA_ARGS=""
QSUB_OPTS=""

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
        --paid)
            PAID=true
            shift
            ;;
        --walltime)
            QSUB_OPTS="$QSUB_OPTS -l walltime=$2"
            shift 2
            ;;
        --filter)
            FILTER="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Warn if no --filter specified (run_experiments.py will enforce the hard check)
if [ -z "$FILTER" ]; then
    echo "WARNING: No --filter specified. All datasets in config will be processed."
else
    EXTRA_ARGS="$EXTRA_ARGS --filter $FILTER"
fi

# Default to free tier queue unless --paid is specified
if [ "$PAID" = true ]; then
    echo "==> Using PAID queue. Jobs will consume GPU-hour allocation."
else
    QSUB_OPTS="$QSUB_OPTS -q auto_free"
    echo "==> Using free tier queue (auto_free). Use --paid to consume GPU-hour allocation."
fi

submit_sequential() {
    # Submit single training job that runs all experiments sequentially
    if [ -n "$EXTRA_ARGS" ]; then
        TRAIN_JOB=$(qsub $QSUB_OPTS -v EXTRA_ARGS="$EXTRA_ARGS" "$JOBS_DIR/train.pbs")
    else
        TRAIN_JOB=$(qsub $QSUB_OPTS "$JOBS_DIR/train.pbs")
    fi
    echo "Training job submitted: $TRAIN_JOB"
    echo "$TRAIN_JOB"
}

submit_parallel() {
    # Query experiment count from run_experiments.py
    CONTAINER=$HOME/workspace/fyp/containers/nerfstudio.sif
    FYP_UTILS=$HOME/workspace/fyp/fyp-utils

    COUNT=$(apptainer exec \
        --bind $HOME/workspace/fyp/sea-splatfacto:/opt/sea-splatfacto \
        --bind $HOME/workspace/fyp/nerfstudio:/opt/nerfstudio \
        "$CONTAINER" \
        python "$FYP_UTILS/scripts/experiments/run_experiments.py" \
            --config "$FYP_UTILS/config/experiment_config.py" \
            --count \
            ${EXTRA_ARGS:-})

    if [ "$COUNT" -le 0 ] 2>/dev/null; then
        echo "Error: no experiments to run (count=$COUNT)"
        exit 1
    fi

    LAST_INDEX=$((COUNT - 1))
    echo "Submitting array job with $COUNT experiments (indices 0-$LAST_INDEX)"

    if [ -n "$EXTRA_ARGS" ]; then
        TRAIN_JOB=$(qsub $QSUB_OPTS -J "0-$LAST_INDEX" -v EXTRA_ARGS="$EXTRA_ARGS" "$JOBS_DIR/train_array.pbs")
    else
        TRAIN_JOB=$(qsub $QSUB_OPTS -J "0-$LAST_INDEX" "$JOBS_DIR/train_array.pbs")
    fi
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
if [ -n "$EXTRA_ARGS" ]; then
    EVAL_JOB=$(qsub -W depend=afterok:"$TRAIN_JOB" -v EXTRA_ARGS="$EXTRA_ARGS" "$JOBS_DIR/eval.pbs")
else
    EVAL_JOB=$(qsub -W depend=afterok:"$TRAIN_JOB" "$JOBS_DIR/eval.pbs")
fi
echo "Eval job submitted: $EVAL_JOB (depends on $TRAIN_JOB)"

if [ "$RENDER" = true ]; then
    # Submit render job — runs only after eval succeeds
    if [ -n "$EXTRA_ARGS" ]; then
        RENDER_JOB=$(qsub -W depend=afterok:"$EVAL_JOB" -v EXTRA_ARGS="$EXTRA_ARGS" "$JOBS_DIR/render.pbs")
    else
        RENDER_JOB=$(qsub -W depend=afterok:"$EVAL_JOB" "$JOBS_DIR/render.pbs")
    fi
    echo "Render job submitted: $RENDER_JOB (depends on $EVAL_JOB)"
fi

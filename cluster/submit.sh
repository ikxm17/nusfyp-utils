#!/bin/bash
# Submit train→eval dependency chain to PBS.
#
# Usage:
#   ./cluster/submit.sh                        # Submit both train + eval
#   ./cluster/submit.sh --train-only           # Submit training only
#   ./cluster/submit.sh --filter torpedo       # Pass extra args to both jobs
#
# Run from the fyp-utils/ directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ONLY=false
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-only)
            TRAIN_ONLY=true
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Submit training job
if [ -n "$EXTRA_ARGS" ]; then
    TRAIN_JOB=$(qsub -v EXTRA_ARGS="$EXTRA_ARGS" "$SCRIPT_DIR/train.pbs")
else
    TRAIN_JOB=$(qsub "$SCRIPT_DIR/train.pbs")
fi
echo "Training job submitted: $TRAIN_JOB"

if [ "$TRAIN_ONLY" = true ]; then
    exit 0
fi

# Submit eval job — runs only after training succeeds
if [ -n "$EXTRA_ARGS" ]; then
    EVAL_JOB=$(qsub -W depend=afterok:"$TRAIN_JOB" -v EXTRA_ARGS="$EXTRA_ARGS" "$SCRIPT_DIR/eval.pbs")
else
    EVAL_JOB=$(qsub -W depend=afterok:"$TRAIN_JOB" "$SCRIPT_DIR/eval.pbs")
fi
echo "Eval job submitted: $EVAL_JOB (depends on $TRAIN_JOB)"

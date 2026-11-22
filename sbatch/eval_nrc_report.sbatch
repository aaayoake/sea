#!/bin/bash
#SBATCH --job-name=esconv-nrc-report
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

INPUT_FILE="${1:-eval_outputs/live_rollouts.jsonl}"
REPORT_FILE="${2:-eval_outputs/nrc_eval_report_${SLURM_JOB_ID}.json}"
REFERENCE_FILE="${3:-}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

ENV_PREFIX="${ENV_PREFIX:-$HOME/.conda/envs/sea}"
PYTHON="$ENV_PREFIX/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Python executable not found: $PYTHON (set ENV_PREFIX if the sea environment is elsewhere)" >&2
    exit 2
fi
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Rollout input file not found: $(pwd)/$INPUT_FILE" >&2
    exit 2
fi
if [[ -n "$REFERENCE_FILE" && ! -f "$REFERENCE_FILE" ]]; then
    echo "Validation reference file not found: $(pwd)/$REFERENCE_FILE" >&2
    exit 2
fi

mkdir -p "$(dirname "$REPORT_FILE")"
ARGS=(
    --input-file "$INPUT_FILE"
    --report-file "$REPORT_FILE"
)
if [[ -n "$REFERENCE_FILE" ]]; then
    ARGS+=(--reference-file "$REFERENCE_FILE")
fi
"$PYTHON" -u eval_nrc.py "${ARGS[@]}"

echo "NRC report saved to $(pwd)/$REPORT_FILE"

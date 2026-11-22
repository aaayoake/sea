#!/bin/bash
#SBATCH --job-name=esconv-eval-nrc
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SEEKER_JOB_ID="${1:-}"
SUPPORTER_JOB_ID="${2:-}"
if [[ ! "$SEEKER_JOB_ID" =~ ^[0-9]+$ || ! "$SUPPORTER_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Usage: sbatch sbatch/eval_nrc.sbatch SEEKER_TRAIN_JOB_ID SUPPORTER_TRAIN_JOB_ID" >&2
    exit 2
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export CONDA_PKGS_DIRS="$HOME/scratch/conda/pkgs"
export HF_HOME="/storage/ice1/0/4/lheck6/CS6604_ECE6256/hf_cache"
mkdir -p "$CONDA_PKGS_DIRS" eval_outputs

ENV_PREFIX="${ENV_PREFIX:-$HOME/.conda/envs/sea}"
PYTHON="$ENV_PREFIX/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Python executable not found: $PYTHON (set ENV_PREFIX if the sea environment is elsewhere)" >&2
    exit 2
fi

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [[ ! -d "$HF_HOME" ]]; then
    echo "Shared Hugging Face cache is not accessible: $HF_HOME" >&2
    exit 2
fi

SEEKER_ADAPTER="checkpoints/${SEEKER_JOB_ID}/seeker/best"
SUPPORTER_ADAPTER="checkpoints/${SUPPORTER_JOB_ID}/supporter/best"
for adapter_dir in "$SEEKER_ADAPTER" "$SUPPORTER_ADAPTER"; do
    if [[ ! -d "$adapter_dir" ]]; then
        echo "Adapter checkpoint not found: $(pwd)/$adapter_dir" >&2
        exit 2
    fi
done

# 1. Generate paired conversations from fresh synthetic scenarios.
"$PYTHON" run_live_rollouts.py \
    --model-id "$MODEL_ID" \
    --seeker-adapter "$SEEKER_ADAPTER" \
    --supporter-adapter "$SUPPORTER_ADAPTER" \
    --output-file "eval_outputs/live_rollouts.jsonl" \
    --num-conversations 50 \
    --min-turns 8 \
    --max-turns 12 \
    --seed 42

# 2. Compare NRC emotion profiles for base-seeker and SFT-seeker rollouts.
# Live rollouts have no human target, so this reports frequencies, not human-alignment Dice scores.
"$PYTHON" eval_nrc.py \
    --input-file "eval_outputs/live_rollouts.jsonl" \
    --report-file "eval_outputs/nrc_eval_report.json"
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
mkdir -p slurm/logs eval_outputs

module load anaconda3
conda activate sea

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the cluster-local base model directory}"
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "MODEL_PATH is not a directory: $MODEL_PATH" >&2
    exit 2
fi

export HF_HOME=~/scratch/hf_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# 1. Generate paired conversations from fresh synthetic scenarios.
python run_live_rollouts.py \
    --model-path "$MODEL_PATH" \
    --seeker-adapter "checkpoints/esconv_seeker_sft/best" \
    --supporter-adapter "checkpoints/esconv_supporter_sft/best" \
    --output-file "eval_outputs/live_rollouts.jsonl" \
    --num-conversations 50 \
    --min-turns 8 \
    --max-turns 12 \
    --seed 42

# 2. Compare NRC emotion profiles for base-seeker and SFT-seeker rollouts.
# Live rollouts have no human target, so this reports frequencies, not human-alignment Dice scores.
python eval_nrc.py \
    --input-file "eval_outputs/live_rollouts.jsonl" \
    --report-file "eval_outputs/nrc_eval_report.json"
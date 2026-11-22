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

cd "$(dirname "$0")/.."

export CONDA_PKGS_DIRS="$HOME/scratch/conda/pkgs"
export HF_HOME="$HOME/scratch/hf_cache"
mkdir -p "$CONDA_PKGS_DIRS" "$HF_HOME" eval_outputs

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sea

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# 1. Generate paired conversations from fresh synthetic scenarios.
python run_live_rollouts.py \
    --model-id "$MODEL_ID" \
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
#!/bin/bash
#SBATCH --job-name=esconv-sft
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROLE="${1:-}"
if [[ "$ROLE" != "seeker" && "$ROLE" != "supporter" ]]; then
    echo "Usage: sbatch sbatch/train_sft.sbatch seeker|supporter" >&2
    exit 2
fi

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

if [[ ! -f "esconv_train.json" ]]; then
    echo "Missing local dataset esconv_train.json; copy it to the project directory first." >&2
    exit 2
fi

python prepare_datasets.py
python prepare_esconv_supporter.py

python train_esconv_sft.py \
    --role "$ROLE" \
    --model-path "$MODEL_PATH" \
    --epochs 2 --batch-size 4 --grad-accum 4 --lr 1e-4 --max-seq-len 1536 --seed 42
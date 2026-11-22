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

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export CONDA_PKGS_DIRS="$HOME/scratch/conda/pkgs"
export HF_HOME="/storage/ice1/0/4/lheck6/CS6604_ECE6256/hf_cache"
mkdir -p "$CONDA_PKGS_DIRS"

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

if [[ ! -d "$HF_HOME" ]]; then
    echo "Shared Hugging Face cache is not accessible: $HF_HOME" >&2
    exit 2
fi

if [[ "$ROLE" == "seeker" ]]; then
    TRAIN_FILE="data/esconv_sft_train.jsonl"
    VAL_FILE="data/esconv_sft_val.jsonl"
else
    TRAIN_FILE="data/esconv_supporter_sft_train.jsonl"
    VAL_FILE="data/esconv_supporter_sft_val.jsonl"
fi
CHECKPOINT_DIR="checkpoints/${SLURM_JOB_ID:-manual}/${ROLE}"

for dataset_file in "$TRAIN_FILE" "$VAL_FILE"; do
    if [[ ! -s "$dataset_file" ]]; then
        echo "Missing or empty prepared dataset: $(pwd)/$dataset_file" >&2
        echo "Run prepare_datasets.py and prepare_esconv_supporter.py once before submitting training jobs." >&2
        exit 2
    fi
done

"$PYTHON" -u train_esconv_sft.py \
    --role "$ROLE" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --model-id "$MODEL_ID" \
    --epochs 2 --batch-size 1 --grad-accum 16 --lr 1e-4 --max-seq-len 1024 --seed 42
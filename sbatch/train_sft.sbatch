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
export HF_HOME="$HOME/scratch/hf_cache"
mkdir -p "$CONDA_PKGS_DIRS" "$HF_HOME"

module load anaconda3
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

if [[ ! -f "esconv_train.json" ]]; then
    echo "Missing local dataset: $(pwd)/esconv_train.json (submit from the project directory or set --chdir)" >&2
    exit 2
fi

"$PYTHON" prepare_datasets.py
"$PYTHON" prepare_esconv_supporter.py

"$PYTHON" train_esconv_sft.py \
    --role "$ROLE" \
    --model-id "$MODEL_ID" \
    --epochs 2 --batch-size 4 --grad-accum 4 --lr 1e-4 --max-seq-len 1536 --seed 42
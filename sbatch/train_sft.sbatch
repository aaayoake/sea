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

export HF_HOME=~/scratch/hf_cache
export TOKENIZERS_PARALLELISM=false

python prepare_datasets.py
python prepare_esconv_supporter.py

python train_esconv_sft.py \
    --role "$ROLE" \
    --model-id "meta-llama/Llama-3.1-8B-Instruct" \
    --epochs 2 --batch-size 4 --grad-accum 4 --lr 1e-4 --max-seq-len 1536 --seed 42
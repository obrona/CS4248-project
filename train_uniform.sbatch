#!/bin/bash
#SBATCH --job-name=bart-uniform
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=uniform-%j.out

set -e

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gec

cd "$HOME/CS4248-project"

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi

srun --cpu-bind=none python -u train_frequency.py \
    --max-train-examples 0 \
    --epochs 10 \
    --batch-size 8 \
    --accumulation 4 \
    --max-length 512 \
    --eval-batch-size 4 \
    --alpha 3 \
    --beta 0 \
    --gold-m2 data/dev/ABCN.dev.gold.bea19.m2 \
    --output-dir runs/bart_uniform_edit_full

echo "Job finished: $(date)"

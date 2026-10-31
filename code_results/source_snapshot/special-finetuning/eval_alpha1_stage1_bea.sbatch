#!/bin/bash
#SBATCH --job-name=alpha1-s1-bea
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=output/alpha1-s1-bea-%j.out
#SBATCH --error=output/alpha1-s1-bea-%j.err

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate gec
cd ~/CS4248-tl2-alpha1/special-finetuning

export HF_HOME="$PWD/output/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python -u eval-bea-2019.py \
  --run_dir output/a1_b0.5_w10 \
  --batch_size 16 \
  --num_beams 5 \
  --max_len 256

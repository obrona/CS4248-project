#!/bin/bash
#SBATCH --job-name=tl2-alpha1-s2
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=output/alpha1-stage2-%j.out
#SBATCH --error=output/alpha1-stage2-%j.err

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate gec
cd ~/CS4248-tl2-alpha1/special-finetuning

export HF_HOME="$PWD/output/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python -u finetune-2.py --run_dir output/a1_b0.5_w10

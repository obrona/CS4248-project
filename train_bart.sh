#!/bin/bash
#SBATCH --job-name=bart-base-gec
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=bart-base-%j.log

set -euo pipefail

# Submit from the transferred project directory: sbatch train_bart.sh
cd "${SLURM_SUBMIT_DIR:?Submit this script using sbatch from the project directory}"

# Change this if Python 3.12 has a different executable name on your cluster.
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
export HF_HOME="$PWD/.cache/huggingface"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ ! -x .venv/bin/python ]]; then
    "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
python -c 'import sys; assert sys.version_info[:2] == (3, 12), "The venv must use Python 3.12"'

echo "Installing training packages..."
python -m pip install --upgrade pip
# Use the CUDA 11.8 build, which supports the cluster's A100 GPUs.
python -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt

echo "Starting BART-base training..."
srun python -u train_bart.py
echo "Done. Final weights and evaluation results: outputs/bart-base-gec/"

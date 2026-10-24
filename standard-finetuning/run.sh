#!/bin/bash
# Slurm job for standard fine-tuning of facebook/bart-base (and CoNLL-14 eval).
#
# Submit from the standard-finetuning/ directory:
#   cd standard-finetuning
#   sbatch run.sh            # fine-tune (1 epoch, all training data)
#   sbatch run.sh eval       # evaluate on CoNLL-2014 with M2Scorer
# Extra arguments are passed to the Python script, e.g.
#   sbatch run.sh train --batch_size 16
#
#SBATCH --job-name=bart-gec
#SBATCH --gpus=h100-47:1          # alternatives: a100-80:1, h100-96:1, h100-47:1, titanrtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=output/slurm-%x-%j.out
#SBATCH --error=output/slurm-%x-%j.err

set -euo pipefail

# Slurm runs a copy of this script, so locate the project from the submit directory.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ -f "$SUBMIT_DIR/finetune.py" ]]; then
    SCRIPT_DIR="$SUBMIT_DIR"
elif [[ -f "$SUBMIT_DIR/standard-finetuning/finetune.py" ]]; then
    SCRIPT_DIR="$SUBMIT_DIR/standard-finetuning"
else
    echo "Cannot find finetune.py; submit from the standard-finetuning/ directory." >&2
    exit 1
fi
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$REPO_ROOT/.venv"
cd "$SCRIPT_DIR"
mkdir -p output

MODE="${1:-train}"
if [[ $# -gt 0 ]]; then shift; fi

# ---- Environment setup (idempotent) ----
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "Creating virtual environment at $VENV"
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet torch "transformers>=4.46,<5" datasets accelerate sentencepiece tqdm

export HF_HOME="$SCRIPT_DIR/output/hf_cache"
export TOKENIZERS_PARALLELISM=false

echo "Job ${SLURM_JOB_ID:-local} on $(hostname), mode=$MODE"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

case "$MODE" in
    train) python finetune.py "$@" ;;
    eval)  python eval-conll14st.py "$@" ;;
    *)     echo "Unknown mode '$MODE' (use 'train' or 'eval')" >&2; exit 1 ;;
esac

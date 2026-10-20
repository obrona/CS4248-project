#!/bin/bash
# Slurm job for category-weighted fine-tuning of facebook/bart-base (and CoNLL-14 eval).
#
# Submit from the special-finetuning/ directory:
#   cd special-finetuning
#   sbatch run.sh            # fine-tune with alpha=3 beta=0.5 w_max=10 -> output/a3_b0.5_w10/
#   sbatch run.sh eval       # evaluate output/a3_b0.5_w10 on CoNLL-2014 with M2Scorer
#   sbatch run.sh eval --run_dir output/a2_b1_w10   # evaluate another run
# Extra arguments are passed to the Python script, e.g.
#   sbatch run.sh train --alpha 2 --beta 1
#
#SBATCH --job-name=bart-gec-weighted
#SBATCH --gpus=h100-47:1          # alternatives: a100-80:1, h100-96:1, h100-47:1, titanrtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=output/slurm-%x-%j.out
#SBATCH --error=output/slurm-%x-%j.err

set -euo pipefail

# Slurm runs a copy of this script, so locate the project from the submit directory.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ -f "$SUBMIT_DIR/special-finetune.py" ]]; then
    SCRIPT_DIR="$SUBMIT_DIR"
elif [[ -f "$SUBMIT_DIR/special-finetuning/special-finetune.py" ]]; then
    SCRIPT_DIR="$SUBMIT_DIR/special-finetuning"
else
    echo "Cannot find special-finetune.py; submit from the special-finetuning/ directory." >&2
    exit 1
fi
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$REPO_ROOT/.venv"
cd "$SCRIPT_DIR"
mkdir -p output

# Category labels come from the M2 files (gitignored, transfer them separately).
if [[ ! -f "$REPO_ROOT/data/raw/lang8.train.auto.bea19.m2" ]]; then
    echo "Missing M2 training files in $REPO_ROOT/data/raw/" >&2
    exit 1
fi

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
    train) python special-finetune.py "$@" ;;
    eval)  python special-eval.py "$@" ;;
    *)     echo "Unknown mode '$MODE' (use 'train' or 'eval')" >&2; exit 1 ;;
esac

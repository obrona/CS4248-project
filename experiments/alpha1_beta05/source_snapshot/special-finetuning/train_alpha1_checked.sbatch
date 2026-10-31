#!/bin/bash
#SBATCH --job-name=tl2-alpha1
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=output/alpha1-%j.out
#SBATCH --error=output/alpha1-%j.err

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate gec
cd ~/CS4248-tl2-alpha1/special-finetuning

export HF_HOME="$PWD/output/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python - <<'PY'
import torch
import finetune as ft

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "GPU unavailable"

sources, targets, word_cats, eos_cats, sizes, mismatches = (
    ft.load_training_data(ft.REPO_ROOT / "data/raw")
)

weights, counts = ft.category_weights(
    word_cats, eos_cats,
    alpha=1.0, beta=0.5, w_max=10.0
)

assert sources, "No training pairs loaded"
assert weights, "No annotated edits found"
assert all(w == 1.0 for w in weights.values()), weights
assert sum(mismatches.values()) == 0, mismatches

print(f"VALIDATION PASSED: {len(sources):,} training pairs")
print("Category weights:", weights)
PY

echo "Validation passed. Starting stage 1."
python -u finetune.py --alpha 1 --beta 0.5 --w_max 10

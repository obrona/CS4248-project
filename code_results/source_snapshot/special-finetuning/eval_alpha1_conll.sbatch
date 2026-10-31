#!/bin/bash
#SBATCH --job-name=alpha1-conll
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=output/alpha1-conll-%j.out
#SBATCH --error=output/alpha1-conll-%j.err

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate gec
cd ~/CS4248-tl2-alpha1/special-finetuning

export HF_HOME="$PWD/output/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python - <<'PY'
from pathlib import Path
import importlib.util
import subprocess
import sys
import torch

assert torch.cuda.is_available(), "GPU unavailable"

scorer = Path.home() / "gec_project/m2scorer_scripts/m2scorer.py"
python2 = (
    Path.home()
    / "miniforge3/envs/gec_m2_py2_fixed/bin/python2.7"
)
assert scorer.is_file(), f"Missing scorer: {scorer}"
assert python2.is_file(), f"Missing Python 2: {python2}"
subprocess.run([str(python2), "--version"], check=True)

spec = importlib.util.spec_from_file_location(
    "conll_eval", Path("eval-conll14st.py")
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)

# Use the already-installed scorer instead of downloading another copy.
evaluation.ensure_m2scorer = lambda: scorer
evaluation.ensure_python2 = lambda: str(python2)

sys.argv = [
    "eval-conll14st.py",
    "--run_dir", "output/a1_b0.5_w10/stage2",
    "--batch_size", "16",
    "--num_beams", "5",
    "--max_len", "256",
]
evaluation.main()
PY

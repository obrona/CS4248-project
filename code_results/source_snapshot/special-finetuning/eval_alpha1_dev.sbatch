#!/bin/bash
#SBATCH --job-name=alpha1-dev
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=output/alpha1-dev-%j.out
#SBATCH --error=output/alpha1-dev-%j.err

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
import torch
import spacy
from transformers import AutoTokenizer, BartForConditionalGeneration

# Reuse tl2's actual generation function.
spec = importlib.util.spec_from_file_location(
    "tl2_eval", Path("eval-conll14st.py")
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)

model_dir = Path("output/a1_b0.5_w10/stage2/bart-base-gec")
out = Path("output/a1_b0.5_w10/stage2/eval_dev_errant")
gold = Path.home() / "gec_project/dev data/ABCN.dev.gold.bea19.m2"
out.mkdir(parents=True, exist_ok=True)

assert torch.cuda.is_available(), "GPU unavailable"
assert (model_dir / "model.safetensors").is_file(), "Model missing"
assert gold.is_file(), "Dev gold missing"

blocks = evaluation.read_m2(gold)
sources = [block["source"] for block in blocks]
print(f"Dev sentences: {len(sources):,}")
print(f"Model: {model_dir.resolve()}")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = BartForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=dtype
).to("cuda").eval()

predictions = evaluation.generate(
    model, tokenizer, sources,
    batch_size=16, num_beams=5, max_len=256, device="cuda"
)
assert len(predictions) == len(sources)

(out / "sources.txt").write_text(
    "\n".join(sources) + "\n", encoding="utf-8"
)
(out / "predictions.raw.txt").write_text(
    "\n".join(predictions) + "\n", encoding="utf-8"
)

# Preserve gold source tokenization; tokenize model predictions.
nlp = spacy.load("en_core_web_sm")
tokenized = [
    " ".join(token.text for token in nlp.make_doc(text))
    for text in predictions
]
(out / "predictions.txt").write_text(
    "\n".join(tokenized) + "\n", encoding="utf-8"
)
print("Predictions saved.")
PY

DEV_OUT="output/a1_b0.5_w10/stage2/eval_dev_errant"
GOLD="$HOME/gec_project/dev data/ABCN.dev.gold.bea19.m2"

errant_parallel \
  -orig "$DEV_OUT/sources.txt" \
  -cor "$DEV_OUT/predictions.txt" \
  -out "$DEV_OUT/predictions.m2"

errant_compare \
  -hyp "$DEV_OUT/predictions.m2" \
  -ref "$GOLD" \
  > "$DEV_OUT/overall.txt"

errant_compare \
  -hyp "$DEV_OUT/predictions.m2" \
  -ref "$GOLD" \
  -cat 3 \
  > "$DEV_OUT/categories.txt"

cat "$DEV_OUT/overall.txt"
cat "$DEV_OUT/categories.txt"

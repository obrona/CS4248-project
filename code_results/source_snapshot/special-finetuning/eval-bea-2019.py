"""Generate corrections for the BEA-2019 shared task test set with a category-weighted BART model.

Same as standard-finetuning/eval-bea-2019.py: only the source sentences of the
BEA-2019 test set are public (bea-2019-test-data/ABCN.test.bea19.orig), so nothing
is scored here. A run of finetune.py lives in output/a{alpha}_b{beta}_w{w_max}/; the
corrected sentences are written one per line to <run_dir>/eval_bea19/predictions.txt,
and zipped into <run_dir>/eval_bea19/submission.zip for upload to Codabench:
https://www.codabench.org/competitions/10960/

Usage:
    python eval-bea-2019.py                               # output/a3_b0.5_w10 (default run)
    python eval-bea-2019.py --run_dir output/a2_b1_w10
    python eval-bea-2019.py --limit 20                    # smoke test on the first 20 sentences
"""

import argparse
import json
import os
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"
TEST_SRC = REPO_ROOT / "bea-2019-test-data" / "ABCN.test.bea19.orig"

os.environ.setdefault("HF_HOME", str(OUTPUT_DIR / "hf_cache"))

import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from transformers import AutoTokenizer, BartForConditionalGeneration  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=Path, default=OUTPUT_DIR / "a3_b0.5_w10",
                   help="output folder of a finetune.py run")
    p.add_argument("--model_dir", type=Path, default=None, help="default: <run_dir>/bart-base-gec")
    p.add_argument("--eval_dir", type=Path, default=None, help="default: <run_dir>/eval_bea19")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_beams", type=int, default=5)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--limit", type=int, default=None, help="only correct the first N sentences")
    args = p.parse_args()
    args.model_dir = args.model_dir or args.run_dir / "bart-base-gec"
    args.eval_dir = args.eval_dir or args.run_dir / "eval_bea19"
    return args


@torch.no_grad()
def generate(model, tokenizer, sources, batch_size, num_beams, max_len, device):
    # Sort by length so batches have little padding, then restore the original order.
    order = sorted(range(len(sources)), key=lambda i: len(sources[i].split()))
    preds = [""] * len(sources)
    for b in tqdm(range(0, len(order), batch_size), desc="Generating"):
        idx = [i for i in order[b:b + batch_size] if sources[i].strip()]
        if not idx:
            continue
        enc = tokenizer(
            [sources[i] for i in idx],
            max_length=max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        # bart-base's config sets no_repeat_ngram_size=3 (a summarization default); GEC
        # must be free to copy repeated n-grams from the source, so disable it.
        out = model.generate(
            **enc, num_beams=num_beams, max_new_tokens=max_len, early_stopping=True, no_repeat_ngram_size=0
        )
        texts = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for i, t in zip(idx, texts):
            preds[i] = " ".join(t.split())
    return preds


def main():
    args = parse_args()
    eval_dir = args.eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)

    sources = TEST_SRC.read_text(encoding="utf-8").rstrip("\n").split("\n")
    if args.limit is not None:
        sources = sources[:args.limit]
    print(f"Loaded {len(sources)} BEA-2019 test sentences from {TEST_SRC}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = BartForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype=dtype).to(device).eval()

    preds = generate(model, tokenizer, sources, args.batch_size, args.num_beams, args.max_len, device)

    pred_path = eval_dir / "predictions.txt"
    pred_path.write_text("\n".join(preds) + "\n", encoding="utf-8")
    (eval_dir / "sources.txt").write_text("\n".join(sources) + "\n", encoding="utf-8")
    with open(eval_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for i, (s, p) in enumerate(zip(sources, preds)):
            f.write(json.dumps({"id": i, "source": s, "prediction": p, "changed": p != s}, ensure_ascii=False) + "\n")
    # Codabench expects a zip holding a single text file of corrected sentences.
    with zipfile.ZipFile(eval_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname=pred_path.name)

    summary = {
        "test_set": str(TEST_SRC.relative_to(REPO_ROOT)),
        "num_sentences": len(sources),
        "num_changed": sum(p != s for p, s in zip(preds, sources)),
        "model_dir": str(args.model_dir),
        "generation": {"num_beams": args.num_beams, "max_len": args.max_len, "batch_size": args.batch_size},
    }
    with open(eval_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Upload {eval_dir / 'submission.zip'} to https://www.codabench.org/competitions/10960/")


if __name__ == "__main__":
    main()

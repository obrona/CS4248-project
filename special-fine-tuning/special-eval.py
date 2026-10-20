"""Evaluate a category-weighted BART model on the CoNLL-2014 shared task test set.

Same evaluation as standard-finetuning/standard-eval.py: generates corrections for
every sentence of conll14st-test-data/noalt/official-2014.combined.m2 and scores
them with the official M2Scorer (P / R / F0.5). A run of special-finetune.py lives
in output/a{alpha}_b{beta}_w{w_max}/; everything needed for later analysis is
written to <run_dir>/eval_conll14/.

M2Scorer is Python 2 only: it is cloned into output/tools/, and a Python 2.7
interpreter is created there with micromamba if `python2` is not on PATH.

Usage:
    python special-eval.py                               # output/a3_b0.5_w10 (default run)
    python special-eval.py --run_dir output/a2_b1_w10
    python special-eval.py --limit 20                    # smoke test on the first 20 sentences
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"
TOOLS_DIR = OUTPUT_DIR / "tools"
GOLD_M2 = REPO_ROOT / "conll14st-test-data" / "noalt" / "official-2014.combined.m2"

os.environ.setdefault("HF_HOME", str(OUTPUT_DIR / "hf_cache"))

import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from transformers import AutoTokenizer, BartForConditionalGeneration  # noqa: E402

M2SCORER_REPO = "https://github.com/nusnlp/m2scorer.git"
MICROMAMBA_URL = "https://micro.mamba.pm/api/micromamba/linux-64/latest"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=Path, default=OUTPUT_DIR / "a3_b0.5_w10",
                   help="output folder of a special-finetune.py run")
    p.add_argument("--model_dir", type=Path, default=None, help="default: <run_dir>/bart-base-gec")
    p.add_argument("--eval_dir", type=Path, default=None, help="default: <run_dir>/eval_conll14")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_beams", type=int, default=5)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--limit", type=int, default=None, help="only evaluate the first N sentences")
    args = p.parse_args()
    args.model_dir = args.model_dir or args.run_dir / "bart-base-gec"
    args.eval_dir = args.eval_dir or args.run_dir / "eval_conll14"
    return args


def read_m2(path):
    """Return a list of blocks: {"source": str, "edits": [{"start", "end", "type", "correction", "annotator"}]}."""
    blocks = []
    with open(path, encoding="utf-8") as f:
        raw = f.read().strip().split("\n\n")
    for chunk in raw:
        lines = chunk.split("\n")
        assert lines[0].startswith("S "), f"Malformed m2 block: {lines[0]!r}"
        edits = []
        for line in lines[1:]:
            fields = line[2:].split("|||")
            start, end = map(int, fields[0].split())
            edits.append({
                "start": start,
                "end": end,
                "type": fields[1],
                "correction": fields[2],
                "annotator": int(fields[5]),
            })
        blocks.append({"source": lines[0][2:], "edits": edits, "raw": chunk})
    return blocks


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
        out = model.generate(**enc, num_beams=num_beams, max_new_tokens=max_len, early_stopping=True)
        texts = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for i, t in zip(idx, texts):
            preds[i] = " ".join(t.split())
    return preds


def ensure_m2scorer():
    m2_dir = TOOLS_DIR / "m2scorer"
    if not (m2_dir / "scripts" / "m2scorer.py").exists():
        TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", M2SCORER_REPO, str(m2_dir)], check=True)
    return m2_dir / "scripts" / "m2scorer.py"


def ensure_python2():
    for name in ("python2", "python2.7"):
        exe = shutil.which(name)
        if exe:
            return exe
    py27 = TOOLS_DIR / "py27" / "bin" / "python2.7"
    if py27.exists():
        return str(py27)

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    micromamba = TOOLS_DIR / "bin" / "micromamba"
    if not micromamba.exists():
        print("Downloading micromamba to create a Python 2.7 environment for M2Scorer")
        subprocess.run(
            f"curl -Ls {MICROMAMBA_URL} | tar -xj -C '{TOOLS_DIR}' bin/micromamba",
            shell=True,
            check=True,
        )
    env = dict(os.environ, MAMBA_ROOT_PREFIX=str(TOOLS_DIR / "mamba_root"))
    subprocess.run(
        [str(micromamba), "create", "-y", "-p", str(TOOLS_DIR / "py27"), "-c", "conda-forge", "python=2.7"],
        check=True,
        env=env,
    )
    return str(py27)


def run_m2scorer(pred_path, gold_path, verbose_path):
    scorer = ensure_m2scorer()
    python2 = ensure_python2()
    print(f"Running M2Scorer ({python2} {scorer})")
    result = subprocess.run(
        [python2, str(scorer), "-v", str(pred_path), str(gold_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    verbose_path.write_text(result.stdout)
    scores = {}
    for key, name in (("precision", "Precision"), ("recall", "Recall"), ("f0.5", "F_0.5")):
        m = re.search(rf"^{re.escape(name)}\s*:\s*([0-9.]+)", result.stdout, re.MULTILINE)
        scores[key] = float(m.group(1)) if m else None
    return scores


def main():
    args = parse_args()
    eval_dir = args.eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)

    blocks = read_m2(GOLD_M2)
    gold_path = GOLD_M2
    if args.limit is not None:
        blocks = blocks[:args.limit]
        gold_path = eval_dir / "gold_subset.m2"
        gold_path.write_text("\n\n".join(b["raw"] for b in blocks) + "\n\n", encoding="utf-8")
    sources = [b["source"] for b in blocks]
    print(f"Loaded {len(sources)} CoNLL-2014 test sentences from {GOLD_M2}")

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
        for i, (b, p) in enumerate(zip(blocks, preds)):
            row = {"id": i, "source": b["source"], "prediction": p, "changed": p != b["source"], "gold_edits": b["edits"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    scores = run_m2scorer(pred_path, gold_path, eval_dir / "m2scorer_verbose.txt")
    summary = {
        "test_set": str(GOLD_M2.relative_to(REPO_ROOT)),
        "num_sentences": len(sources),
        "num_changed": sum(p != s for p, s in zip(preds, sources)),
        "m2scorer": scores,
        "model_dir": str(args.model_dir),
        "generation": {"num_beams": args.num_beams, "max_len": args.max_len, "batch_size": args.batch_size},
    }
    with open(eval_dir / "scores.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

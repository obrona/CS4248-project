"""Category-aware weighted fine-tuning of facebook/bart-base for grammatical error correction.

Same data, model and hyperparameters as standard-finetuning/standard-finetune.py, but
the token-level cross-entropy is weighted: target tokens produced by a correction get
a weight that depends on the correction's ERRANT error category (rarer categories get
larger weights), copied tokens keep weight 1.

    L = sum_t w_t * CE_t / sum_t w_t

Category weights (n_c = number of edited target positions of category c):

    r_c = n_c ** -beta
    w_c = min(1 + (alpha - 1) * r_c / sum_c' p_c' r_c', w_max),   p_c = n_c / sum_c' n_c'

so the average weight of an edited token is alpha (before capping), beta=0 gives the
same weight to every edit, and alpha=1 gives back the standard loss.

Training pairs are rebuilt from the BEA-2019 M2 files (data/raw/*.m2) so that every
target word knows which edit produced it; they are checked against the *-src-tgt files
used by the baseline. No evaluation is run here -- see run.sh eval.

Usage:
    python special-finetune.py                                  # alpha=3, beta=0.5
    python special-finetune.py --alpha 2 --beta 1 --w_max 5
    python special-finetune.py --max_train_samples 64 --max_steps 2   # smoke test
"""

import argparse
import json
import os
import platform
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"

# Cache HF downloads inside output/ so the run is self-contained.
os.environ.setdefault("HF_HOME", str(OUTPUT_DIR / "hf_cache"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import transformers  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402

MODEL_NAME = "facebook/bart-base"

# (M2 stem, src-tgt directory used by the baseline, drop sentences without corrections).
# The baseline's FCE and NUCLE src-tgt files only contain sentences with at least one
# correction; Lang-8 and W&I keep every sentence.
TRAIN_FILES = [
    ("fce.train.gold.bea19", "fce-src-tgt", True),
    ("fce.dev.gold.bea19", "fce-src-tgt", True),
    ("lang8.train.auto.bea19", "lang8-src-tgt", False),
    ("nucle.train.gold.bea19", "nucle-src-tgt", True),
    ("A.train.gold.bea19", "w&-src-tgt", False),
    ("B.train.gold.bea19", "w&-src-tgt", False),
    ("C.train.gold.bea19", "w&-src-tgt", False),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", default=MODEL_NAME)
    p.add_argument("--output_dir", type=Path, default=None,
                   help="default: output/a{alpha}_b{beta}_w{w_max}")
    p.add_argument("--m2_dir", type=Path, default=REPO_ROOT / "data" / "raw")
    # Category-aware weighting
    p.add_argument("--alpha", type=float, default=3.0, help="average weight of an edited token (1 = standard loss)")
    p.add_argument("--beta", type=float, default=0.5, help="inverse-frequency exponent (0 = same weight for all edits)")
    p.add_argument("--w_max", type=float, default=10.0, help="cap on any category weight")
    p.add_argument("--max_len", type=int, default=256, help="max source/target length in subword tokens")
    p.add_argument("--batch_size", type=int, default=32, help="per-device train batch size")
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataloader_workers", type=int, default=0)
    p.add_argument("--num_proc", type=int, default=min(8, os.cpu_count() or 1), help="workers for tokenization")
    # Debug / smoke-test options
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=-1)
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = OUTPUT_DIR / f"a{args.alpha:g}_b{args.beta:g}_w{args.w_max:g}"
    return args


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


# ---------------------------------------------------------------------------
# M2 -> (source words, edits) -> (target words, per-word category)
# ---------------------------------------------------------------------------

def coarse_category(error_type):
    """'R:VERB:SVA' -> 'VERB:SVA'; 'UNK' stays 'UNK'."""
    return error_type.split(":", 1)[1] if error_type[1:2] == ":" else error_type


def read_m2(path):
    """Yield (source_words, edits) per sentence; edits = [(start, end, category, correction_words)].

    Only annotator 0 is used (the BEA-2019 training files have a single annotator).
    Edits are stable-sorted by start offset, so insertions at the same position keep
    their file order -- this is the order the baseline's .tgt files were built with.
    """
    with open(path, encoding="utf-8") as f:
        chunks = f.read().strip().split("\n\n")
    for chunk in chunks:
        lines = chunk.split("\n")
        assert lines[0].startswith("S"), f"Malformed m2 block in {path}: {lines[0]!r}"
        # split(" "), not split(): M2 indices count the empty tokens some Lang-8 lines contain.
        src_words = lines[0][2:].split(" ")
        edits = []
        for line in lines[1:]:
            span, error_type, correction, *_, annotator = line[2:].split("|||")
            if error_type == "noop" or annotator.strip() != "0":
                continue
            start, end = map(int, span.split())
            corr = [] if correction in ("", "-NONE-") else correction.split(" ")
            edits.append((start, end, coarse_category(error_type), corr))
        edits.sort(key=lambda e: e[0])
        yield src_words, edits


def apply_edits(src_words, edits):
    """Apply edits and record which category produced each target word.

    Returns (tgt_words, word_cats, eos_cat): word_cats[i] is None for copied words.
    A deletion produces no target word, so its category goes on the next target word
    (the step where the model has to skip the deleted word instead of copying it), or
    on EOS when nothing follows. UNK edits leave the text unchanged.
    """
    tgt, cats, pending, i = [], [], [], 0

    def emit(word, cat_list):
        nonlocal pending
        tgt.append(word)
        cats.append(cat_list + pending or None)
        pending = []

    for start, end, cat, corr in edits:
        while i < start:
            emit(src_words[i], [])
            i += 1
        if cat == "UNK":
            continue  # UNK leaves the source span in place; it is copied below
        if corr:
            for word in corr:
                emit(word, [cat])
        else:
            pending = pending + [cat]
        i = max(i, end)
    while i < len(src_words):
        emit(src_words[i], [])
        i += 1
    return tgt, cats, pending or None


def load_training_data(m2_dir):
    """Rebuild the baseline's training pairs from M2 with per-word categories."""
    sources, word_cats, targets, sizes, mismatches = [], [], [], {}, {}
    eos_cats = []
    for stem, subdir, drop_unchanged in TRAIN_FILES:
        base_src = read_lines(REPO_ROOT / subdir / f"{stem}.src")
        base_tgt = read_lines(REPO_ROOT / subdir / f"{stem}.tgt")
        rows = []
        for src_words, edits in read_m2(m2_dir / f"{stem}.m2"):
            tgt_words, cats, eos_cat = apply_edits(src_words, edits)
            if drop_unchanged and tgt_words == src_words:
                continue
            rows.append((" ".join(src_words), tgt_words, cats, eos_cat))
        assert len(rows) == len(base_src) == len(base_tgt), (
            f"{stem}: {len(rows)} pairs from M2 vs {len(base_src)} baseline lines")
        bad = 0
        for (src, tgt_words, cats, eos_cat), b_src, b_tgt in zip(rows, base_src, base_tgt):
            assert src == b_src, f"{stem}: source mismatch\n  m2:   {src}\n  base: {b_src}"
            if " ".join(tgt_words) != b_tgt:
                # Keep the baseline target so both models see identical data; no edit
                # weights for this sentence since we can't place them reliably.
                bad += 1
                tgt_words = b_tgt.split(" ")
                cats, eos_cat = [None] * len(tgt_words), None
            sources.append(src)
            targets.append(tgt_words)
            word_cats.append(cats)
            eos_cats.append(eos_cat)
        sizes[stem] = len(rows)
        mismatches[stem] = bad
        print(f"Loaded {len(rows):>9,} pairs from {stem}.m2 ({bad} target mismatches vs {subdir})")
    print(f"Total: {len(sources):,} pairs")
    return sources, targets, word_cats, eos_cats, sizes, mismatches


def category_weights(word_cats, eos_cats, alpha, beta, w_max):
    """Inverse-frequency (tempered by beta) weights, normalised so edited tokens average alpha."""
    counts = Counter()
    for cats in word_cats:
        for c in cats:
            if c:
                counts.update(c)
    for c in eos_cats:
        if c:
            counts.update(c)
    total = sum(counts.values())
    r = {c: n ** -beta for c, n in counts.items()}
    z = sum(n / total * r[c] for c, n in counts.items())
    weights = {c: min(1.0 + (alpha - 1.0) * r[c] / z, w_max) for c in counts}
    return weights, counts


def word_weights(cats, weights):
    """Category lists -> one weight per word (max if a word is touched by several edits)."""
    return [max(weights[c] for c in cat) if cat else 1.0 for cat in cats]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class WeightedSeq2SeqCollator(DataCollatorForSeq2Seq):
    """DataCollatorForSeq2Seq that also pads `label_weights` (with 0) to the label length."""

    def __call__(self, features, return_tensors=None):
        weights = [f.pop("label_weights") for f in features]
        batch = super().__call__(features, return_tensors)
        length = batch["labels"].shape[1]
        batch["label_weights"] = torch.tensor(
            [list(w) + [0.0] * (length - len(w)) for w in weights], dtype=torch.float32)
        return batch


class WeightedLossTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with the category-weighted token-level cross-entropy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Our loss is a per-micro-batch mean, so let Trainer divide it by grad_accum.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weights = inputs.pop("label_weights")
        labels = inputs.pop("labels")  # decoder_input_ids were already built by the collator
        outputs = model(**inputs)
        logits = outputs.logits
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            ignore_index=-100, reduction="none",
        ).view_as(labels)
        w = weights.to(ce.dtype) * (labels != -100)
        # Normalise by the total weight (not the token count) so the gradient scale
        # matches the baseline for every alpha.
        loss = (w * ce).sum() / w.sum().clamp_min(1.0)
        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()
    out_dir = args.output_dir
    model_dir = out_dir / "bart-base-gec"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    transformers.set_seed(args.seed)

    sources, targets, word_cats, eos_cats, sizes, mismatches = load_training_data(args.m2_dir)
    cat_weights, cat_counts = category_weights(word_cats, eos_cats, args.alpha, args.beta, args.w_max)
    print("Category weights (edited target positions, weight):")
    for c, n in cat_counts.most_common():
        print(f"  {c:12s}{n:>9,}  {cat_weights[c]:.2f}")

    dataset = Dataset.from_dict({
        "src": sources,
        "tgt": [" ".join(t) for t in targets],
        "word_weights": [word_weights(c, cat_weights) for c in word_cats],
        "eos_weight": [max(cat_weights[c] for c in e) if e else 1.0 for e in eos_cats],
    })
    del sources, targets, word_cats, eos_cats
    if args.max_train_samples is not None:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BartForConditionalGeneration.from_pretrained(args.model_name)

    def tokenize(batch):
        enc = tokenizer(batch["src"], max_length=args.max_len, truncation=True)
        tgt = tokenizer(batch["tgt"], max_length=args.max_len, truncation=True, return_offsets_mapping=True)
        enc["labels"] = tgt["input_ids"]
        enc["label_weights"] = []
        for text, offsets, ww, eos_w in zip(batch["tgt"], tgt["offset_mapping"], batch["word_weights"],
                                            batch["eos_weight"]):
            # Character index -> word index (-1 for the separating spaces).
            char2word = []
            for k, word in enumerate(text.split(" ")):
                char2word += [k] * len(word) + [-1]
            # Each subword takes the weight of the word containing its last character
            # (the first character may be the BPE leading space); <s>/</s> have empty spans.
            lw = [ww[char2word[e - 1]] if e > s and char2word[e - 1] >= 0 else 1.0 for s, e in offsets]
            lw[-1] = max(lw[-1], eos_w)  # deletion at the end of the sentence -> </s>
            enc["label_weights"].append(lw)
        return enc

    dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=["src", "tgt", "word_weights", "eos_weight"],
        desc="Tokenizing",
    )

    # Show a few weighted examples as a sanity check.
    shown = 0
    for ex in dataset:
        if any(w != 1.0 for w in ex["label_weights"]):
            toks = tokenizer.convert_ids_to_tokens(ex["labels"])
            print(" ".join(f"{t}" if w == 1.0 else f"[{t}|{w:.1f}]" for t, w in zip(toks, ex["label_weights"])))
            shown += 1
            if shown == 3:
                break

    # Pads labels with -100 (ignored by the loss) and label_weights with 0,
    # and builds decoder_input_ids by shifting labels right (teacher forcing).
    collator = WeightedSeq2SeqCollator(tokenizer, model=model, label_pad_token_id=-100)

    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    fp16 = use_cuda and not bf16

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        label_smoothing_factor=0.0,
        group_by_length=True,
        bf16=bf16,
        fp16=fp16,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_workers,
        report_to="none",
        seed=args.seed,
        # Otherwise Trainer drops label_weights, which model.forward doesn't accept.
        remove_unused_columns=False,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    last_ckpt = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.is_dir() else None
    if last_ckpt:
        print(f"Resuming from {last_ckpt}")
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"Saved fine-tuned model to {model_dir}")

    with open(out_dir / "train_log.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    config = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dataset_sizes": sizes,
        "target_mismatches": mismatches,
        "num_train_examples": len(dataset),
        "category_weights": {c: {"count": n, "weight": cat_weights[c]} for c, n in cat_counts.most_common()},
        "train_metrics": train_result.metrics,
        "precision": "bf16" if bf16 else "fp16" if fp16 else "fp32",
        "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    with open(out_dir / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()

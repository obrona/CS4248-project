"""Standard fine-tuning of facebook/bart-base for grammatical error correction.

Trains for one epoch on all BEA-2019 training data (FCE train+dev, Lang-8,
NUCLE, W&I A/B/C) with the standard token-level cross-entropy loss.
No evaluation is run here -- see standard-eval.py.

Usage:
    python standard-finetune.py                       # full run
    python standard-finetune.py --max_train_samples 64 --max_steps 2   # smoke test
"""

import argparse
import json
import os
import platform
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"

# Cache HF downloads inside output/ so the run is self-contained.
os.environ.setdefault("HF_HOME", str(OUTPUT_DIR / "hf_cache"))

import torch  # noqa: E402
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

# Parallel (source, target) files, one tokenized sentence per line.
TRAIN_FILES = [
    ("fce-src-tgt", "fce.train.gold.bea19"),
    ("fce-src-tgt", "fce.dev.gold.bea19"),
    ("lang8-src-tgt", "lang8.train.auto.bea19"),
    ("nucle-src-tgt", "nucle.train.gold.bea19"),
    ("w&-src-tgt", "A.train.gold.bea19"),
    ("w&-src-tgt", "B.train.gold.bea19"),
    ("w&-src-tgt", "C.train.gold.bea19"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", default=MODEL_NAME)
    p.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
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
    return p.parse_args()


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def load_training_data():
    sources, targets, sizes = [], [], {}
    for subdir, stem in TRAIN_FILES:
        src = read_lines(REPO_ROOT / subdir / f"{stem}.src")
        tgt = read_lines(REPO_ROOT / subdir / f"{stem}.tgt")
        assert len(src) == len(tgt), f"{stem}: {len(src)} src vs {len(tgt)} tgt lines"
        sources.extend(src)
        targets.extend(tgt)
        sizes[stem] = len(src)
        print(f"Loaded {len(src):>9,} pairs from {subdir}/{stem}")
    print(f"Total: {len(sources):,} pairs")
    return sources, targets, sizes


def main():
    args = parse_args()
    out_dir = args.output_dir
    model_dir = out_dir / "bart-base-gec"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    transformers.set_seed(args.seed)

    sources, targets, sizes = load_training_data()
    dataset = Dataset.from_dict({"src": sources, "tgt": targets})
    if args.max_train_samples is not None:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BartForConditionalGeneration.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(
            batch["src"],
            text_target=batch["tgt"],
            max_length=args.max_len,
            truncation=True,
        )

    dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=["src", "tgt"],
        desc="Tokenizing",
    )

    # Pads labels with -100 so padding is ignored by the cross-entropy loss,
    # and builds decoder_input_ids by shifting labels right (teacher forcing).
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

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
    )

    trainer = Seq2SeqTrainer(
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
        "num_train_examples": len(dataset),
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

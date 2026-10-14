"""Train BART-base for grammatical error correction, evaluate, then save."""

import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from importlib.metadata import version
from itertools import zip_longest
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import errant
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

# Training settings. No checkpoints or evaluation are performed during training.
MODEL_NAME = "facebook/bart-base"
EPOCHS = 3
LEARNING_RATE = 3e-5
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 16
MAX_LENGTH = 512
SEED = 42

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs/bart-base-gec"
TRAIN_SOURCES = [
    ROOT / "lang8-src-tgt/lang8.train.auto.bea19.src",
    ROOT / "fce-src-tgt/fce.train.gold.bea19.src",
    ROOT / "nucle-src-tgt/nucle.train.gold.bea19.src",
    ROOT / "w&-src-tgt/A.train.gold.bea19.src",
    ROOT / "w&-src-tgt/B.train.gold.bea19.src",
    ROOT / "w&-src-tgt/C.train.gold.bea19.src",
]
DEV_SOURCE = ROOT / "w&-src-tgt/ABCN.dev.gold.bea19.src"
GOLD_M2 = ROOT / "data/raw/ABCN.dev.gold.bea19.m2"


def load_pairs(source_paths: list[str]) -> Iterator[dict[str, str]]:
    """Read aligned pairs without discarding unchanged or empty target sentences."""
    for source_path in source_paths:
        source_file = Path(source_path)
        target_file = source_file.with_suffix(".tgt")
        with source_file.open(encoding="utf-8") as sources, target_file.open(
            encoding="utf-8"
        ) as targets:
            count = 0
            for count, (source, target) in enumerate(
                zip_longest(sources, targets), start=1
            ):
                if source is None or target is None:
                    raise ValueError(f"Line counts differ: {source_file}, {target_file}")
                if not source.strip():
                    raise ValueError(f"Empty source: {source_file}, line {count}")
                yield {"src": source.rstrip("\r\n"), "tgt": target.rstrip("\r\n")}
            if count == 0:
                raise ValueError(f"Empty data file: {source_file}")
            print(f"Loaded {count:,} pairs from {source_file.name}", flush=True)


def prepare_gold(dev_sources: list[str]) -> None:
    """Extract the development reference if necessary and check sentence alignment."""
    if not GOLD_M2.exists():
        GOLD_M2.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(ROOT / "data/raw.zip") as archive:
            with archive.open("raw/ABCN.dev.gold.bea19.m2") as source:
                with GOLD_M2.open("wb") as target:
                    shutil.copyfileobj(source, target)
    with GOLD_M2.open(encoding="utf-8") as reference:
        gold_sources = [
            line[2:].rstrip("\r\n") for line in reference if line.startswith("S ")
        ]
    if dev_sources != gold_sources:
        raise ValueError("Development sources do not align with the gold M2 sentences")


def write_predictions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sources: list[str],
    prediction_file: Path,
    precision: torch.dtype,
) -> None:
    """Generate one corrected sentence per development source, in original order."""
    model.eval()
    with prediction_file.open("w", encoding="utf-8") as output:
        with torch.inference_mode(), torch.autocast("cuda", dtype=precision):
            for start in range(0, len(sources), BATCH_SIZE):
                inputs = tokenizer(
                    sources[start : start + BATCH_SIZE],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                ).to(model.device)
                generated = model.generate(
                    **inputs,
                    num_beams=4,
                    do_sample=False,
                    max_new_tokens=MAX_LENGTH,
                )
                predictions = tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for prediction in predictions:
                    # ERRANT expects whitespace-tokenised sentences, one per line.
                    output.write(" ".join(prediction.split()) + "\n")
                if start % 100 == 0:
                    print(f"Generated {min(start + BATCH_SIZE, len(sources)):,}/{len(sources):,}")


def score_predictions(prediction_file: Path) -> str:
    """Score generated edits against the supplied gold annotations with ERRANT."""
    hypothesis = OUTPUT_DIR / "dev_predictions.m2"
    bin_dir = Path(sys.executable).parent
    subprocess.run(
        [
            str(bin_dir / "errant_parallel"),
            "-orig", str(DEV_SOURCE),
            "-cor", str(prediction_file),
            "-out", str(hypothesis),
        ],
        check=True,
    )
    result = subprocess.run(
        [str(bin_dir / "errant_compare"), "-hyp", str(hypothesis), "-ref", str(GOLD_M2)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    (OUTPUT_DIR / "errant_scores.txt").write_text(result.stdout, encoding="utf-8")
    return result.stdout


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Run this script in the Python 3.12 cluster venv")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available. Submit with sbatch train_bart.sh")
    # Load scoring resources before investing time in training.
    errant.load("en")
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading training and development sentence pairs...")
    train_data = Dataset.from_generator(
        load_pairs,
        gen_kwargs={"source_paths": [str(path) for path in TRAIN_SOURCES]},
    )
    dev_data = Dataset.from_generator(
        load_pairs, gen_kwargs={"source_paths": [str(DEV_SOURCE)]}
    )
    dev_sources = list(dev_data["src"])
    prepare_gold(dev_sources)
    print(f"Training: {len(train_data):,} pairs; development: {len(dev_data):,} pairs")

    print(f"Downloading/loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def tokenize(batch: dict[str, list[str]]) -> BatchEncoding:
        return tokenizer(
            batch["src"],
            text_target=batch["tgt"],
            max_length=MAX_LENGTH,
            truncation=True,
        )

    train_tokens = train_data.map(tokenize, batched=True, remove_columns=train_data.column_names)
    dev_tokens = dev_data.map(tokenize, batched=True, remove_columns=dev_data.column_names)
    bf16 = torch.cuda.is_bf16_supported()
    precision = torch.bfloat16 if bf16 else torch.float16
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        weight_decay=0.01,
        warmup_ratio=0.05,
        seed=SEED,
        bf16=bf16,
        fp16=not bf16,
        eval_strategy="no",
        save_strategy="no",
        prediction_loss_only=True,
        logging_steps=100,
        report_to="none",
        push_to_hub=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokens,
        eval_dataset=dev_tokens,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, label_pad_token_id=-100
        ),
    )

    print("Fine-tuning BART-base...")
    train_result = trainer.train()
    metrics: dict[str, Any] = {
        "train": train_result.metrics,
        "errant_version": version("errant"),
        "settings": {
            "model": MODEL_NAME,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "max_length": MAX_LENGTH,
            "seed": SEED,
            "precision": "bf16" if bf16 else "fp16",
            "train_pairs": len(train_data),
            "dev_pairs": len(dev_data),
        },
    }
    try:
        print("Evaluating on ABCN.dev.gold.bea19...")
        metrics["evaluation"] = trainer.evaluate()
        print(json.dumps(metrics["evaluation"], indent=2))
        prediction_file = OUTPUT_DIR / "dev_predictions.txt"
        write_predictions(model, tokenizer, dev_sources, prediction_file, precision)
        print(score_predictions(prediction_file))
    finally:
        # Preserve the finished model even if final scoring fails.
        print(f"Saving final weights and tokenizer to {OUTPUT_DIR}...")
        trainer.save_model(str(OUTPUT_DIR))
        tokenizer.save_pretrained(OUTPUT_DIR)
        (OUTPUT_DIR / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

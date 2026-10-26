import argparse
import json
import math
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from data_loader import load_dev_data
from category_alignment import (
    load_training_data,
    category_weights,
    align_token_weights,
)
from evaluate_gec import write_hypothesis_m2, validate_gold_alignment

def args_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--max-train-examples", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulation", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--output-dir", default="runs/bart_frequency_trial")
    p.add_argument("--gold-m2", required=True)
    p.add_argument("--scorer", default="errant_compare")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")

    # Category-aware edit weighting
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--w-max", type=float, default=10.0)

    # Original M2 annotations
    p.add_argument(
        "--m2-dir",
        default="data/original_m2/raw",
    )

    return p.parse_args()

def make_collate(tok, max_length, category_weights):
    def collate(batch):
        src, tgt, word_cats, eos_cats = zip(*batch)

        x = tok(
            list(src),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        labels = tok(
            text_target=list(tgt),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )["input_ids"]

        # Construct per-token weights before padding.
        token_weights = [
            align_token_weights(
                tok,
                target,
                cats,
                eos,
                category_weights,
                max_length=max_length,
            )
            for target, cats, eos in zip(tgt, word_cats, eos_cats)
        ]

        # Padding tokens must contribute zero to the loss.
        label_weights = torch.zeros(
            labels.shape,
            dtype=torch.float32,
        )

        for i, weights in enumerate(token_weights):
            if len(weights) != int((labels[i] != tok.pad_token_id).sum()):
                raise ValueError(
                    f"Token-weight alignment mismatch at batch index {i}"
                )

            label_weights[i, :len(weights)] = torch.tensor(
                weights,
                dtype=torch.float32,
            )

        labels[labels == tok.pad_token_id] = -100

        x["labels"] = labels
        x["label_weights"] = label_weights

        return x

    return collate

def generate(model, tok, sources, device, max_length, batch_size):
    model.eval()
    predictions = []

    with torch.inference_mode():
        for i in range(0, len(sources), batch_size):
            batch = tok(
                sources[i:i + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            ids = model.generate(
                **batch,
                num_beams=4,
                do_sample=False,
                max_new_tokens=512,
                length_penalty=1.0,
            )

            predictions.extend(
                tok.batch_decode(ids, skip_special_tokens=True)
            )

            if (i // batch_size) % 100 == 0:
                print(
                    f"Generated {min(i + batch_size, len(sources))}/"
                    f"{len(sources)}",
                    flush=True,
                )

    return predictions

def score(gold_m2, system_m2, scorer):
    command = [
        scorer,
        "-hyp", str(system_m2),
        "-ref", str(gold_m2),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    output = result.stdout + "\n" + result.stderr

    if result.returncode != 0:
        raise RuntimeError(
            f"ERRANT scorer failed:\n{output}"
        )

    print(output, flush=True)

    # Find the column header in ERRANT's output.
    lines = [line.strip() for line in output.splitlines()]

    for i, line in enumerate(lines):
        columns = line.split()

        if "F0.5" not in columns:
            continue

        score_index = columns.index("F0.5")

        # The next nonempty line should contain metric values.
        for next_line in lines[i + 1:]:
            values = next_line.split()

            if not values:
                continue

            if len(values) == len(columns):
                try:
                    f05 = float(values[score_index])
                    return f05, output
                except ValueError:
                    pass

    raise RuntimeError(
        f"Could not extract F0.5 from ERRANT output:\n{output}"
    )

def compute_weighted_loss(logits, labels, label_weights):
    """
    Token-level weighted cross-entropy.

    Padding positions are ignored.
    """
    vocab_size = logits.size(-1)

    token_losses = F.cross_entropy(
        logits.float().reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(labels)

    valid = labels != -100
    weights = label_weights.float() * valid.float()

    return (token_losses * weights).sum() / weights.sum().clamp_min(1.0)

def main():
    a = args_parser()

    if a.batch_size * a.accumulation != 32:
        raise ValueError(
            "Expected effective batch size of 32 on one GPU"
        )

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")

    device = torch.device("cuda")

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "config.json").write_text(
        json.dumps(vars(a), indent=2),
        encoding="utf-8",
    )

    print("Loading annotated training data...", flush=True)

    sources_train, targets_train, word_cats, eos_cats, sizes, mismatches = (
        load_training_data(Path(a.m2_dir))
    )

    # The annotated loader returns targets as lists of words.
    # Convert them to strings, matching the baseline representation.
    targets_train = [
        " ".join(words) if isinstance(words, list) else words
        for words in targets_train
    ]

    # Keep each sentence and its grammatical-error annotations together.
    all_train = list(zip(
        sources_train,
        targets_train,
        word_cats,
        eos_cats,
    ))

    rng = random.Random(a.seed)
    rng.shuffle(all_train)

    train = (
        all_train[:a.max_train_examples]
        if a.max_train_examples > 0
        else all_train
    )

    # Calculate category frequencies from the full training corpus,
    # not just the smoke-test subset.
    weights, counts = category_weights(
        word_cats,
        eos_cats,
        alpha=a.alpha,
        beta=a.beta,
        w_max=a.w_max,
    )

    print(
        f"Training examples: {len(train):,}",
        flush=True,
    )
    print(
        f"Category weights: {weights}",
        flush=True,
    )

    (out / "category_weights.json").write_text(
        json.dumps(
            {
                "alpha": a.alpha,
                "beta": a.beta,
                "w_max": a.w_max,
                "weights": weights,
                "counts": dict(counts),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    dev = load_dev_data()
    sources = [x for x, _ in dev]

    # Verify gold M2 alignment before loading the model.
    validate_gold_alignment(sources, a.gold_m2)

    tok = AutoTokenizer.from_pretrained("facebook/bart-base")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/bart-base"
    ).to(device)

    loader = DataLoader(
        train,
        batch_size=a.batch_size,
        shuffle=True,
        collate_fn=make_collate(tok, a.max_length, weights),
    )

    steps_per_epoch = math.ceil(
        len(loader) / a.accumulation
    )

    total_steps = steps_per_epoch * a.epochs

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=3e-5,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01,
    )

    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=math.ceil(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best = -1
    stale = 0
    history = []
    start_epoch = 1

    checkpoint_path = out / "checkpoint_latest.pt"

    if a.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        best = checkpoint["best_f05"]
        stale = checkpoint["stale"]
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1

        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

        print(
            f"Resuming from epoch {start_epoch}",
            flush=True,
        )

    for epoch in range(start_epoch, a.epochs + 1):
        model.train()
        optim.zero_grad(set_to_none=True)

        running = 0.0

        for idx, batch in enumerate(loader):
            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            label_weights = batch.pop("label_weights")

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)

                loss = compute_weighted_loss(
                    outputs.logits,
                    batch["labels"],
                    label_weights,
                )

            remaining = len(loader) % a.accumulation

            if remaining == 0:
                remaining = a.accumulation

            group_size = (
                remaining
                if idx >= len(loader) - remaining
                else a.accumulation
            )

            (loss / group_size).backward()
            
            running += loss.item()

            if (
                (idx + 1) % a.accumulation == 0
                or (idx + 1) == len(loader)
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )

                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)

            if (idx + 1) % 500 == 0:
                print(
                    f"Epoch {epoch} microbatch "
                    f"{idx + 1}/{len(loader)} "
                    f"loss {loss.item():.4f}",
                    flush=True,
                )

        mean_loss = running / len(loader)

        print(
            f"Epoch {epoch} mean training loss "
            f"{mean_loss:.4f}",
            flush=True,
        )

        preds = generate(
            model,
            tok,
            sources,
            device,
            a.max_length,
            a.eval_batch_size,
        )

        ep = out / f"epoch_{epoch}"
        ep.mkdir(exist_ok=True)

        pred_path = ep / "predictions.txt"

        pred_path.write_text(
            "\n".join(preds) + "\n",
            encoding="utf-8",
        )

        system_m2 = write_hypothesis_m2(
            sources,
            preds,
            ep / "hypothesis.m2",
        )

        f05, scorer_output = score(
            a.gold_m2,
            system_m2,
            a.scorer,
        )

        (ep / "errant_output.txt").write_text(
            scorer_output,
            encoding="utf-8",
        )

        history.append({
            "epoch": epoch,
            "training_loss": mean_loss,
            "dev_f05": f05,
        })

        (out / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        if f05 > best:
            best, stale = f05, 0

            best_dir = out / "best_model"

            model.save_pretrained(best_dir)
            tok.save_pretrained(best_dir)

            (out / "best_epoch.txt").write_text(
                f"{epoch}\n{best}\n",
                encoding="utf-8",
            )

            print(
                f"New best dev F0.5: {best}",
                flush=True,
            )

        else:
            stale += 1

        # Save checkpoint after every completed epoch
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_f05": best,
            "stale": stale,
            "history": history,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        }

        checkpoint_path = out / "checkpoint_latest.pt"
        temporary_path = out / "checkpoint_latest.tmp"

        torch.save(checkpoint, temporary_path)
        temporary_path.replace(checkpoint_path)

        print(
            f"Saved checkpoint after epoch {epoch}",
            flush=True,
        )

        # Check early stopping only AFTER saving
        if stale >= 2:
            print(
                "Early stopping: 2 epochs without improvement",
                flush=True,
            )
            break

    print(f"Finished. Best dev F0.5 = {best}")


if __name__ == "__main__":
    main()


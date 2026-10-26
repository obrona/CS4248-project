import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from train_frequency import make_collate, compute_weighted_loss

tok = AutoTokenizer.from_pretrained("facebook/bart-base")

weights = {
    "VERB:SVA": 4.0,
    "DET": 3.0,
    "OTHER": 2.0,
}

examples = [
    (
        "She go to school.",
        "She goes to school.",
        [None, ["VERB:SVA"], None, None],
        None,
    ),
    (
        "I have apple.",
        "I have an apple.",
        [None, None, ["DET"], None],
        None,
    ),
    (
        "Unnecessary words.",
        "",
        [None],
        ["OTHER"],
    ),
]

collate = make_collate(tok, max_length=64, category_weights=weights)
batch = collate(examples)

print("Input shape:", batch["input_ids"].shape)
print("Label shape:", batch["labels"].shape)
print("Weight shape:", batch["label_weights"].shape)

assert batch["labels"].shape == batch["label_weights"].shape

for i, example in enumerate(examples):
    print(f"\nExample {i + 1}: {example[1]!r}")

    for token_id, weight, label in zip(
        batch["labels"][i],
        batch["label_weights"][i],
        batch["labels"][i],
    ):
        if label.item() == -100:
            assert weight.item() == 0
            continue

        token = tok.convert_ids_to_tokens(int(token_id))
        print(f"{token:12s} weight={weight.item():.1f}")

# CPU-only forward pass; no GPU required.
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
model.eval()

label_weights = batch.pop("label_weights")

with torch.no_grad():
    outputs = model(**batch)

    loss = compute_weighted_loss(
        outputs.logits,
        batch["labels"],
        label_weights,
    )

assert torch.isfinite(loss), "Loss is not finite!"
print(f"\nPASS: End-to-end weighted loss = {loss.item():.4f}")

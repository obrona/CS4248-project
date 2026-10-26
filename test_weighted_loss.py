import torch
import torch.nn.functional as F

from train_frequency import compute_weighted_loss

torch.manual_seed(42)

logits = torch.randn(2, 4, 100)
labels = torch.tensor([
    [5, 10, 15, -100],
    [8, 12, -100, -100],
])

weights = (labels != -100).float()

standard_loss = F.cross_entropy(
    logits.reshape(-1, 100),
    labels.reshape(-1),
    ignore_index=-100,
)

weighted_loss = compute_weighted_loss(
    logits,
    labels,
    weights,
)

print(f"Standard loss: {standard_loss.item():.8f}")
print(f"Weighted loss: {weighted_loss.item():.8f}")

assert torch.allclose(
    standard_loss,
    weighted_loss,
    atol=1e-6,
), "Uniform weights do not reproduce standard cross-entropy!"

print("PASS: Uniform token weights reproduce standard cross-entropy.")

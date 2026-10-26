from pathlib import Path

from category_alignment import load_training_data, category_weights

m2_dir = Path("data/original_m2/raw")

print("Loading M2 annotations...", flush=True)

sources, targets, word_cats, eos_cats, sizes, mismatches = (
    load_training_data(m2_dir)
)

print(f"Training examples: {len(sources):,}")
print(f"Target mismatches: {sum(mismatches.values()) if isinstance(mismatches, dict) else mismatches}")

for beta in [0.0, 0.5]:
    weights, counts = category_weights(
        word_cats,
        eos_cats,
        alpha=3.0,
        beta=beta,
        w_max=10.0,
    )

    print(f"\nBeta = {beta}")
    print(f"Categories: {len(weights)}")

    for category in ["OTHER", "NOUN", "DET", "VERB:INFL"]:
        print(
            f"{category:12s} "
            f"count={counts[category]:>8,} "
            f"weight={weights[category]:.4f}"
        )

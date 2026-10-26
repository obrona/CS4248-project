from pathlib import Path

TRAIN_FILES = [
    "fce-src-tgt/fce.train.gold.bea19",
    "fce-src-tgt/fce.dev.gold.bea19",
    "lang8-src-tgt/lang8.train.auto.bea19",
    "nucle-src-tgt/nucle.train.gold.bea19",
    "w&-src-tgt/A.train.gold.bea19",
    "w&-src-tgt/B.train.gold.bea19",
    "w&-src-tgt/C.train.gold.bea19",
]

DEV_FILE = "w&-src-tgt/ABCN.dev.gold.bea19"

def load_pairs(prefix):
    src_path = Path(prefix + ".src")
    tgt_path = Path(prefix + ".tgt")

    with src_path.open(encoding="utf-8") as f:
        sources = [line.rstrip("\n") for line in f]

    with tgt_path.open(encoding="utf-8") as f:
        targets = [line.rstrip("\n") for line in f]

    assert len(sources) == len(targets), (
        f"Mismatch in {prefix}: "
        f"{len(sources)} sources vs {len(targets)} targets"
    )

    for i, (src, tgt) in enumerate(zip(sources, targets)):
        if not src.strip():
            raise ValueError(
                f"Empty source in {prefix}, line {i + 1}"
            )

    return list(zip(sources, targets))

def load_training_data():
    all_pairs = []

    for prefix in TRAIN_FILES:
        pairs = load_pairs(prefix)
        print(f"{prefix}: {len(pairs):,} pairs")
        all_pairs.extend(pairs)

    print(f"Total training pairs: {len(all_pairs):,}")
    return all_pairs


def load_dev_data():
    pairs = load_pairs(DEV_FILE)
    print(f"Development pairs: {len(pairs):,}")
    return pairs


if __name__ == "__main__":
    train = load_training_data()
    dev = load_dev_data()

    print("\nExample training pair:")
    print("Source:", train[0][0])
    print("Target:", train[0][1])


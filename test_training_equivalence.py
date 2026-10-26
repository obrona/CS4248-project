from pathlib import Path

from data_loader import load_training_data as load_baseline
from category_alignment import load_training_data as load_annotated

print("Loading baseline data...", flush=True)
baseline = load_baseline()

print("Loading annotated data...", flush=True)
sources, targets, word_cats, eos_cats, sizes, mismatches = (
    load_annotated(Path("data/original_m2/raw"))
)

assert len(baseline) == len(sources) == len(targets)
assert len(word_cats) == len(eos_cats) == len(baseline)

different = []

for i, ((baseline_src, baseline_tgt), src, tgt) in enumerate(
    zip(baseline, sources, targets)
):
    if baseline_src != src or baseline_tgt != tgt:
        different.append(i)

print(f"Total examples: {len(baseline):,}")
print(f"Pair mismatches: {len(different)}")

if different:
    print("First mismatching indices:", different[:10])
    raise AssertionError("Annotated data differs from baseline data!")

print("PASS: All training pairs match the baseline in the same order.")

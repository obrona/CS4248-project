#!/usr/bin/env python3

"""Paired randomisation test for ERRANT category F0.5."""

import argparse
import csv
import random
from pathlib import Path

from bootstrap_categories import (
    load_errant,
    read_m2,
    make_args,
    prepare_model,
    replay,
    score_sample,
    get_differences,
    read_expected,
    validate,
)


def run_randomisation(
    errant,
    prepared_u,
    prepared_f,
    original_u,
    original_f,
    iterations=1000,
    seed=42,
):
    rng = random.Random(seed)

    n = len(prepared_u)

    if n != len(prepared_f):
        raise ValueError("Models have different sentence counts")

    category_names = sorted(set(original_u) | set(original_f))

    # F0.5(frequency) - F0.5(uniform)
    observed = get_differences(
        original_u, original_f, category_names
    )

    extreme_counts = {cat: 0 for cat in category_names}

    for iteration in range(iterations):

        # For each sentence, independently swap the model labels
        # with probability 0.5.
        random_u = []
        random_f = []

        for i in range(n):
            if rng.random() < 0.5:
                random_u.append(prepared_f[i])
                random_f.append(prepared_u[i])
            else:
                random_u.append(prepared_u[i])
                random_f.append(prepared_f[i])

        # Recompute ERRANT's greedy reference selection.
        # score_sample() returns category counts only.
        u_counts = score_sample(
            errant, random_u, range(n)
        )
        f_counts = score_sample(
            errant, random_f, range(n)
        )

        differences = get_differences(
            u_counts, f_counts, category_names
        )

        for cat in category_names:
            if observed[cat] is None:
                continue

            delta = differences[cat]

            # Defined convention for a category absent
            # from both randomised model groups.
            if delta is None:
                delta = 0.0

            # Two-sided randomisation test
            if abs(delta) >= abs(observed[cat]) - 1e-12:
                extreme_counts[cat] += 1

        if (iteration + 1) % 100 == 0:
            print(
                f"Randomisation {iteration + 1}/{iterations}",
                flush=True,
            )

    results = []

    for cat in category_names:
        if observed[cat] is None:
            continue

        p_value = (
            extreme_counts[cat] + 1
        ) / (iterations + 1)

        results.append({
            "category": cat,
            "delta_f05": observed[cat],
            "p_randomisation": p_value,
            "extreme_count": extreme_counts[cat],
            "iterations": iterations,
        })

    return results

def apply_holm(results):
    """Add Holm-adjusted p-values across all tested categories."""

    ordered = sorted(
        results,
        key=lambda row: row["p_randomisation"]
    )

    m = len(ordered)
    running_max = 0.0

    for i, row in enumerate(ordered):
        adjusted = (m - i) * row["p_randomisation"]
        running_max = max(running_max, adjusted)

        row["p_holm"] = min(1.0, running_max)

    return results

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gold", required=True)
    parser.add_argument("--uniform", required=True)
    parser.add_argument("--frequency", required=True)
    parser.add_argument("--compiled", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be positive")

    errant = load_errant()
    scoring_args = make_args()

    gold = read_m2(args.gold)

    prepared_models = {}
    original_categories = {}

    for model in ("uniform", "frequency"):

        hypotheses = read_m2(getattr(args, model))

        if len(hypotheses) != len(gold):
            raise ValueError(
                f"{model}: {len(hypotheses)} hypotheses "
                f"vs {len(gold)} references"
            )

        prepared = prepare_model(
            errant, hypotheses, gold, scoring_args
        )

        prepared_models[model] = prepared

        overall, categories, _ = replay(
            errant, prepared
        )

        original_categories[model] = categories

        # Confirm exact agreement with existing ERRANT results.
        expected = read_expected(args.compiled, model)
        validate(categories, expected, model)

        print(
            f"{model}: {len(hypotheses)} sentences; "
            f"TP={overall['tp']}, "
            f"FP={overall['fp']}, "
            f"FN={overall['fn']}"
        )

    print("\nStarting paired randomisation test...")

    results = run_randomisation(
        errant,
        prepared_models["uniform"],
        prepared_models["frequency"],
        original_categories["uniform"],
        original_categories["frequency"],
        iterations=args.iterations,
        seed=args.seed,
    )
    results = apply_holm(results)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "delta_f05",
                "p_randomisation",
                "p_holm",
                "extreme_count",
                "iterations",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} categories to {output_path}")

    print("\nSmallest randomisation p-values:")
    for row in sorted(
        results, key=lambda r: r["p_randomisation"]
    )[:15]:
        print(
            f'{row["category"]:<18} '
            f'delta={row["delta_f05"]:+.4f} '
            f'p={row["p_randomisation"]:.4f} '
            f'holm={row["p_holm"]:.4f}'
        )


if __name__ == "__main__":
    main()

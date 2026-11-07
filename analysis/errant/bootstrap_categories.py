#!/usr/bin/env python3

"""Prepare and validate sentence-level ERRANT category scoring."""

import random
import argparse
import csv
import importlib
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace


def load_errant():
    return importlib.import_module("errant.commands.compare_m2")


def read_m2(path):
    return Path(path).read_text(encoding="utf-8").strip().split("\n\n")


def make_args():
    # Match errant_compare -cs -cat 3.
    return SimpleNamespace(
        dt=False,
        ds=False,
        cs=True,
        cse=False,
        single=False,
        multi=False,
        filt=[],
        beta=0.5,
        verbose=False,
        cat=3,
    )


def prepare_model(errant, hyp_blocks, ref_blocks, args):
    """Precompute all hypothesis/reference scoring combinations."""
    prepared = []

    for sent_id, (hyp, ref) in enumerate(zip(hyp_blocks, ref_blocks)):
        hyp_dict = errant.process_edits(errant.simplify_edits(hyp), args)
        ref_dict = errant.process_edits(errant.simplify_edits(ref), args)

        combinations = []

        for hyp_id in hyp_dict:
            for ref_id in ref_dict:
                tp, fp, fn, cats = errant.compareEdits(
                    hyp_dict[hyp_id], ref_dict[ref_id]
                )
                combinations.append({
                    "hyp_id": hyp_id,
                    "ref_id": ref_id,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "cats": cats,
                })

        prepared.append(combinations)

    return prepared


def select_best(errant, combinations, cumulative):
    """Reproduce ERRANT's greedy cumulative-F0.5 selection."""
    chosen = None
    best_f = -1

    for candidate in combinations:
        tp = candidate["tp"]
        fp = candidate["fp"]
        fn = candidate["fn"]

        _, _, f = errant.computeFScore(
            cumulative["tp"] + tp,
            cumulative["fp"] + fp,
            cumulative["fn"] + fn,
            0.5,
        )

        if (
            chosen is None
            or f > best_f
            or (
                f == best_f
                and (
                    tp > chosen["tp"]
                    or (
                        tp == chosen["tp"]
                        and (
                            fp < chosen["fp"]
                            or (
                                fp == chosen["fp"]
                                and fn < chosen["fn"]
                            )
                        )
                    )
                )
            )
        ):
            chosen = candidate
            best_f = f

    return chosen


def replay(errant, prepared):
    """Score sentences in original order and record category counts."""
    cumulative = Counter({"tp": 0, "fp": 0, "fn": 0})
    category_totals = defaultdict(lambda: [0, 0, 0])
    sentence_rows = []

    for sent_id, combinations in enumerate(prepared):
        selected = select_best(errant, combinations, cumulative)

        for key in ("tp", "fp", "fn"):
            cumulative[key] += selected[key]

        for category, counts in selected["cats"].items():
            for i in range(3):
                category_totals[category][i] += counts[i]

            sentence_rows.append({
                "sentence_id": sent_id,
                "category": category,
                "tp": counts[0],
                "fp": counts[1],
                "fn": counts[2],
                "selected_ref": selected["ref_id"],
            })

    return cumulative, category_totals, sentence_rows


def read_expected(path, model):
    """Read expected category counts from compiled wide CSV."""
    expected = {}

    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row[f"{model}__present"] == "1":
                expected[row["category"]] = [
                    int(row[f"{model}__{key}"])
                    for key in ("tp", "fp", "fn")
                ]

    return expected


def validate(actual, expected, model):
    categories = set(actual) | set(expected)
    mismatches = []

    for category in sorted(categories):
        observed = actual.get(category, [0, 0, 0])
        target = expected.get(category, [0, 0, 0])

        if list(observed) != list(target):
            mismatches.append((category, observed, target))

    if mismatches:
        print(f"\n{model}: {len(mismatches)} category mismatches")
        for category, observed, target in mismatches[:15]:
            print(f"  {category}: actual={observed}, expected={target}")
        raise ValueError(f"{model}: validation failed")

    print(f"{model}: category counts match compiled CSV exactly")

def category_f05(tp, fp, fn):
    """F0.5 from aggregate counts; undefined if no edits exist."""
    if tp == 0 and fp == 0 and fn == 0:
        return None

    denominator = 1.25 * tp + 0.25 * fn + fp
    return 1.25 * tp / denominator if denominator else 0.0


def score_sample(errant, prepared, indices):
    """Replay ERRANT reference selection in sampled order."""
    cumulative = Counter({"tp": 0, "fp": 0, "fn": 0})
    categories = defaultdict(lambda: [0, 0, 0])

    for sent_id in indices:
        selected = select_best(errant, prepared[sent_id], cumulative)

        for key in ("tp", "fp", "fn"):
            cumulative[key] += selected[key]

        for cat, counts in selected["cats"].items():
            for j in range(3):
                categories[cat][j] += counts[j]

    return categories


def get_differences(uniform, frequency, category_names):
    differences = {}

    for cat in category_names:
        u_counts = uniform.get(cat, (0, 0, 0))
        f_counts = frequency.get(cat, (0, 0, 0))

        u_empty = sum(u_counts) == 0
        f_empty = sum(f_counts) == 0

        if u_empty and f_empty:
            differences[cat] = None
            continue

        u = 0.0 if u_empty else category_f05(*u_counts)
        f = 0.0 if f_empty else category_f05(*f_counts)

        differences[cat] = f - u

    return differences


def percentile(sorted_values, q):
    """Linearly interpolated percentile."""
    position = (len(sorted_values) - 1) * q
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low

    return (
        sorted_values[low] * (1 - weight)
        + sorted_values[high] * weight
    )


def run_bootstrap(errant, prepared_u, prepared_f,
                  original_u, original_f,
                  iterations=1000, seed=42):

    rng = random.Random(seed)
    n = len(prepared_u)

    if n != len(prepared_f):
        raise ValueError("Models have different sentence counts")

    category_names = sorted(set(original_u) | set(original_f))

    observed = get_differences(
        original_u, original_f, category_names
    )

    samples = {cat: [] for cat in category_names}

    for iteration in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]

        # Crucially, both models receive the same indices.
        u_counts = score_sample(errant, prepared_u, indices)
        f_counts = score_sample(errant, prepared_f, indices)

        differences = get_differences(
            u_counts, f_counts, category_names
        )

        for cat, difference in differences.items():
            if difference is not None:
                samples[cat].append(difference)

        if (iteration + 1) % 100 == 0:
            print(
                f"Bootstrap {iteration + 1}/{iterations}",
                flush=True
            )

    results = []

    for cat in category_names:
        values = sorted(samples[cat])
        delta = observed[cat]

        if not values or delta is None:
            continue

        lower = percentile(values, 0.025)
        upper = percentile(values, 0.975)

        # Descriptive bootstrap tail proportions.
        # These are NOT formal hypothesis-test p-values.
        p_freq = sum(x <= 0 for x in values) / len(values)
        p_uniform = sum(x >= 0 for x in values) / len(values)
        p_two_sided = min(1.0, 2 * min(p_freq, p_uniform))

        results.append({
            "category": cat,
            "delta_f05": delta,
            "ci_lower": lower,
            "ci_upper": upper,
            "bootstrap_valid": len(values),
            "bootstrap_total": iterations,
            "tail_freq_better": p_freq,
            "tail_uniform_better": p_uniform,
            "tail_two_sided": p_two_sided,
        })

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
    outputs = []
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

        overall, categories, rows = replay(errant, prepared)
        original_categories[model] = categories

        expected = read_expected(args.compiled, model)
        validate(categories, expected, model)

        print(
            f"{model}: {len(hypotheses)} sentences; "
            f"overall TP={overall['tp']}, "
            f"FP={overall['fp']}, FN={overall['fn']}"
        )

        for row in rows:
            outputs.append({"model": model, **row})

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model", "sentence_id", "category",
                "tp", "fp", "fn", "selected_ref"
            ],
        )
        writer.writeheader()
        writer.writerows(outputs)

    print(f"\nSaved {len(outputs)} rows to {output_path}")

    print("\nStarting paired bootstrap...")

    results = run_bootstrap(
        errant,
        prepared_models["uniform"],
        prepared_models["frequency"],
        original_categories["uniform"],
        original_categories["frequency"],
        iterations=args.iterations,
        seed=args.seed,
    )

    bootstrap_path = Path(args.out).with_name(
        "bootstrap_categories.csv"
    )

    with bootstrap_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "delta_f05",
                "ci_lower",
                "ci_upper",
                "bootstrap_valid",
                "bootstrap_total",
                "tail_freq_better",
                "tail_uniform_better",
                "tail_two_sided",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved bootstrap results to {bootstrap_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyse GEC disagreements using validated ERRANT category scores."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from bootstrap_categories import (
    load_errant,
    read_m2,
    make_args,
    prepare_model,
    replay,
    select_best,
    read_expected,
    validate,
)


def read_edits(block, annotator=None):
    """Extract M2 edits, optionally for one gold annotator."""
    edits = []

    for line in block.splitlines()[1:]:
        if not line.startswith("A "):
            continue

        parts = line[2:].split("|||")
        if len(parts) < 3:
            continue

        if annotator is not None:
            edit_annotator = (
                int(parts[5]) if len(parts) > 5 else 0
            )
            if edit_annotator != int(annotator):
                continue

        if parts[1] == "noop":
            continue

        edits.append({
            "span": parts[0],
            "category": parts[1],
            "correction": parts[2],
        })

    return edits


def reconstruct(block):
    """Apply predicted M2 edits to obtain the corrected sentence."""
    tokens = block.splitlines()[0][2:].split()
    edits = read_edits(block)

    parsed = []
    for edit in edits:
        start, end = map(int, edit["span"].split())
        parsed.append((start, end, edit["correction"]))

    for start, end, correction in sorted(
        parsed, key=lambda x: (x[0], x[1]), reverse=True
    ):
        replacement = (
            [] if correction in ("", "-NONE-")
            else correction.split()
        )
        tokens[start:end] = replacement

    return " ".join(tokens)


def selected_combinations(errant, prepared):
    """Replay ERRANT selection, retaining one choice per sentence."""
    cumulative = Counter({"tp": 0, "fp": 0, "fn": 0})
    selected = []

    for combinations in prepared:
        choice = select_best(errant, combinations, cumulative)
        selected.append(choice)

        for key in ("tp", "fp", "fn"):
            cumulative[key] += choice[key]

    return selected


def category_counts(choice, category):
    return list(choice["cats"].get(category, (0, 0, 0)))


def classify(u, f):
    u_tp, u_fp, u_fn = u
    f_tp, f_fp, f_fn = f

    if u == f:
        return "Same category counts"
    if f_tp > u_tp and f_fp <= u_fp:
        return "Frequency advantage"
    if u_tp > f_tp and u_fp <= f_fp:
        return "Uniform advantage"
    if f_fp > u_fp and f_tp <= u_tp:
        return "Frequency more FP"
    if u_fp > f_fp and u_tp <= f_tp:
        return "Uniform more FP"
    return "Mixed trade-off"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--uniform", required=True)
    parser.add_argument("--frequency", required=True)
    parser.add_argument("--compiled", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["R:SPELL", "R:NOUN:NUM"],
    )
    args = parser.parse_args()

    errant = load_errant()
    scoring_args = make_args()
    gold = read_m2(args.gold)

    hypotheses = {}
    selections = {}

    for model in ("uniform", "frequency"):
        blocks = read_m2(getattr(args, model))

        if len(blocks) != len(gold):
            raise ValueError(f"{model}: sentence count mismatch")

        prepared = prepare_model(
            errant, blocks, gold, scoring_args
        )

        overall, categories, _ = replay(errant, prepared)
        validate(
            categories,
            read_expected(args.compiled, model),
            model,
        )

        hypotheses[model] = blocks
        selections[model] = selected_combinations(
            errant, prepared
        )

        # Confirm the selected combinations reproduce replay().
        totals = Counter()
        for choice in selections[model]:
            for key in ("tp", "fp", "fn"):
                totals[key] += choice[key]

        if totals != overall:
            raise ValueError(
                f"{model}: selected combinations do not match replay"
            )

        print(
            f"{model}: TP={overall['tp']} "
            f"FP={overall['fp']} FN={overall['fn']}"
        )

    rows = []
    focus = set(args.categories)

    for i, gold_block in enumerate(gold):
        u_block = hypotheses["uniform"][i]
        f_block = hypotheses["frequency"][i]

        u_prediction = reconstruct(u_block)
        f_prediction = reconstruct(f_block)

        if u_prediction == f_prediction:
            continue

        u_choice = selections["uniform"][i]
        f_choice = selections["frequency"][i]

        relevant = (
            set(u_choice["cats"])
            | set(f_choice["cats"])
        ) & focus

        if not relevant:
            continue

        source = gold_block.splitlines()[0][2:]

        for category in sorted(relevant):
            u = category_counts(u_choice, category)
            f = category_counts(f_choice, category)

            rows.append({
                "sentence_id": i,
                "category": category,
                "outcome": classify(u, f),
                "source": source,
                "uniform_prediction": u_prediction,
                "frequency_prediction": f_prediction,
                "uniform_tp": u[0],
                "uniform_fp": u[1],
                "uniform_fn": u[2],
                "frequency_tp": f[0],
                "frequency_fp": f[1],
                "frequency_fn": f[2],
                "uniform_selected_ref": u_choice["ref_id"],
                "frequency_selected_ref": f_choice["ref_id"],
                "same_reference": (
                    u_choice["ref_id"] == f_choice["ref_id"]
                ),
                "uniform_edits": json.dumps(
                    read_edits(u_block), ensure_ascii=False
                ),
                "frequency_edits": json.dumps(
                    read_edits(f_block), ensure_ascii=False
                ),
                "uniform_gold_edits": json.dumps(
                    read_edits(
                        gold_block,
                        annotator=u_choice["ref_id"],
                    ),
                    ensure_ascii=False,
                ),
                "frequency_gold_edits": json.dumps(
                    read_edits(
                        gold_block,
                        annotator=f_choice["ref_id"],
                    ),
                    ensure_ascii=False,
                ),
            })

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        with output.open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0])
            )
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {output}")

    summary = Counter(
        (r["category"], r["outcome"]) for r in rows
    )

    print("\nDisagreement summary:")
    for (category, outcome), count in sorted(summary.items()):
        print(f"{category:<15} {outcome:<25} {count}")

    different_refs = sum(
        not r["same_reference"] for r in rows
    )
    print(
        f"\nRows with different selected references: "
        f"{different_refs}/{len(rows)}"
    )


if __name__ == "__main__":
    main()

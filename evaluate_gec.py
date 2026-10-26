
"""Validate BEA development alignment and convert GEC predictions to M2."""

import argparse
from pathlib import Path

import errant


def validate_gold_alignment(sources, gold_m2):
    """Verify each development source matches the gold M2 source in order."""
    gold_m2 = Path(gold_m2)

    if not gold_m2.is_file():
        raise FileNotFoundError(f"Gold M2 not found: {gold_m2}")

    blocks = [
        block
        for block in gold_m2.read_text(encoding="utf-8").split("\n\n")
        if block.strip()
    ]

    if len(blocks) != len(sources):
        raise ValueError(
            f"Gold/source count mismatch: {len(blocks)} M2 blocks vs "
            f"{len(sources)} source sentences"
        )

    for index, (block, source) in enumerate(zip(blocks, sources), start=1):
        first_line = block.splitlines()[0]

        if not first_line.startswith("S "):
            raise ValueError(
                f'Gold M2 block {index} does not start with "S "'
            )

        gold_source = first_line[2:]

        if gold_source != source:
            raise ValueError(
                f"Source mismatch at example {index}:\n"
                f"  gold: {gold_source!r}\n"
                f"  data: {source!r}"
            )

    print(
        f"Gold alignment verified: {len(sources)} sentences.",
        flush=True
    )


def write_hypothesis_m2(sources, predictions, destination):
    """Convert source/prediction sentence pairs into ERRANT M2 edits."""

    if len(sources) != len(predictions):
        raise ValueError(
            f"Source/prediction count mismatch: "
            f"{len(sources)} vs {len(predictions)}"
        )

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    annotator = errant.load("en")

    with destination.open("w", encoding="utf-8") as out:
        for source, prediction in zip(sources, predictions):
            source_doc = annotator.parse(source)
            prediction_doc = annotator.parse(prediction)

            edits = annotator.annotate(source_doc, prediction_doc)

            out.write(f"S {source}\n")

            if edits:
                for edit in edits:
                    out.write(edit.to_m2() + "\n")
            else:
                out.write(
                    "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n"
                )

            out.write("\n")

    return destination


def main():
    parser = argparse.ArgumentParser(
        description="Convert GEC predictions to M2."
    )

    parser.add_argument(
        "--src",
        required=True,
        help="One source sentence per line"
    )

    parser.add_argument(
        "--pred",
        required=True,
        help="One model prediction per line"
    )

    parser.add_argument(
        "--output-m2",
        required=True,
        help="Output system M2 path"
    )

    parser.add_argument(
        "--gold-m2",
        help="Optional gold M2 for source alignment validation"
    )

    args = parser.parse_args()

    sources = Path(args.src).read_text(
        encoding="utf-8"
    ).splitlines()

    predictions = Path(args.pred).read_text(
        encoding="utf-8"
    ).splitlines()

    if args.gold_m2:
        validate_gold_alignment(sources, args.gold_m2)

    result = write_hypothesis_m2(
        sources,
        predictions,
        args.output_m2
    )

    print(f"Wrote system M2: {result}", flush=True)


if __name__ == "__main__":
    main()


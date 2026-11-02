#!/usr/bin/env python3
"""
Re-annotate CoNLL-style M2 gold references using ERRANT 3.x.

Preserves:
- Original source sentences and tokenisation
- Each human annotator's corrected sentence
- Multiple annotators and their original IDs
- Sentence blocks without any A lines

Validates:
- Edit boundaries and overlapping edits
- Reconstructed corrections
- ERRANT source/target tokenisation
- Round-trip reconstruction of generated ERRANT edits

Does NOT modify the input gold file.
"""

import argparse
from collections import Counter
from pathlib import Path

import errant


NOOP = "noop"
NONE = "-NONE-"


def parse_edit(line, sentence_id):
    """Read one M2 annotation line."""

    if not line.startswith("A "):
        raise ValueError(
            f"Sentence {sentence_id}: invalid edit line: {line!r}"
        )

    fields = line[2:].split("|||")

    if len(fields) != 6:
        raise ValueError(
            f"Sentence {sentence_id}: expected 6 M2 fields: {line!r}"
        )

    span, category, correction, requirement, comment, coder = fields

    positions = span.split()

    if len(positions) != 2:
        raise ValueError(
            f"Sentence {sentence_id}: invalid span: {span!r}"
        )

    return {
        "start": int(positions[0]),
        "end": int(positions[1]),
        "category": category,
        "correction": correction,
        "coder": int(coder),
        "raw": line,
    }


def parse_blocks(path):
    """Read all M2 blocks, including blocks without annotations."""

    text = path.read_text(encoding="utf-8-sig")

    for sentence_id, raw_block in enumerate(
        text.strip().split("\n\n")
    ):
        lines = [
            line.rstrip("\r")
            for line in raw_block.splitlines()
            if line.strip()
        ]

        if not lines or not lines[0].startswith("S "):
            raise ValueError(
                f"Sentence {sentence_id}: missing source line"
            )

        source = lines[0][2:]

        if any(not line.startswith("A ") for line in lines[1:]):
            raise ValueError(
                f"Sentence {sentence_id}: unexpected line type"
            )

        edits_by_coder = {}

        for line in lines[1:]:
            edit = parse_edit(line, sentence_id)
            coder = edit["coder"]

            edits_by_coder.setdefault(coder, []).append(edit)

        # No A lines: treat the unchanged sentence as annotator 0.
        if not edits_by_coder:
            edits_by_coder[0] = []

        yield sentence_id, source, edits_by_coder


def validate_edits(edits, source_length, sentence_id, coder):
    """
    Check edit spans without incorrectly treating a same-start
    insertion and replacement as overlapping.
    """

    non_noop = []

    for edit in edits:
        start = edit["start"]
        end = edit["end"]

        if edit["category"].lower() == NOOP:
            if start != -1 or end != -1:
                raise ValueError(
                    f"Sentence {sentence_id}, coder {coder}: "
                    "invalid noop span"
                )
            continue

        if not (0 <= start <= end <= source_length):
            raise ValueError(
                f"Sentence {sentence_id}, coder {coder}: "
                f"invalid span [{start}, {end})"
            )

        non_noop.append(edit)

    if len(non_noop) != len(edits):
        if non_noop:
            raise ValueError(
                f"Sentence {sentence_id}, coder {coder}: "
                "noop combined with real edits"
            )
        return []

    # Check pairwise conflicts.
    for i, a in enumerate(non_noop):
        for b in non_noop[i + 1:]:
            a0, a1 = a["start"], a["end"]
            b0, b1 = b["start"], b["end"]

            a_insert = a0 == a1
            b_insert = b0 == b1

            if a_insert and b_insert:
                # Multiple insertions at the same boundary are valid.
                # Preserve their original order in the M2 file.
                conflict = False

            elif a_insert:
                # Insertion at a replacement boundary is allowed.
                # Insertion strictly inside a replacement is ambiguous.
                conflict = b0 < a0 < b1

            elif b_insert:
                conflict = a0 < b0 < a1

            else:
                # Half-open interval intersection.
                conflict = max(a0, b0) < min(a1, b1)

            if conflict:
                raise ValueError(
                    f"Sentence {sentence_id}, coder {coder}: "
                    f"conflicting edits:\n"
                    f"  {a['raw']}\n"
                    f"  {b['raw']}"
                )

    return non_noop


def apply_edits(source_tokens, edits):
    """
    Apply validated M2 edits to a token sequence.

    Supports:
    - Multiple insertions at the same boundary
    - Insertion and replacement sharing a start position
    - Insertion at the end of a replacement
    - Replacements and deletions

    Insertions at the same boundary retain their original
    order in the M2 file.

    Assumes validate_edits() has already checked for
    invalid spans and genuinely overlapping edits.
    """

    insertions = {}
    replacements = {}

    for edit in edits:
        start = edit["start"]
        end = edit["end"]
        correction = edit["correction"]

        replacement_tokens = (
            [] if correction == NONE
            else correction.split()
        )

        if start == end:
            # Preserve original edit order.
            insertions.setdefault(start, []).extend(
                replacement_tokens
            )
        else:
            replacements[start] = {
                "end": end,
                "tokens": replacement_tokens
            }

    result = []
    i = 0
    n = len(source_tokens)

    while i <= n:
        # Emit all insertions at this token boundary.
        result.extend(insertions.get(i, []))

        # We've reached the end of the source sentence.
        if i == n:
            break

        if i in replacements:
            # Replace/delete the original token span.
            edit = replacements[i]
            result.extend(edit["tokens"])
            i = edit["end"]
        else:
            # Preserve an unchanged source token.
            result.append(source_tokens[i])
            i += 1

    return result


def regenerate(
    annotator,
    source,
    edits_by_coder,
    sentence_id,
    stats
):
    """Reconstruct and reannotate one source sentence."""

    source_tokens = source.split()
    original_doc = annotator.parse(source, tokenise=False)

    if [t.text for t in original_doc] != source_tokens:
        raise ValueError(
            f"Sentence {sentence_id}: "
            "ERRANT changed source tokenisation"
        )

    generated = ["S " + source]

    for coder in sorted(edits_by_coder):
        edits = validate_edits(
            edits_by_coder[coder],
            len(source_tokens),
            sentence_id,
            coder
        )

        corrected_tokens = apply_edits(source_tokens, edits)
        corrected_text = " ".join(corrected_tokens)

        corrected_doc = annotator.parse(
            corrected_text,
            tokenise=False
        )

        if [t.text for t in corrected_doc] != corrected_tokens:
            raise ValueError(
                f"Sentence {sentence_id}, coder {coder}: "
                "ERRANT changed target tokenisation"
            )

        new_edits = annotator.annotate(
            original_doc,
            corrected_doc,
            lev=False,
            merging="rules"
        )

        new_lines = [
            edit.to_m2(coder)
            for edit in new_edits
        ]

        # Validate ERRANT's generated M2 edits by reconstructing
        # the correction again from their serialized annotations.
        parsed_new = [
            parse_edit(line, sentence_id)
            for line in new_lines
        ]

        valid_new = validate_edits(
            parsed_new,
            len(source_tokens),
            sentence_id,
            coder
        )

        reconstructed = apply_edits(
            source_tokens,
            valid_new
        )

        if reconstructed != corrected_tokens:
            raise ValueError(
                f"Sentence {sentence_id}, coder {coder}: "
                "ERRANT round-trip reconstruction mismatch\n"
                f"  Expected: {corrected_tokens!r}\n"
                f"  Actual:   {reconstructed!r}"
            )

        if not new_lines:
            new_lines = [
                f"A -1 -1|||noop|||{NONE}|||"
                f"REQUIRED|||{NONE}|||{coder}"
            ]
            stats["no_correction_references"] += 1

        generated.extend(new_lines)

        stats["annotator_references"] += 1
        stats["original_edits"] += len(edits)
        stats["errant_edits"] += len(parsed_new)

    stats["sentences"] += 1

    return "\n".join(generated)


def main():
    parser = argparse.ArgumentParser(
        description="Reannotate M2 gold references using ERRANT"
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )

    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input not found: {args.input}")

    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")

    annotator = errant.load("en")
    stats = Counter()
    output_blocks = []

    for sentence_id, source, edits_by_coder in parse_blocks(
        args.input
    ):
        block = regenerate(
            annotator,
            source,
            edits_by_coder,
            sentence_id,
            stats
        )

        output_blocks.append(block)

    # Write only after all sentences pass validation.
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    args.output.write_text(
        "\n\n".join(output_blocks) + "\n",
        encoding="utf-8"
    )

    print("Gold reannotation completed")
    print(f"Output: {args.output}")

    for key in [
        "sentences",
        "annotator_references",
        "original_edits",
        "errant_edits",
        "no_correction_references"
    ]:
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()

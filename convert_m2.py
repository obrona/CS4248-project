"""Convert M2 annotations to aligned source and target text files.

Run ``python3 convert_m2.py`` to convert data/raw/**/*.m2 to data/processed.
Each input produces <stem>.src.txt and <stem>.tgt.txt, with one sentence per
line and matching line numbers for each source/target pair.
Sentences keep their original token spacing. Annotator 0 is used by default;
alternative annotators are never combined into the same target sentence.
"""

import argparse
from pathlib import Path


def sentence_pair(lines, annotator_id=0):
    """Apply one annotator's edits using offsets in the original source tokens."""
    if not lines[0].startswith("S "):
        raise ValueError("M2 block must begin with an S sentence")
    source = lines[0][2:]
    tokens = source.split()
    edits = []
    for line in lines[1:]:
        if not line.startswith("A "):
            raise ValueError(f"Unexpected M2 line: {line!r}")
        fields = line[2:].split("|||")
        if len(fields) != 6:
            raise ValueError(f"Malformed M2 annotation: {line!r}")
        if int(fields[5]) != annotator_id:
            continue
        start, end = map(int, fields[0].split())
        if fields[1] in {"noop", "UNK", "Um"}:
            continue
        if not 0 <= start <= end <= len(tokens):
            raise ValueError(f"Invalid edit offsets: {line!r}")
        replacement = [] if fields[2] == "-NONE-" else fields[2].split()
        edits.append((start, end, replacement))

    # Build from the original tokens so earlier edits cannot shift later offsets.
    target = []
    cursor = 0
    for start, end, replacement in sorted(edits, key=lambda edit: (edit[0], edit[1])):
        if start < cursor:
            raise ValueError(f"Overlapping edits in sentence: {source!r}")
        target.extend(tokens[cursor:start])
        target.extend(replacement)
        cursor = end
    target.extend(tokens[cursor:])
    return source, " ".join(target)


def convert_file(input_path, source_path, target_path, annotator_id=0):
    """Stream blocks from an M2 file, including a final block without a blank line."""
    source_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    block = []
    with input_path.open(encoding="utf-8") as input_file, source_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as source_file, target_path.open("w", encoding="utf-8", newline="\n") as target_file:
        def write_block():
            nonlocal count
            source, target = sentence_pair(block, annotator_id)
            source_file.write(f"{source}\n")
            target_file.write(f"{target}\n")
            count += 1

        for line in input_file:
            line = line.rstrip("\r\n")
            if line.strip():
                block.append(line)
            elif block:
                write_block()
                block = []
        if block:
            write_block()
    return count


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=root / "data/processed")
    parser.add_argument("--annotator-id", type=int, default=0)
    args = parser.parse_args()
    files = sorted(args.input_dir.rglob("*.m2"))
    if not files:
        parser.error(f"No .m2 files found in {args.input_dir}")
    total = 0
    for input_path in files:
        relative_path = input_path.relative_to(args.input_dir)
        source_path = args.output_dir / relative_path.with_suffix(".src.txt")
        target_path = args.output_dir / relative_path.with_suffix(".tgt.txt")
        count = convert_file(input_path, source_path, target_path, args.annotator_id)
        total += count
        print(f"{input_path.name} -> {source_path.name} + {target_path.name}: {count:,} pairs")
    print(f"Converted {len(files)} files, {total:,} pairs total.")


if __name__ == "__main__":
    main()

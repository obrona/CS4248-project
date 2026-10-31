"""Rebuild alpha=1-only M2 files from tl2's committed sentence pairs."""
from pathlib import Path
from difflib import SequenceMatcher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
args = parser.parse_args()
repo = args.repo.expanduser().resolve()

corpora = [
    ("fce-src-tgt", "fce.train.gold.bea19"),
    ("fce-src-tgt", "fce.dev.gold.bea19"),
    ("lang8-src-tgt", "lang8.train.auto.bea19"),
    ("nucle-src-tgt", "nucle.train.gold.bea19"),
    ("w&-src-tgt", "A.train.gold.bea19"),
    ("w&-src-tgt", "B.train.gold.bea19"),
    ("w&-src-tgt", "C.train.gold.bea19"),
]

destination = repo / "data/raw"
destination.mkdir(parents=True, exist_ok=True)

for folder, stem in corpora:
    output = destination / f"{stem}.m2"
    if output.exists() or output.is_symlink():
        print("SKIP:", output)
        continue

    source = (repo / folder / f"{stem}.src").read_text(
        encoding="utf-8"
    ).splitlines()
    target = (repo / folder / f"{stem}.tgt").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(source) == len(target), stem

    with output.open("x", encoding="utf-8") as out:
        for src, tgt in zip(source, target):
            out.write(f"S {src}\n")
            edits = 0
            src_words, tgt_words = src.split(" "), tgt.split(" ")
            matcher = SequenceMatcher(
                None, src_words, tgt_words, autojunk=False
            )
            for tag, i, j, a, b in matcher.get_opcodes():
                if tag != "equal":
                    correction = " ".join(tgt_words[a:b])
                    out.write(
                        f"A {i} {j}|||OTHER|||{correction}"
                        "|||REQUIRED|||-NONE-|||0\n"
                    )
                    edits += 1
            if not edits:
                out.write(
                    "A -1 -1|||noop|||-NONE-"
                    "|||REQUIRED|||-NONE-|||0\n"
                )
            out.write("\n")
    print("CREATED:", output)

#!/usr/bin/env python3
"""Compile ERRANT compare_m2 -cat 3 outputs for any number of models.

Retains raw category names exactly, and distinguishes absent categories
from zero scores.

For the current CoNLL-2014 analysis, gold references and model
predictions have been annotated using ERRANT-style categories.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path


HEADER = re.compile(
    r'^Category\s+TP\s+FP\s+FN\s+P\s+R\s+F(?:0\.5|[\d.]+)\s*$'
)

OVERALL_HEADER = re.compile(
    r'^TP\s+FP\s+FN\s+Prec\s+Rec\s+F(?:0\.5|[\d.]+)\s*$'
)

INT = re.compile(r'^\d+$')

MODERN = re.compile(
    r'^[MUR]:(?:[A-Z][A-Z0-9_]*(?::[A-Z][A-Z0-9_]*)*)$'
)

FIELDS = [
    'tp',
    'fp',
    'fn',
    'precision',
    'recall',
    'f05',
    'selected_reference_support',
    'predicted_count'
]


def score(tp, fp, fn):
    """Reproduce ERRANT's F0.5 calculation and empty-class behaviour."""

    p = tp / (tp + fp) if fp else 1.0
    r = tp / (tp + fn) if fn else 1.0

    f = (
        1.25 * p * r / (0.25 * p + r)
        if p + r else 0.0
    )

    return round(p, 4), round(r, 4), round(f, 4)


def read_report(path):
    """Parse and validate one ERRANT category report."""

    lines = path.read_text(encoding='utf-8').splitlines()

    header_indices = [
        i for i, line in enumerate(lines)
        if HEADER.fullmatch(line.strip())
    ]

    if len(header_indices) != 1:
        raise ValueError(
            f'{path}: expected exactly one category header '
            f'(run errant_compare -cs -cat 3), '
            f'found {len(header_indices)}'
        )

    start = header_indices[0] + 1
    rows = {}

    # Read category-level results.
    for line in lines[start:]:
        s = line.strip()

        if not s or s.startswith('='):
            break

        parts = s.split()

        if (
            len(parts) != 7
            or not all(INT.fullmatch(x) for x in parts[1:4])
        ):
            raise ValueError(
                f'{path}: unexpected category row: {line!r}'
            )

        cat = parts[0]

        if cat in rows:
            raise ValueError(
                f'{path}: duplicated category {cat!r}'
            )

        tp, fp, fn = map(int, parts[1:4])
        published = tuple(map(float, parts[4:7]))
        computed = score(tp, fp, fn)

        # Allow for ERRANT's four-decimal rounding.
        if any(
            abs(a - b) > 0.00011
            for a, b in zip(published, computed)
        ):
            raise ValueError(
                f'{path}: score mismatch for {cat}: '
                f'{published} vs {computed}'
            )

        rows[cat] = dict(zip(
            FIELDS,
            (
                tp,
                fp,
                fn,
                *published,
                tp + fn,
                tp + fp
            )
        ))

    if not rows:
        raise ValueError(f'{path}: no category rows')

    # Read overall ERRANT results.
    overall_heads = [
        i for i, line in enumerate(lines)
        if OVERALL_HEADER.fullmatch(line.strip())
    ]

    if len(overall_heads) != 1:
        raise ValueError(
            f'{path}: expected exactly one overall score header'
        )

    ix = overall_heads[0] + 1

    if ix >= len(lines):
        raise ValueError(f'{path}: missing overall values')

    parts = lines[ix].split()

    if (
        len(parts) != 6
        or not all(INT.fullmatch(x) for x in parts[:3])
    ):
        raise ValueError(
            f'{path}: invalid overall row {lines[ix]!r}'
        )

    tp, fp, fn = map(int, parts[:3])
    printed = tuple(map(float, parts[3:]))
    computed = score(tp, fp, fn)

    if any(
        abs(a - b) > 0.00011
        for a, b in zip(printed, computed)
    ):
        raise ValueError(
            f'{path}: invalid overall metrics: '
            f'{printed} vs {computed}'
        )

    # Ensure the category counts add up to overall counts.
    sums = tuple(
        sum(row[k] for row in rows.values())
        for k in ('tp', 'fp', 'fn')
    )

    if sums != (tp, fp, fn):
        raise ValueError(
            f'{path}: category TP/FP/FN totals {sums} '
            f'do not match overall {(tp, fp, fn)}'
        )

    overall = dict(zip(
        ['tp', 'fp', 'fn', 'precision', 'recall', 'f05'],
        (tp, fp, fn, *printed)
    ))

    return rows, overall


def write_csv(path, fields, records):
    """Write CSV with a consistent header and UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        'w',
        encoding='utf-8',
        newline=''
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(records)

def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        '--model',
        action='append',
        required=True,
        metavar='NAME=REPORT',
        help=(
            'Repeat for each model, e.g. '
            '--model uniform=analysis/errant/conll14/'
            'uniform_categories.txt'
        )
    )

    parser.add_argument(
        '--out',
        type=Path,
        default=Path('analysis/errant/conll14/compiled')
    )

    args = parser.parse_args()

    results = {}
    overall = {}
    input_paths = {}

    # Load every supplied model.
    for spec in args.model:
        if '=' not in spec:
            parser.error(
                f'--model requires NAME=REPORT: {spec!r}'
            )

        name, filename = spec.split('=', 1)
        name = name.strip()

        if (
            not name
            or not filename.strip()
            or name in results
        ):
            parser.error(
                f'empty or duplicate model name/path: {spec!r}'
            )

        path = Path(filename)

        if not path.is_file():
            parser.error(f'not a file: {path}')

        results[name], overall[name] = read_report(path)
        input_paths[name] = str(path.resolve())

    # Union of all observed categories across models.
    categories = sorted(
        set().union(
            *(set(d) for d in results.values())
        )
    )

    long_rows = []
    wide_rows = []

    for cat in categories:
        taxonomy = (
            'ERRANT-style'
            if MODERN.fullmatch(cat)
            else 'legacy_or_other'
        )

        wide = {
            'category': cat,
            'taxonomy': taxonomy
        }

        for name, cats in results.items():
            present = cat in cats
            d = cats.get(cat)

            # Long format: one row per model and category.
            long_rows.append({
                'model': name,
                'category': cat,
                'taxonomy': taxonomy,
                'present': int(present),
                **{
                    k: d[k] if present else ''
                    for k in FIELDS
                }
            })

            # Wide format: one row per category.
            wide[f'{name}__present'] = int(present)

            for k in FIELDS:
                wide[f'{name}__{k}'] = (
                    d[k] if present else ''
                )

        wide_rows.append(wide)

    out = args.out

    # Export long-format results.
    write_csv(
        out / 'categories_long.csv',
        [
            'model',
            'category',
            'taxonomy',
            'present',
            *FIELDS
        ],
        long_rows
    )

    # Export side-by-side model comparisons.
    wide_fields = ['category', 'taxonomy']

    for model in results:
        wide_fields.append(f'{model}__present')

        for field in FIELDS:
            wide_fields.append(f'{model}__{field}')

    write_csv(
        out / 'categories_wide.csv',
        wide_fields,
        wide_rows
    )

    # Export overall ERRANT scores.
    write_csv(
        out / 'overall_errant.csv',
        [
            'model',
            'tp',
            'fp',
            'fn',
            'precision',
            'recall',
            'f05'
        ],
        [
            {'model': name, **d}
            for name, d in overall.items()
        ]
    )

    # Save methodological notes.
    metadata = {
        'reports': input_paths,
        'evaluation': (
            'ERRANT -cs -cat 3; '
            'category labels preserved verbatim'
        ),
        'missing_category': (
            'empty CSV cells with present=0; not zero'
        ),
        'selected_reference_support': (
            'TP+FN, depends on model-specific '
            'reference selection'
        ),
        'taxonomy_note': (
            'Gold references and model predictions use '
            'ERRANT-normalised grammatical error categories.'
        ),
        'empty_class_warning': (
            'ERRANT defines p=1 if FP=0 and r=1 '
            'if FN=0, even when TP=0; '
            'refer to raw counts.'
        )
    }

    (out / 'metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n',
        encoding='utf-8'
    )

    print(
        f'Compiled {len(results)} models and '
        f'{len(categories)} unique categories '
        f'into {out}/'
    )

    print(
        'Category labels use ERRANT-normalised taxonomy.'
    )

    print(
        'Overall F0.5: ' +
        ', '.join(
            f'{m}={v["f05"]:.4f}'
            for m, v in overall.items()
        )
    )


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as e:
        sys.exit(f'ERROR: {e}')

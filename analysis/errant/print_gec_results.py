#!/usr/bin/env python3
"""Print and save a readable summary of existing CoNLL-2014 ERRANT analyses.
Run from the project root: python analysis/errant/print_gec_results.py
No third-party dependencies; this script does not recalculate any metrics.
"""
import argparse
import csv
import textwrap
from pathlib import Path

FOCUS = ('R:SPELL', 'R:NOUN:NUM', 'R:PUNCT', 'M:PUNCT', 'M:DET')
EXAMPLES = (684, 619, 947, 233)


def load(path):
    if not path.is_file():
        return None
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def pick(row, *names):
    for name in names:
        if name in row and row[name] not in ('', None):
            return row[name]
    return '—'


def number(value, digits=4):
    try:
        return f'{float(value):.{digits}f}'
    except (ValueError, TypeError):
        return str(value)


def table(headers, rows):
    rows = [[str(x) for x in r] for r in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    def line(items):
        return '  '.join(str(x).ljust(widths[i]) for i, x in enumerate(items))
    return '\n'.join([line(headers), '  '.join('-' * w for w in widths), *(line(r) for r in rows)])


def section(title):
    return '\n' + '=' * 90 + '\n' + title + '\n' + '=' * 90


def category(row):
    return pick(row, 'category', 'cat', 'error_type')


def numeric(row, *names):
    value = pick(row, *names)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, default=Path('analysis/errant/conll14'))
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--all-categories', action='store_true', help='Show all categories, not just the five focus categories')
    args = ap.parse_args()
    base = args.base
    out = args.out or base / 'analysis_summary.txt'
    compiled = base / 'compiled_normalised'
    chunks = ['GEC MODEL COMPARISON — CoNLL-2014', 'Uniform vs frequency-weighted BART', 'Source: saved ERRANT analysis CSVs; no scores recomputed']

    overall = load(compiled / 'overall_errant.csv')
    chunks.append(section('1. OVERALL ERRANT RESULTS'))
    if overall is None:
        chunks.append('Missing: ' + str(compiled / 'overall_errant.csv'))
    elif not overall:
        chunks.append('Overall CSV is empty.')
    else:
        cols = list(overall[0])
        chunks.append(table(cols, [[r.get(c, '') for c in cols] for r in overall]))

    wide = load(compiled / 'categories_wide.csv')
    chunks.append(section('2. CATEGORY PERFORMANCE (ERRANT F0.5 AND COUNTS)'))
    if wide is None:
        chunks.append('Missing: ' + str(compiled / 'categories_wide.csv'))
    else:
        selected = wide if args.all_categories else [r for r in wide if category(r) in FOCUS]
        selected.sort(key=lambda r: (FOCUS.index(category(r)) if category(r) in FOCUS else len(FOCUS), category(r)))
        data = []
        for r in selected:
            u = numeric(r, 'uniform__f05')
            f = numeric(r, 'frequency__f05')
            diff = f-u if f is not None and u is not None else None
            data.append([category(r), number(u), number(f), f'{diff:+.4f}' if diff is not None else '—',
                         '/'.join(pick(r, f'uniform__{x}') for x in ('tp','fp','fn')),
                         '/'.join(pick(r, f'frequency__{x}') for x in ('tp','fp','fn'))])
        chunks.append(table(['Category', 'Uniform', 'Frequency', 'Delta F0.5', 'U TP/FP/FN', 'F TP/FP/FN'], data))
        chunks.append('Delta = frequency minus uniform. TP/FP/FN are ERRANT edit counts.')

    boot = load(base / 'bootstrap_categories.csv')
    rand = load(base / 'randomisation_categories.csv')
    chunks.append(section('3. STATISTICAL TESTING (10,000 RESAMPLES / RANDOMISATIONS)'))
    if boot is None:
        chunks.append('Missing: ' + str(base / 'bootstrap_categories.csv'))
    if rand is None:
        chunks.append('Missing: ' + str(base / 'randomisation_categories.csv'))
    if boot is not None:
        chunks.append('\nBOOTSTRAP — original columns preserved (for exact CI interpretation):')
        selected = boot if args.all_categories else [r for r in boot if category(r) in FOCUS]
        if selected:
            cols = list(selected[0])
            chunks.append(table(cols, [[r.get(c, '') for c in cols] for r in selected]))
        else:
            chunks.append('No matching categories; available columns: ' + (', '.join(boot[0]) if boot else '(empty file)'))
    if rand is not None:
        chunks.append('\nRANDOMISATION — original columns preserved (including Holm adjustment):')
        selected = rand if args.all_categories else [r for r in rand if category(r) in FOCUS]
        if selected:
            cols = list(selected[0])
            chunks.append(table(cols, [[r.get(c, '') for c in cols] for r in selected]))
        else:
            chunks.append('No matching categories; available columns: ' + (', '.join(rand[0]) if rand else '(empty file)'))
    chunks.append('\nInterpretation: ordinary per-category bootstrap CIs are not simultaneous 95% intervals; use Holm-adjusted randomisation p-values for multiple-testing claims.')

    examples = load(base / 'analysed_disagreements.csv')
    chunks.append(section('4. SELECTED QUALITATIVE EXAMPLES'))
    if examples is None:
        chunks.append('Missing: ' + str(base / 'analysed_disagreements.csv'))
    else:
        by_id = {str(r.get('sentence_id')): r for r in examples}
        for sid in EXAMPLES:
            r = by_id.get(str(sid))
            if r is None:
                chunks.append(f'Sentence {sid}: not found')
                continue
            chunks.append(f'\nSentence {sid} | {pick(r,"category")} | {pick(r,"outcome")} | Same gold ref: {pick(r,"same_reference")}')
            for label, field in [('SOURCE', 'source'), ('UNIFORM', 'uniform_prediction'), ('FREQUENCY', 'frequency_prediction')]:
                chunks.append(textwrap.fill(f'{label}: {r.get(field, "")}', width=110, subsequent_indent='    '))
            chunks.append('Uniform TP/FP/FN: ' + '/'.join(r.get('uniform_'+x,'?') for x in ('tp','fp','fn')))
            chunks.append('Frequency TP/FP/FN: ' + '/'.join(r.get('frequency_'+x,'?') for x in ('tp','fp','fn')))
            chunks.append('Selected gold edits: ' + r.get('uniform_gold_edits',''))

    chunks.append(section('5. REPORTING REMINDERS'))
    chunks.append('1) Report per-category differences as descriptive unless Holm-adjusted p < 0.05.')
    chunks.append('2) Distinguish ERRANT metrics here from official CoNLL M2 scorer metrics.')
    chunks.append('3) Sentence IDs in analysed_disagreements.csv are zero-based.')
    report = '\n'.join(line.rstrip() for line in '\n'.join(chunks).splitlines()) + '\n'

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding='utf-8')

    print(report)
    print(f'\nSaved report: {out}')

if __name__ == '__main__':
    main()

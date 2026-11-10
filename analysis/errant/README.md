# ERRANT Analysis Pipeline

Reusable CoNLL-2014 grammatical error correction analysis for the CS4248 project.

## Overview

This pipeline evaluates model predictions using ERRANT and compiles per-category precision, recall and F0.5.

Existing models:
- `standard`: Standard BART fine-tuning
- `uniform`: Uniform edit-weighted BART
- `frequency`: Frequency-aware edit-weighted BART

All models use the same 1,312 CoNLL-2014 test sentences.

## Files

| File | Purpose |
|---|---|
| `normalise_gold.py` | Reannotate original CoNLL gold using ERRANT |
| `compile_categories.py` | Compile category reports across models |
| `conll14/predictions/*.txt` | Model predictions, one sentence per line |
| `conll14/{model}.m2` | ERRANT-generated hypothesis edits |
| `conll14/{model}_normalised_categories.txt` | Individual model evaluation reports |
| `conll14/compiled_normalised/` | Combined results across models |
| `conll14/prediction_disagreements.tsv` | Existing sentence-level comparison of model outputs |

`gold_errant.m2` and `sources.txt` are generated locally and are not committed.

## Setup

From the repository root, activate the project's Python environment and ensure ERRANT 3.0.2 is installed.

The original CoNLL-2014 gold file must be available at:

`conll14st-test-data/noalt/official-2014.combined.m2`

### 1. Prepare the reference data

Generate `sources.txt` from the original CoNLL M2 file:

```bash
python - <<'PY'
from pathlib import Path

gold = Path("conll14st-test-data/noalt/official-2014.combined.m2")
out = Path("analysis/errant/conll14/sources.txt")

blocks = gold.read_text(encoding="utf-8").strip().split("\n\n")
sources = [block.splitlines()[0][2:] for block in blocks]

assert len(sources) == 1312
assert all(block.startswith("S ") for block in blocks)

out.write_text("\n".join(sources) + "\n", encoding="utf-8")
print(f"Generated {len(sources)} source sentences.")
PY
```

Generate the normalised gold:

```bash
python analysis/errant/normalise_gold.py \
  --input conll14st-test-data/noalt/official-2014.combined.m2 \
  --output analysis/errant/conll14/gold_errant.m2
```

The normalisation preserves the original corrected sentences for each annotator while regenerating their edit categories with ERRANT.

### 2. Evaluate a new model

Save the model's predictions as `analysis/errant/conll14/predictions/NEW_MODEL.txt`.

There must be exactly 1,312 lines, aligned with the original CoNLL source sentences.

Generate hypothesis edits:

```bash
errant_parallel \
  -orig analysis/errant/conll14/sources.txt \
  -cor analysis/errant/conll14/predictions/NEW_MODEL.txt \
  -out analysis/errant/conll14/NEW_MODEL.m2
```

Evaluate:

```bash
errant_compare \
  -hyp analysis/errant/conll14/NEW_MODEL.m2 \
  -ref analysis/errant/conll14/gold_errant.m2 \
  -cs -cat 3 \
  > analysis/errant/conll14/NEW_MODEL_normalised_categories.txt
```

### 3. Compile model results

Run:

```bash
python analysis/errant/compile_categories.py \
  --model standard=analysis/errant/conll14/standard_normalised_categories.txt \
  --model uniform=analysis/errant/conll14/uniform_normalised_categories.txt \
  --model frequency=analysis/errant/conll14/frequency_normalised_categories.txt \
  --model NEW_MODEL=analysis/errant/conll14/NEW_MODEL_normalised_categories.txt \
  --out analysis/errant/conll14/compiled_normalised
```

Remove the `NEW_MODEL` argument if only compiling the three existing models. Add additional `--model` arguments for further models.

## Compiled outputs

- `categories_long.csv`: One row per model/category combination.
- `categories_wide.csv`: Side-by-side category comparisons.
- `overall_errant.csv`: Overall TP, FP, FN, precision, recall and F0.5.
- `metadata.json`: Compilation metadata.

## Existing normalised ERRANT results

| Model | Precision | Recall | F0.5 |
|---|---:|---:|---:|
| Standard | 0.7152 | 0.2337 | 0.5065 |
| Uniform | 0.6302 | 0.4060 | 0.5675 |
| Frequency | 0.6149 | 0.4133 | 0.5602 |

## Important methodological notes

1. Use `gold_errant.m2` for consistent ERRANT category comparisons. The original CoNLL gold contains legacy error categories that should not be mixed directly with modern ERRANT categories.
2. Use the original CoNLL gold and M2Scorer for official CoNLL-2014 benchmark results. These scores differ from the ERRANT results above.
3. ERRANT can select different annotator references for different models, so category-level reference support may vary.
4. The compiled category CSVs are aggregate statistics. For paired bootstrap significance testing, resample the same sentence indices across models and recompute the metric.
5. All new models must use the same source sentences and sentence ordering.

## Per-category significance and qualitative analysis

This analysis compares the uniform and frequency-weighted models on CoNLL-2014 using ERRANT-normalised categories.

### Viewing the results

All results are stored in `analysis/errant/conll14/` and are included in the repository.

To print a formatted summary in the terminal, run from the project root:

```bash
python analysis/errant/print_gec_results.py
```

To display the full per-category results:

```bash
python analysis/errant/print_gec_results.py --all-categories
```

The consolidated results are also saved in `analysis/errant/conll14/analysis_summary.txt`, which can be opened directly without running any scripts.

### Results files

| File | Contents |
|---|---|
| `analysis_summary.txt` | Readable summary of the main results |
| `compiled_normalised/categories_wide.csv` | Per-category TP, FP, FN, precision, recall and F0.5 for all three models |
| `bootstrap_categories.csv` | Paired bootstrap confidence intervals for differences between uniform and frequency |
| `randomisation_categories.csv` | Approximate randomisation p-values and Holm-adjusted p-values |
| `analysed_disagreements.csv` | Qualitative examples of differences between model predictions |
| `sentence_categories.csv` | Sentence-level TP, FP and FN counts used in significance testing |

### Reproducing the analysis

The analysis scripts are located in `analysis/errant/`:

- `bootstrap_categories.py` — sentence-level category extraction and paired bootstrap testing.
- `randomisation_categories.py` — paired approximate randomisation with multiple-comparison correction.
- `analyse_disagreements.py` — qualitative comparison of differing model predictions.
- `print_gec_results.py` — formatted reporting of results.

The scripts used ERRANT-normalised gold references (`conll14/gold_errant.m2`), which are generated locally and excluded from Git. The committed results can be viewed without this file, but reproducing the analysis requires regenerating it.

**Interpretation:** No category-level differences were statistically significant after Holm correction. Individual category differences and qualitative examples should therefore be treated as exploratory findings.

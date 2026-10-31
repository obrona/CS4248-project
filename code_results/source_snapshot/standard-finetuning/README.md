# Standard fine-tuning

This fine-tunes `facebook/bart-base` for grammatical error correction using the standard token-level cross-entropy loss.
Training has two stages:

1. **Stage 1** (`finetune.py`): 1 epoch on all BEA-2019 training data (FCE train+dev, Lang-8, NUCLE, W&I A/B/C).
2. **Stage 2** (`finetune-2.py`): continues from the stage-1 model on the clean data only (Lang-8 dropped). Uses lr 1e-5 for 4 epochs.

Every step runs on the SoC Compute Cluster through `run.sh`. The first argument chooses the mode, and all remaining arguments go to the Python script.
Always submit from this directory:

```bash
cd standard-finetuning
```

On first run, `run.sh` creates `../.venv` and installs the dependencies. It caches Hugging Face downloads in `output/hf_cache`.
Slurm logs go to `output/slurm-bart-gec-<jobid>.{out,err}`.

## Prerequisites

The training data is the parallel `*-src-tgt/` folders at the repo root. The test data is `conll14st-test-data/` and `bea-2019-test-data/`. All of these are committed.

## 1. Stage 1: fine-tune

```bash
sbatch run.sh train                 # same as: sbatch run.sh
```

Output goes to `output/`:

| Path | Contents |
|-|-|
| `output/bart-base-gec/` | final model + tokenizer |
| `output/checkpoints/` | trainer checkpoints. Resubmitting the job resumes from the latest one |
| `output/train_log.json`, `output/train_config.json` | loss log and run config |

## 2. Stage 2: continue on clean data

Run this after stage 1 has finished:

```bash
sbatch run.sh train2
```

This reads `output/bart-base-gec/` and writes everything (model, checkpoints, logs) to `output/stage2/`. The model ends up in `output/stage2/bart-base-gec/`.

## 3. Evaluate

The eval scripts load `output/bart-base-gec` (the stage-1 model) by default.

### CoNLL-2014 (scored with M2Scorer)

```bash
# stage 1
sbatch run.sh eval
# stage 2: pass BOTH flags, or the stage-1 results in output/eval_conll14/ get overwritten
sbatch run.sh eval --model_dir output/stage2/bart-base-gec --eval_dir output/stage2/eval_conll14
```

Results are written to the eval dir:

- `scores.json`: P / R / F0.5
- `predictions.txt`, `sources.txt`, `results.jsonl`
- `m2scorer_verbose.txt`

M2Scorer needs Python 2. On first use it is cloned into `output/tools/` automatically, along with a Python 2.7 interpreter.

### BEA-2019 (submitted to Codabench)

The BEA-2019 test references are not public, so this step only generates the corrections:

```bash
# stage 1
sbatch run.sh bea
# stage 2
sbatch run.sh bea --model_dir output/stage2/bart-base-gec --eval_dir output/stage2/eval_bea19
```

Upload `<eval_dir>/submission.zip` to https://www.codabench.org/competitions/10960/.

## Smoke tests

```bash
sbatch run.sh train  --max_train_samples 64 --max_steps 2
sbatch run.sh train2 --max_train_samples 64 --max_steps 2
sbatch run.sh eval   --limit 20
sbatch run.sh bea    --limit 20
```

Smoke runs write to the same output folders as real runs, so delete them before starting a real run.
Options such as `--lr`, `--epochs`, `--batch_size` and `--num_beams` are listed by `python <script>.py --help`.

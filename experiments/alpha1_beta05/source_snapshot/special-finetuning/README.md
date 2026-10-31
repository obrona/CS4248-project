# Category-weighted fine-tuning

This fine-tunes `facebook/bart-base` for grammatical error correction. The data and hyperparameters match `standard-finetuning/`, but the cross-entropy loss is weighted.
Target tokens produced by a correction get a weight that depends on the edit's ERRANT error category: rarer categories get larger weights. Copied tokens keep weight 1.
The formula is in the `finetune.py` docstring. How tokens are matched to categories is explained in [token-category-matching.md](token-category-matching.md).

Training has two stages:

1. **Stage 1** (`finetune.py`): 1 epoch on all BEA-2019 training data (FCE train+dev, Lang-8, NUCLE, W&I A/B/C).
2. **Stage 2** (`finetune-2.py`): continues a stage-1 run on the clean data only (Lang-8 dropped). Uses lr 1e-5 for 4 epochs.
   It reuses that run's alpha / beta / w_max and recomputes the category weights on the clean data.

Each set of hyperparameters gets its own **run dir**, `output/a{alpha}_b{beta}_w{w_max}/`. The default is `output/a3_b0.5_w10`.

Every step runs on the SoC Compute Cluster through `run.sh`. The first argument chooses the mode, and all remaining arguments go to the Python script.
Always submit from this directory:

```bash
cd special-finetuning
```

On first run, `run.sh` creates `../.venv` and installs the dependencies. It caches Hugging Face downloads in `output/hf_cache`.
Slurm logs go to `output/slurm-bart-gec-weighted-<jobid>.{out,err}`.

## Prerequisites

The category labels come from the BEA-2019 **M2 files**, which must be in `../data/raw/` (`*.train.gold.bea19.m2`, `lang8.train.auto.bea19.m2`, ...).
They are gitignored, so copy them to the cluster separately (e.g. with `../transfer_files.sh`). `run.sh` exits if they are missing.
The parallel `*-src-tgt/` folders and the test data are committed.

## 1. Stage 1: fine-tune

```bash
sbatch run.sh train                                 # alpha=3 beta=0.5 w_max=10 -> output/a3_b0.5_w10/
sbatch run.sh train --alpha 2 --beta 1 --w_max 10   # -> output/a2_b1_w10/
```

Output goes to the run dir:

| Path | Contents |
|-|-|
| `<run_dir>/bart-base-gec/` | final model + tokenizer |
| `<run_dir>/checkpoints/` | trainer checkpoints. Resubmitting the job resumes from the latest one |
| `<run_dir>/train_config.json` | run config, including alpha / beta / w_max and the category weights |

## 2. Stage 2: continue on clean data

Run this after stage 1 has finished:

```bash
sbatch run.sh train2                                # continues output/a3_b0.5_w10
sbatch run.sh train2 --run_dir output/a2_b1_w10     # continues another run
```

This reads `<run_dir>/bart-base-gec/` and writes to `<run_dir>/stage2/`. The model ends up in `<run_dir>/stage2/bart-base-gec/`.
Because `stage2/` is laid out like a run dir, you can pass it as `--run_dir` to the eval scripts.

## 3. Evaluate

The eval scripts load `<run_dir>/bart-base-gec` and write results inside the run dir. The default run dir is `output/a3_b0.5_w10`.

### CoNLL-2014 (scored with M2Scorer)

```bash
sbatch run.sh eval                                        # stage 1 -> output/a3_b0.5_w10/eval_conll14/
sbatch run.sh eval --run_dir output/a3_b0.5_w10/stage2    # stage 2 -> output/a3_b0.5_w10/stage2/eval_conll14/
sbatch run.sh eval --run_dir output/a2_b1_w10             # another run
```

Results are written to `<run_dir>/eval_conll14/`:

- `scores.json`: P / R / F0.5
- `predictions.txt`, `sources.txt`, `results.jsonl`
- `m2scorer_verbose.txt`

M2Scorer needs Python 2. On first use it is cloned into `output/tools/` automatically, along with a Python 2.7 interpreter.

### BEA-2019 (submitted to Codabench)

The BEA-2019 test references are not public, so this step only generates the corrections:

```bash
sbatch run.sh bea                                         # stage 1 -> output/a3_b0.5_w10/eval_bea19/
sbatch run.sh bea --run_dir output/a3_b0.5_w10/stage2     # stage 2 -> output/a3_b0.5_w10/stage2/eval_bea19/
```

Upload `<run_dir>/eval_bea19/submission.zip` to https://www.codabench.org/competitions/10960/.

## Smoke tests

```bash
sbatch run.sh train  --max_train_samples 64 --max_steps 2
sbatch run.sh train2 --max_train_samples 64 --max_steps 2
sbatch run.sh eval   --limit 20
sbatch run.sh bea    --limit 20
```

Smoke runs write to the same run dir as real runs, so delete it before starting a real run.
Options such as `--lr`, `--epochs`, `--batch_size` and `--num_beams` are listed by `python <script>.py --help`.

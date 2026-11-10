# CS4248 Project: Grammatical Error Correction with BART

We fine-tune `facebook/bart-base` for grammatical error correction (GEC) on the BEA-2019 training data and evaluate on CoNLL-2014 and BEA-2019. There are two training setups:

| Folder | Loss | Details |
|-|-|-|
| [`standard-finetuning/`](standard-finetuning/README.md) | standard token-level cross-entropy | the baseline |
| [`special-finetuning/`](special-finetuning/README.md) | cross-entropy weighted by ERRANT error category | see [token-category-matching.md](special-finetuning/token-category-matching.md) |

This README is a quick start. The per-folder READMEs cover every option.

> **Note: the final submission does not use stage 2.**
> Both folders include a second training stage (`finetune-2.py` / `run.sh train2`) that continues on the clean data only, without Lang-8.
> All models in our final submission are **stage-1 models only**. The stage-2 scripts are kept for reference.

## Models in the final submission

| Name in the report | Training command (run from its folder) | Model / results dir |
|-|-|-|
| Standard | `sbatch run.sh train` in `standard-finetuning/` | `standard-finetuning/output/` |
| Uniform edit-weighted (beta = 0) | `sbatch run.sh train --alpha 3 --beta 0 --w_max 10` in `special-finetuning/` | `special-finetuning/output/a3_b0_w10/` |
| Frequency-aware edit-weighted (beta = 0.5) | `sbatch run.sh train` in `special-finetuning/` (default: alpha=3, beta=0.5, w_max=10) | `special-finetuning/output/a3_b0.5_w10/` |

## Requirements

- Everything runs on the **SoC Compute Cluster (Slurm)**. Each step is a job submitted with `sbatch run.sh <mode>`.
- Python 3.12 and a CUDA GPU. `run.sh` requests one H100 by default. Edit the `#SBATCH --gpus` line to use another GPU.
- On first run, `run.sh` creates `.venv/` at the repo root and installs the dependencies (`torch`, `transformers`, `datasets`, `accelerate`, `sentencepiece`, `tqdm`). No manual setup is needed.

## Data

- **Training data:** the parallel `*-src-tgt/` folders (FCE, Lang-8, NUCLE, W&I+LOCNESS) are committed.
- **Test data:** `conll14st-test-data/` and `bea-2019-test-data/` are committed.
- **M2 files (only needed for `special-finetuning/`):** the weighted loss reads error categories from the BEA-2019 M2 files in `data/raw/`, which is gitignored. Unzip the committed archive:

  ```bash
  unzip data/raw.zip -d data/      # creates data/raw/*.m2
  ```

  If you cloned the repo somewhere other than the cluster, copy `data/raw/` over as well (for example with `./transfer_files.sh`).

## Running the fine-tuning

The first argument to `run.sh` selects the mode: `train`, `eval` (CoNLL-2014) or `bea` (BEA-2019). Any remaining arguments are passed to the Python script.
Always submit from inside the folder.

### Standard fine-tuning

```bash
cd standard-finetuning
sbatch run.sh train      # 1 epoch on all training data -> output/bart-base-gec/
sbatch run.sh eval       # CoNLL-2014 with M2Scorer     -> output/eval_conll14/scores.json
sbatch run.sh bea        # BEA-2019 corrections         -> output/eval_bea19/submission.zip
```

### Category-weighted fine-tuning

```bash
cd special-finetuning

# Frequency-aware weights (the default: alpha=3, beta=0.5, w_max=10)
sbatch run.sh train                                   # -> output/a3_b0.5_w10/bart-base-gec/
sbatch run.sh eval                                    # -> output/a3_b0.5_w10/eval_conll14/
sbatch run.sh bea                                     # -> output/a3_b0.5_w10/eval_bea19/

# Uniform weights (beta=0)
sbatch run.sh train --alpha 3 --beta 0 --w_max 10     # -> output/a3_b0_w10/bart-base-gec/
sbatch run.sh eval --run_dir output/a3_b0_w10
sbatch run.sh bea  --run_dir output/a3_b0_w10
```

Each set of hyperparameters gets its own run dir, `output/a{alpha}_b{beta}_w{w_max}/`.

### Outputs

- Slurm logs: `output/slurm-<job-name>-<jobid>.{out,err}`
- If a training job is interrupted, resubmit it. Training resumes from the latest checkpoint in `checkpoints/`.
- CoNLL-2014: `eval_conll14/scores.json` holds P / R / F0.5. Predictions and the M2Scorer verbose output are in the same folder. M2Scorer and a Python 2.7 interpreter are downloaded into `output/tools/` on first use.
- BEA-2019: the test references are not public. Upload `eval_bea19/submission.zip` to [Codabench](https://www.codabench.org/competitions/10960/) for scoring.

### Smoke test

Before a full run, check that the pipeline works end to end:

```bash
sbatch run.sh train --max_train_samples 64 --max_steps 2
sbatch run.sh eval  --limit 20
```

Smoke runs write to the same output folders as real runs, so delete them before starting a real run.

## Analysis

[`analysis/errant/`](analysis/errant/README.md) has the per-category ERRANT analysis of the three models on CoNLL-2014: per-category P / R / F0.5, bootstrap and randomisation significance tests, and qualitative disagreement examples.

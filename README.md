# BART-base grammatical error correction

Transfer this project to the SoC cluster, enter the project directory, and submit:

```bash
sbatch train_bart.sh
```

The job creates a Python 3.12 `.venv`, installs the packages, downloads
`facebook/bart-base`, fine-tunes on Lang-8, FCE, NUCLE, and W&I A/B/C, evaluates
once on `w&-src-tgt/ABCN.dev.gold.bea19.src` / `.tgt`, then saves the final weights.
There are no intermediate checkpoints or evaluations during training.

Transfer all four `*-src-tgt` directories and `data/raw.zip` along with
`train_bart.sh`, `train_bart.py`, and `requirements.txt`. The job extracts the gold
development M2 file from the archive if it is missing. Development data is never
included in training. Compute nodes need internet access for the initial package
and model downloads; no Hugging Face token is required for this public model.

The Slurm script defaults to `gpu-long`, one `a100-40` GPU, 4 CPUs, 32 GB RAM, and
48 hours. These are submission defaults, not a runtime estimate. Adjust the
`#SBATCH` lines if your cluster allocation uses different settings. Python 3.12
must be available as `python3.12`; change `PYTHON_BIN` in the script if its name
differs. If your cluster requires a Python module, add its `module load` command
before the venv setup.

Training settings are at the top of `train_bart.py`: 3 epochs, learning rate
`3e-5`, batch size 4, gradient accumulation 16 (effective batch size 64), and
maximum source/target length 512 tokens. Longer sentences are truncated.
Mixed precision is selected automatically for the assigned GPU. Unchanged
sentence pairs and empty corrected targets are retained.

The job log is `bart-base-<job-id>.log`. Results are in `outputs/bart-base-gec/`:

- `model.safetensors`: final model weights, with configuration and tokenizer files
  alongside it for loading with `from_pretrained`.
- `metrics.json`: training metrics, development loss, and training settings.
- `errant_scores.txt`: development precision, recall, and F0.5.
- `dev_predictions.txt` and `.m2`: generated corrections and their edit annotations.

ERRANT 3.0.2 is used consistently for this baseline. These are development scores;
ERRANT versions can produce different scores from the original BEA-2019 scorer.
See the [ERRANT documentation](https://github.com/chrisjbryant/errant) for details.

A new submission trains from BART-base again and replaces the previous results
in the output directory. An interrupted training job must restart because no
intermediate checkpoints are saved. If final evaluation fails, the completed
model is still saved and the job exits with an error.

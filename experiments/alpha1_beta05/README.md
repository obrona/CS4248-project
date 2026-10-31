# BART standard-loss control: alpha=1, beta=0.5

This experiment uses the tl2 category-weighting training pipeline with
alpha=1, beta=0.5 and w_max=10. All valid target-token weights are 1,
so beta has no effect on the loss in this experiment.

## Training

- Model: facebook/bart-base.
- Stage 1: 1 epoch on 1,112,307 sentence pairs, including Lang-8.
- Stage 2: continue the stage-1 model for 4 epochs on clean corpora,
  excluding Lang-8, at learning rate 1e-5.
- Both jobs completed successfully.
- Stage-1 job: 1048042; elapsed 33m24s.
- Stage-2 job: 1048578; elapsed 10m32s.
- Exact arguments, package versions and training histories are preserved
  in results/ and environment.yml.

Training M2 files were reconstructed from parallel sentence pairs using
SequenceMatcher and placeholder OTHER labels. Sentence reconstruction
was validated against tl2's pairs with zero target mismatches.
These annotations support this alpha=1 control; they must not be used
as grammatical-category annotations for category-weighted experiments.

## Evaluation results

| Dataset / scorer | Precision | Recall | F0.5 |
|---|---:|---:|---:|
| W&I+LOCNESS dev / ERRANT | 57.21% | 33.80% | 50.25% |
| CoNLL-2014 / official M2Scorer | 70.69% | 41.84% | 62.12% |
| BEA-2019 test / Codabench | Pending | Pending | Pending |

Dev: 4,384 sentences; TP=2522, FP=1886, FN=4939.
CoNLL-2014: 1,312 sentences.
BEA-2019: predictions generated for 4,477 sentences.
The Codabench score has not yet been recorded in this report.

Generation: 5 beams, input limit 256 tokens, max_new_tokens=256,
early_stopping=True, no_repeat_ngram_size=0, batch size 16.
Dev predictions were spaCy-tokenized before ERRANT annotation.
CoNLL uses tl2's evaluation script and the existing official Python-2
M2Scorer installation.

These results should be compared against models with matching training
stages, data and decoding settings. This is a standard-loss control;
the scores do not demonstrate a benefit from category weighting.

## Package contents

- source_snapshot/: actual training/evaluation code and submitted jobs.
- rebuild_alpha1_m2.py: reconstruct the alpha=1 training M2 inputs.
- results/: configs, histories, predictions and evaluation reports.
- logs/: training and evaluation logs.
- environment.yml: exported gec environment.
- source_commit.txt: source repository commit.
- training_m2_sha256.json: training M2 checksums.

## Reproduction

1. Clone https://github.com/obrona/CS4248-project and check out the
   commit in source_commit.txt.
2. Copy source_snapshot/special-finetuning/ into the checkout's
   special-finetuning/ directory.
3. Recreate the environment from environment.yml, adapting the name
   if needed. The supplied job scripts activate an environment named gec.
4. Run:
   python rebuild_alpha1_m2.py --repo /path/to/checkout
5. Adapt the home-directory paths and Slurm settings in the saved jobs.
6. Submit train_alpha1_checked.sbatch from special-finetuning/.
7. After stage 1 succeeds, submit train_alpha1_stage2.sbatch.
8. Submit the saved evaluation jobs afterward.

The M2 reconstruction uses the repository's sentence pairs, which were
checked to match the original cluster pairs before this experiment.

## Loading the shared trained models

Extract the separate model archive from the checkout's
special-finetuning/ directory:

    tar -xzf alpha1_beta05_models.tar.gz

Final model and tokenizer:

    output/a1_b0.5_w10/stage2/bart-base-gec/

Load using AutoTokenizer.from_pretrained and
BartForConditionalGeneration.from_pretrained with that path.

Model weights and optimizer checkpoints are not included in this
GitHub code package. The separate archive contains both exported
stage models and results, excluding intermediate checkpoints.
Training data is not included.

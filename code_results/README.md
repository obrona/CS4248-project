# Frequency weighting: Stage 1

- Alpha: 1; beta: 0.5.
- Original model: output/a1_b0.5_w10/bart-base-gec
- Generation: 5 beams, maximum length 256, batch size 16.
- Dev ERRANT P/R/F0.5: 0.5998 / 0.1422 / 0.3649.
- CoNLL-2014 M2 P/R/F0.5: 0.7422 / 0.2528 / 0.5350.
- BEA-2019 test span-level correction P/R/F0.5: 79.15% / 29.01% / 58.81%.
- Codabench submission: https://www.codabench.org/competitions/10960/detailed_results/1023976/
- Stage 1 evaluation scripts: source_snapshot/special-finetuning/eval_alpha1_stage1_*.sbatch
- Model and tokenizer are in the separate models archive.
- Source snapshot includes the original two-stage project code.

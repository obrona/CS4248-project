# Description

We will fine-tune a pretrained BART encoder-decoder transformer for grammatical error correction. Given a grammatically erroneous sentence as input, BART will generate the corrected sentence. We will first implement standard BART fine-tuning using token-level cross-entropy loss as our baseline.

Our proposed extension is a grammar-error-category-aware weighted loss. Standard sequence-to-sequence training assigns equal weight to all target tokens, even though most tokens in GEC training examples remain unchanged. ERRANT annotations are used to identify and categorise grammatical edits. Instead of assigning the same weight to every target token, tokens corresponding to grammatical corrections will receive weights based on their error category. This prevents outlier categories and unchanged tokens from affecting training.

We will compare the category-weighted model against standard BART fine-tuning using precision, recall and F0.5. ERRANT error-type results will determine whether weighting improves correction of underrepresented grammatical error categories and whether this affects the overall precision-recall trade-off.

## Additional and publicly available training data

We will use the provided English GEC corpora, FCE, Lang-8, NUCLE, and W&I+LOCNESS. We will use the W&I+LOCNESS development set for model development and established GEC test sets for evaluation.

## External supporting code

PyTorch and Hugging Face transformers/datasets for BART fine-tuning and inference. ERRANT will be used to extract and classify grammatical edits and evaluate performance. The provided BEA-2019 M2 data will be used to obtain erroneous–corrected sentence pairs for BART training.

## Uniqueness and comparison with prior methods

Previous work has demonstrated that pretrained BART can be directly fine-tuned as a strong baseline for GEC. Other GEC work has also explored edit-weighted and data-weighted training, motivated by the fact that only a small proportion of tokens in GEC sentences require correction.

Our proposed method extends this idea by making the loss aware of the grammatical category of each edit rather than treating all corrections equally. We will use the distribution of ERRANT error categories in the training data to assign greater training emphasis to underrepresented grammatical errors while retaining standard BART generation.

The method therefore investigates whether a lightweight modification to BART's training objective can improve learning across grammatical error categories. We will perform an ablation comparing standard BART, uniform edit weighting, and category-aware edit weighting. ERRANT category-level evaluation will determine if any overall improvement is associated with better performance on underrepresented error types.
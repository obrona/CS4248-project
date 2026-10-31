"""Stage-2 fine-tuning: continue from the stage-1 model on the clean data only.

Starts from output/bart-base-gec (written by finetune.py) and trains on the
annotator-corrected corpora (FCE train+dev, NUCLE, W&I A/B/C), dropping Lang-8,
whose volunteer corrections are noisy. Same loss as stage 1, lower learning rate.
The model is saved to output/stage2/bart-base-gec.

Usage:
    python finetune-2.py                     # full run
    python finetune-2.py --lr 2e-5 --epochs 1
    python finetune-2.py --max_train_samples 64 --max_steps 2   # smoke test

Evaluate with:
    python eval-conll14st.py --model_dir output/stage2/bart-base-gec --eval_dir output/stage2/eval_conll14
"""

import sys
from pathlib import Path

from finetune import OUTPUT_DIR, TRAIN_FILES, build_parser, main

CLEAN_TRAIN_FILES = [(subdir, stem) for subdir, stem in TRAIN_FILES if not stem.startswith("lang8")]


def parse_args():
    p = build_parser()
    p.description = __doc__
    p.set_defaults(
        model_name=str(OUTPUT_DIR / "bart-base-gec"),  # the stage-1 model
        output_dir=OUTPUT_DIR / "stage2",
        lr=1e-5,
        epochs=4.0,
        warmup_ratio=0.05,
        save_steps=500,
        logging_steps=50,
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.model_name).is_dir():
        sys.exit(f"Stage-1 model not found at {args.model_name}; run finetune.py first or pass --model_name.")
    main(args, CLEAN_TRAIN_FILES)

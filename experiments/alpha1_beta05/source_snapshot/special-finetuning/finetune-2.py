"""Stage-2 category-weighted fine-tuning: continue from a stage-1 run on the clean data only.

Starts from <run_dir>/bart-base-gec (written by finetune.py) and trains on the
annotator-corrected corpora (FCE train+dev, NUCLE, W&I A/B/C), dropping Lang-8,
whose volunteer corrections are noisy. Same weighted loss as stage 1, lower learning
rate. alpha/beta/w_max default to the stage-1 run's values (from its train_config.json);
the category weights are recomputed from the category counts of the clean data.
The model is saved to <run_dir>/stage2/bart-base-gec.

Usage:
    python finetune-2.py                                # continue output/a3_b0.5_w10
    python finetune-2.py --run_dir output/a2_b1_w10
    python finetune-2.py --max_train_samples 64 --max_steps 2   # smoke test

Evaluate with:
    python eval-conll14st.py --run_dir output/a3_b0.5_w10/stage2
"""

import json
import sys
from pathlib import Path

from finetune import OUTPUT_DIR, TRAIN_FILES, build_parser, main

CLEAN_TRAIN_FILES = [f for f in TRAIN_FILES if not f[0].startswith("lang8")]


def parse_args():
    p = build_parser()
    p.description = __doc__
    p.add_argument("--run_dir", type=Path, default=OUTPUT_DIR / "a3_b0.5_w10", help="stage-1 run to continue from")
    run_dir = p.parse_known_args()[0].run_dir
    stage1_config = run_dir / "train_config.json"
    if stage1_config.is_file():
        with open(stage1_config) as f:
            stage1_args = json.load(f)["args"]
        p.set_defaults(**{k: stage1_args[k] for k in ("alpha", "beta", "w_max")})
    p.set_defaults(
        model_name=str(run_dir / "bart-base-gec"),  # the stage-1 model
        output_dir=run_dir / "stage2",
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
        sys.exit(f"Stage-1 model not found at {args.model_name}; run finetune.py first or pass --run_dir.")
    main(args, CLEAN_TRAIN_FILES)

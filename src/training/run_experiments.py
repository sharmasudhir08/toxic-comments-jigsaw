"""
Run BERT / DistilBERT / RoBERTa on Jigsaw (multi-label) for 5 seeds each.

    python -m src.run_experiments --smoke        # tiny end-to-end check (minutes, CPU ok)
    python -m src.run_experiments                # the real thing: 3 models x 5 seeds
    python -m src.run_experiments --models distilbert --seeds 42 43

Finished runs are skipped, so if it stops (power cut, Ctrl+C) just run the same
command again and it continues where it left off.
"""

import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import SEEDS
from src.model_registry import MODELS
from src.training.train import run_one


def main():
    ap = argparse.ArgumentParser(description="Multi-label toxic comment experiments")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default=None, help="default: results (results_smoke with --smoke)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=2, help="early-stopping patience (epochs)")
    ap.add_argument("--max_train_samples", type=int, default=None,
                    help="use only N training comments (faster; for CPU / quick tests)")
    ap.add_argument("--max_test_samples", type=int, default=None)
    ap.add_argument("--save_model", action="store_true",
                    help="keep the best run of each model in models/ (needed for predict.py / app.py)")
    ap.add_argument("--overwrite", action="store_true", help="redo runs that already finished")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny test: 1500 train / 500 test comments, 1 epoch, 2 seeds")
    ap.add_argument("--protocol", choices=["standard", "common"], default="standard",
                    help="common = preregistration v4 protocol (light cleaning, weighted loss, macro-F1 selection, "
                         "patience 3, tuned thresholds); writes to results_proposed/ by default")
    a = ap.parse_args()
    if a.protocol == "common":
        if a.patience == 2:
            a.patience = 3
        if a.save_model:
            print("Note: --save_model is ignored with --protocol common (models/ is never touched).")

    if a.smoke:
        a.max_train_samples, a.max_test_samples = a.max_train_samples or 1500, a.max_test_samples or 500
        a.epochs, a.batch_size, a.max_len = 1, 16, 64
        if a.seeds == SEEDS:
            a.seeds = SEEDS[:2]
    if a.protocol == "common":
        out_dir = a.out_dir or ("results_proposed_smoke" if a.smoke else "results_proposed")
    else:
        out_dir = a.out_dir or ("results_smoke" if a.smoke else "results")

    has_gpu = torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    print(f"Device: {'GPU' if has_gpu else 'CPU'}   models={a.models}   seeds={a.seeds}   out={out_dir}")
    if not has_gpu and not a.max_train_samples:
        print("WARNING: no GPU found. Full-data fine-tuning on CPU will take a very long time. "
              "Consider  --max_train_samples 20000  (see README).")

    # seed-major order: after each seed, every model has one more finished run
    for seed in a.seeds:
        for key in a.models:
            run_one(key, seed, a.data_dir, out_dir, a.epochs, a.batch_size, a.lr, a.max_len,
                    a.val_size, a.patience, a.max_train_samples, a.max_test_samples,
                    a.save_model, a.overwrite, a.protocol)

    print("\nAll runs finished. Now run:  python generate_results.py"
          + (f" --results_dir {out_dir}" if out_dir != "results" else ""))


if __name__ == "__main__":
    main()

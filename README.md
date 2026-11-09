# How Reliable Are Architecture Rankings?

Code and result CSVs for a multi-seed comparison of three recurrent architectures (LSTM, Bi-LSTM,
GRU) and three pretrained transformers (BERT, DistilBERT, RoBERTa) on multi-label toxic comment
classification, three random seeds per model, on the official Jigsaw Toxic Comment Classification
Challenge dataset. Licensed under the Apache License 2.0.

## Purpose

This repository lets a reviewer verify every number reported in the paper by re-running the exact
statistical test on the authors' saved model outputs (`src/analysis/`). The training code
(`src/training/`) is included for completeness and for anyone building on this work later, but
**verifying a reported number does not require retraining anything** — every analysis script reads
already-computed results.

## What you need to verify a number (no training required)

The saved per-model, per-seed test-set outputs (`metrics.json` and `test_outputs.npz` for each of
the 18 runs: 6 models x 3 seeds) and, for the near-duplicate screen, a precomputed similarity mask
against the training set. **These are not included in this repository** and are available from the
authors on reasonable request. Place them under `results_proposed/runs/<model>_p4/seed_<seed>/` in
the same layout the scripts expect (see each script's own docstring for the exact path), then skip
straight to "Reproducing a number" below.

## Dataset (needed only to retrain, not to verify)

Not included. Download the official Jigsaw Toxic Comment Classification Challenge dataset from
Kaggle (`train.csv`, `test.csv`, `test_labels.csv`) and place it in a `data/` folder at the
project root:

<https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data>

For the recurrent models, also download the FastText `crawl-300d-2M` word vectors (about 1.4 GB)
into `data/embeddings/`:

```
mkdir -p data/embeddings
curl -L -o data/embeddings/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip data/embeddings/crawl-300d-2M.vec.zip -d data/embeddings/
```

## Environment

```
pip install -r requirements.txt
```

Training was run on a MacBook, Apple M2, 16 GB unified memory. The three transformer models used
the integrated GPU through PyTorch's Metal (`mps`) backend; the three recurrent models used CPU.
Exact package versions are pinned in `requirements.txt` (PyTorch 2.14.1, Transformers 5.21.0 —
confirmed from every saved run's own recorded configuration). Verifying a number from saved
outputs only needs the statistical-analysis packages, not PyTorch/Transformers.

## Reproducing a number

1. Find the item in `results/MAPPING.csv` (paper table/section -> result file -> source CSV ->
   script).
2. Run the matching script from this repository's root, against the saved outputs above:
   - `generate_results.py --results_dir results_proposed` for Tables 1 and 3.
   - `python -m src.analysis.<script_name>` for everything else (see `TRACEABILITY.md` for the
     exact script per table).
3. Compare the output CSV against the matching file in `results/` — they should match exactly.

## Retraining (optional, not needed to verify any reported number)

Each command below trains one model for the three primary seeds (42, 43, 44). A finished run (one
that already has a `metrics.json`) is skipped automatically. Approximate wall-clock time per run,
measured on the hardware above:

| Model | Command | Time per run (3 seeds total) |
|---|---|---|
| LSTM | `python -m src.training.rnn_proposed` once, then `python -m src.training.train_proposed_lstm --seeds 42 43 44` | ~23-36 min each |
| Bi-LSTM | `python -m src.training.train_proposed_bilstm --seeds 42 43 44` | ~37-66 min each |
| GRU | `python -m src.training.train_proposed_gru --seeds 42 43 44` | ~24-34 min each |
| BERT | `python -m src.training.run_experiments --protocol common --epochs 4 --patience 1 --models bert --seeds 42 43 44` | ~9.3-9.5 h each |
| DistilBERT | `python -m src.training.run_experiments --protocol common --epochs 4 --patience 1 --models distilbert --seeds 42 43 44` | ~4.75-4.8 h each |
| RoBERTa | `python -m src.training.run_experiments --protocol common --epochs 4 --patience 1 --models roberta --seeds 42 43 44` | ~4.9-9.6 h each |

`python -m src.training.rnn_proposed` (no arguments) must be run once before any recurrent model,
to build the shared preprocessing cache and fine-tuned FastText embedding table.

## Layout

- `src/analysis/` — every statistical-test script behind a reported number (nested ANOVA, Tukey
  HSD + Holm, DeLong, paired bootstrap, paired-by-seed tests, the benchmark comparison with
  within-family Holm correction, McNemar + Fisher and the permutation robustness check, the
  per-length-bin tests with Holm over 10, calibration/BSS, seed-sensitivity CV ratios, the
  truncation check, and the near-duplicate screen).
- `src/training/` — the six models' training scripts and the shared training loop, plus
  `rnn_preprocessing.py` (tokeniser/embedding-cache builder) and `rnn_train.py` (recurrent
  training loop, imported by the per-model wrapper scripts).
- `src/` (root) — shared utility modules imported by the scripts above: `config.py`, `data.py`,
  `preprocessing.py`, `metrics.py`, `model_registry.py`, `maslej_baselines.py`,
  `paper_baselines.py`.
- `generate_results.py` — overall-performance and per-label tables from saved per-run metrics.
- `results/` — every CSV behind the paper's tables, named by final table number, plus
  `MAPPING.csv`.
- `figures/` — the six figures exactly as they appear in the final manuscript.
- `TRACEABILITY.md` — table/figure number -> result file -> script -> training run.

## Not included

The dataset, the FastText embedding file, model checkpoints, saved predictions, and the manuscript
file itself. Figure-generation and manuscript-preparation scripts are intentionally excluded —
this package contains training and statistical-analysis code only.

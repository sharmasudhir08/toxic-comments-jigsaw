"""
Re-checks H3a/H3b against the CORRECT tuned-threshold Maslej comparator (0.592 recurrent, 0.621
transformer), resolving the threshold-mismatch flagged 3 Nov 2026: the original comparison used our
own tuned F1(M) against Maslej's UNTUNED 0.56/0.55 (Table 4/9, threshold 0.5). The 0.592/0.621 values
are mechanically recomputed from Maslej's own published tuned per-class P/R (Tables 17-18, Sec. 5's
own macro-F1 formula) -- see reference.md Sec. A2.1 for the full derivation. Not a new experiment:
reads the same saved metrics.json files the original H3a/H3b script used.

Usage: python -m src.h3_h4_vs_maslej_tuned_comparator
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

SEEDS = [42, 43, 44]
MODELS = {"lstm": ("LSTM", "recurrent", "lstm_p4"), "bilstm": ("Bi-LSTM", "recurrent", "bilstm_p4"),
          "gru": ("GRU", "recurrent", "gru_p4"), "bert": ("BERT", "transformer", "bert_p4"),
          "distilbert": ("DistilBERT", "transformer", "distilbert_p4"),
          "roberta": ("RoBERTa", "transformer", "roberta_p4")}
# Tuned-threshold Maslej comparators, computed from Tables 17-18 (reference.md Sec. A2.1), not the
# original untuned 0.56/0.55 (Table 4/9).
BENCHMARK_TUNED = {"recurrent": 0.592, "transformer": 0.621}


def holm(pvals):
    idx = np.argsort(pvals)
    n = len(pvals)
    adj = [0.0] * n
    running = 0.0
    for rank, i in enumerate(idx):
        c = pvals[i] * (n - rank)
        running = max(running, c)
        adj[i] = min(running, 1.0)
    return adj


def main():
    rows = []
    for key, (label, family, folder) in MODELS.items():
        vals = np.array([json.load(open(f"results_proposed/runs/{folder}/seed_{s}/metrics.json"))["test"]["f1_macro_m"]
                          for s in SEEDS])
        target = BENCHMARK_TUNED[family]
        diff = vals.mean() - target
        t_stat, p_two = stats.ttest_1samp(vals, target)
        p_one = p_two / 2 if diff > 0 else 1 - p_two / 2
        rows.append({"model": label, "family": family, "mean_f1_macro_m": vals.mean(),
                     "benchmark_tuned": target, "diff": diff, "seeds_above": f"{(vals > target).sum()}/3",
                     "t_stat": t_stat, "df": 2, "p_one_sided": p_one})

    df = pd.DataFrame(rows)
    df["p_holm_6models"] = holm(df["p_one_sided"].tolist())
    df["sig_0.05"] = df["p_holm_6models"] < 0.05
    print(df.round(5).to_string(index=False))
    df.round(5).to_csv("results_proposed/report/table_h3_vs_maslej_tuned_comparator.csv", index=False)
    print("\nSaved: table_h3_vs_maslej_tuned_comparator.csv")


if __name__ == "__main__":
    main()

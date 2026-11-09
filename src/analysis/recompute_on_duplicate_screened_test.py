"""
H7 (preregistration.md): recompute headline metrics on the test set with near-exact/near-duplicate
train-test comments removed (TF-IDF cosine >= 0.9, src/duplicate_audit.py), vs. the full
63,978-comment official test set that every saved result in this project actually used.

Unlike a project with an additional short-comment screen, this project's test_outputs.npz already
holds all 63,978 rows in exactly the row order of data/test.csv (src/data.py's load_official_test,
order-preserving, no filter beyond excluding unscored -1 rows) -- so the duplicate mask aligns
directly, no row-order reconstruction needed.

No retraining, no new predictions: re-aggregates the already-saved test_outputs.npz arrays after
masking out near-duplicate rows.

Usage: python -m src.recompute_on_duplicate_screened_test
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from src.data import load_official_test, LABELS

MODELS = ["lstm", "bilstm", "gru", "bert", "distilbert", "roberta"]
SEEDS = [42, 43, 44]  # primary-analysis seeds only
RESULTS_DIR = "results_proposed/runs"
MASLEJ = {"lstm": 0.56, "bilstm": 0.56, "gru": 0.56, "bert": 0.55, "distilbert": 0.55, "roberta": 0.55}


def build_dup_mask():
    test_df = load_official_test("data")
    test_ids = test_df["id"].values
    d = np.load("data/duplicate_screen.npz", allow_pickle=True)
    id_to_dup = dict(zip(d["test_id"], d["is_near_duplicate"]))
    missing = [i for i in test_ids[:5] if i not in id_to_dup]
    if missing:
        raise RuntimeError(f"ID alignment failed, sample missing ids: {missing}")
    mask = np.array([bool(id_to_dup[i]) for i in test_ids])
    return test_ids, mask


def macro_f1_m(y, p, threshold=0.5):
    y_pred = (p >= threshold).astype(int)
    precs = [precision_score(y[:, j], y_pred[:, j], zero_division=0) for j in range(y.shape[1])]
    recs = [recall_score(y[:, j], y_pred[:, j], zero_division=0) for j in range(y.shape[1])]
    P, R = np.mean(precs), np.mean(recs)
    return 2 * P * R / (P + R) if (P + R) > 0 else 0.0


def macro_auc(y, p):
    aucs = []
    for j in range(y.shape[1]):
        if y[:, j].sum() == 0 or y[:, j].sum() == len(y):
            continue
        aucs.append(roc_auc_score(y[:, j], p[:, j]))
    return float(np.mean(aucs))


def ece_label(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == m].mean() - p[idx == m].mean()) * (idx == m).mean()
                      for m in range(bins) if (idx == m).any()))


def macro_ece(y, p):
    return float(np.mean([ece_label(y[:, j], p[:, j]) for j in range(y.shape[1])]))


def load_thresholds(model, seed):
    with open(os.path.join(RESULTS_DIR, f"{model}_p4", f"seed_{seed}", "metrics.json"), encoding="utf-8") as f:
        m = json.load(f)
    thr = m.get("thresholds")
    if thr is None:
        return 0.5
    if isinstance(thr, dict):
        return np.array([thr[label] for label in LABELS])
    return np.array(thr)


def main():
    print("=" * 70)
    print("H7: recompute on the near-duplicate-screened test set vs. the full 63,978-comment set")
    print("=" * 70)

    test_ids, dup_mask = build_dup_mask()
    print(f"\nTest set: {len(test_ids)} rows (expect 63,978)")
    print(f"Near-duplicate rows (cosine >= 0.9): {int(dup_mask.sum())} (expect 2,029)")
    keep_mask = ~dup_mask
    screened_n = int(keep_mask.sum())
    print(f"Duplicate-screened row count: {screened_n} (expect {63978 - 2029} = 61,949)")

    rows = []
    for m in MODELS:
        for s in SEEDS:
            d = np.load(os.path.join(RESULTS_DIR, f"{m}_p4", f"seed_{s}", "test_outputs.npz"))
            probs, y = d["probs"], d["y_true"]
            assert len(probs) == len(test_ids), f"{m} seed {s}: row mismatch {len(probs)} vs {len(test_ids)}"
            thr = load_thresholds(m, s)

            f1m_before = macro_f1_m(y, probs, thr)
            auc_before = macro_auc(y, probs)
            ece_before = macro_ece(y, probs)

            probs_s, y_s = probs[keep_mask], y[keep_mask]
            f1m_after = macro_f1_m(y_s, probs_s, thr)
            auc_after = macro_auc(y_s, probs_s)
            ece_after = macro_ece(y_s, probs_s)

            rows.append({
                "model": m, "seed": s,
                "f1_macro_m_full": f1m_before, "f1_macro_m_screened": f1m_after,
                "roc_auc_macro_full": auc_before, "roc_auc_macro_screened": auc_after,
                "ece_macro_full": ece_before, "ece_macro_screened": ece_after,
            })
    df = pd.DataFrame(rows)
    os.makedirs("results_proposed/report", exist_ok=True)
    df.round(6).to_csv("results_proposed/report/table_h7_dup_screened_per_seed.csv", index=False)

    print("\n=== Per-model mean over 3 seeds: full (63,978) vs duplicate-screened ===")
    summary_rows = []
    for m in MODELS:
        sub = df[df["model"] == m]
        row = {"model": m}
        for metric in ["f1_macro_m", "roc_auc_macro", "ece_macro"]:
            b = sub[f"{metric}_full"].mean()
            a = sub[f"{metric}_screened"].mean()
            row[f"{metric}_full"] = b
            row[f"{metric}_screened"] = a
            row[f"{metric}_diff"] = a - b
        summary_rows.append(row)
        print(f"  {m:12s} F1(M) {row['f1_macro_m_full']:.4f}->{row['f1_macro_m_screened']:.4f} "
              f"({row['f1_macro_m_diff']:+.4f})  "
              f"AUC {row['roc_auc_macro_full']:.4f}->{row['roc_auc_macro_screened']:.4f} "
              f"({row['roc_auc_macro_diff']:+.4f})  "
              f"ECE {row['ece_macro_full']:.4f}->{row['ece_macro_screened']:.4f} "
              f"({row['ece_macro_diff']:+.4f})")
    summary = pd.DataFrame(summary_rows)
    summary.round(6).to_csv("results_proposed/report/table_h7_dup_screened_summary.csv", index=False)

    print("\n=== H3a/H3b retest: macro-F1(M) vs Maslej-Kresnakova benchmark, on screened set ===")
    h3_rows, pvals = [], []
    for m in MODELS:
        sub = df[df["model"] == m]
        vals = sub["f1_macro_m_screened"].values
        diff = vals.mean() - MASLEJ[m]
        t, p_two = stats.ttest_1samp(vals, MASLEJ[m])
        p_one = p_two / 2 if t > 0 else 1 - p_two / 2
        pvals.append(p_one)
        h3_rows.append({"model": m, "mean_f1_macro_m_screened": vals.mean(), "benchmark": MASLEJ[m],
                        "diff": diff, "seeds_above": int((vals > MASLEJ[m]).sum()), "p_one_sided": p_one})
    h3 = pd.DataFrame(h3_rows)
    order = np.argsort(pvals)
    adj = [0.0] * len(MODELS)
    running = 0.0
    for rank, i in enumerate(order):
        c = pvals[i] * (len(MODELS) - rank)
        running = max(running, c)
        adj[i] = min(running, 1.0)
    h3["p_holm_6models"] = adj
    h3["sig_0.05"] = h3["p_holm_6models"] < 0.05
    print(h3.to_string(index=False))
    h3.round(6).to_csv("results_proposed/report/table_h7_dup_screened_h3_retest.csv", index=False)

    print(f"\nPer-seed table: results_proposed/report/table_h7_dup_screened_per_seed.csv")
    print(f"Summary table: results_proposed/report/table_h7_dup_screened_summary.csv")
    print(f"H3a/H3b retest: results_proposed/report/table_h7_dup_screened_h3_retest.csv")


if __name__ == "__main__":
    main()

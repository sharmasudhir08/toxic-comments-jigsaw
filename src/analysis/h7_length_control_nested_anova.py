"""
H7 (preregistration.md): formal per-bin repeat of H1's nested-ANOVA family test, to test whether
"the family conclusion of H1 holds under length control (within length bins)". Previously this was
only checked directionally (raw per-model means, src/length_robustness.py); this script runs the
actual statistical test, bin by bin.

Reuses the exact nested-ANOVA method from src/family_comparison.py (H1's primary test) and the
exact per-bin prediction-loading method from src/length_robustness.py. No new training or
inference -- reads already-saved results_proposed/runs/<model>_p3/seed_<seed>/test_outputs.npz.

Usage: python -m src.h7_length_control_nested_anova
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from src.data import load_official_test

MODELS = ["lstm", "bilstm", "gru", "bert", "distilbert", "roberta"]
RECURRENT = ["lstm", "bilstm", "gru"]
TRANSFORMER = ["bert", "distilbert", "roberta"]
SEEDS = [42, 43, 44]
RESULTS_DIR = "results_proposed/runs"

BIN_EDGES = [26, 51, 101, 201]
BIN_NAMES = ["<=25", "26-50", "51-100", "101-200", ">200"]


def ece_score(probs, y_true, n_bins=10):
    label_eces = []
    for li in range(probs.shape[1]):
        p, y = probs[:, li], y_true[:, li]
        edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for b in range(n_bins):
            mask = (p >= edges[b]) & (p < edges[b + 1] if b < n_bins - 1 else p <= edges[b + 1])
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / len(p)) * abs(y[mask].mean() - p[mask].mean())
        label_eces.append(ece)
    return float(np.mean(label_eces))


def macro_f1_m(y, preds):
    precs, recs = [], []
    for li in range(y.shape[1]):
        yt, yp = y[:, li], preds[:, li]
        if yt.sum() == 0 and yp.sum() == 0:
            continue
        precs.append(precision_score(yt, yp, zero_division=0))
        recs.append(recall_score(yt, yp, zero_division=0))
    P, R = np.mean(precs), np.mean(recs)
    return 2 * P * R / (P + R) if (P + R) > 0 else 0.0


def macro_auc(y, probs):
    aucs = []
    for li in range(y.shape[1]):
        yt = y[:, li]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, probs[:, li]))
    return float(np.mean(aucs)) if aucs else float("nan")


def nested_anova(long_df):
    """Same formula as src/family_comparison.py's nested_anova -- F(1,4), family vs model(family)."""
    d = long_df
    grand_mean = d["value"].mean()
    n_total = len(d)
    fam_means = d.groupby("family")["value"].mean()
    model_means = d.groupby(["family", "model"])["value"].mean()
    n_seeds_per_model = d.groupby("model").size().iloc[0]
    n_models_per_family = d.groupby("family")["model"].nunique()
    n_families = fam_means.shape[0]
    n_models = model_means.shape[0]

    ss_family = sum(n_models_per_family[fam] * n_seeds_per_model * (fmean - grand_mean) ** 2
                     for fam, fmean in fam_means.items())
    df_family = n_families - 1
    ss_model_in_family = sum(n_seeds_per_model * (mmean - fam_means[fam]) ** 2
                              for (fam, model), mmean in model_means.items())
    df_model_in_family = n_models - n_families

    ms_family = ss_family / df_family
    ms_model_in_family = ss_model_in_family / df_model_in_family
    f_family = ms_family / ms_model_in_family if ms_model_in_family > 0 else np.nan
    p_family = float(stats.f.sf(f_family, df_family, df_model_in_family)) if not np.isnan(f_family) else np.nan
    return f_family, p_family


def main():
    test_df = load_official_test("data")
    lengths = test_df["comment_text"].str.split().str.len().values
    bin_idx = np.digitize(lengths, BIN_EDGES)

    print("=" * 70)
    print("H7: per-bin repeat of H1's nested-ANOVA family test")
    print("=" * 70)

    rows = []
    for b, bin_name in enumerate(BIN_NAMES):
        mask = bin_idx == b
        n_in_bin = mask.sum()
        print(f"\n--- Bin {bin_name} (n={n_in_bin} comments) ---")

        long_rows = {"f1_macro_m": [], "roc_auc_macro": [], "ece_macro": []}
        for m in MODELS:
            fam = "recurrent" if m in RECURRENT else "transformer"
            for s in SEEDS:
                d = np.load(os.path.join(RESULTS_DIR, f"{m}_p3", f"seed_{s}", "test_outputs.npz"))
                probs, y = d["probs"][mask], d["y_true"][mask]
                preds = (probs >= 0.5).astype(int)
                f1m = macro_f1_m(y, preds)
                auc = macro_auc(y, probs)
                ece = ece_score(probs, y)
                for metric, val in [("f1_macro_m", f1m), ("roc_auc_macro", auc), ("ece_macro", ece)]:
                    long_rows[metric].append({"family": fam, "model": m, "seed": s, "value": val})

        for metric in ["f1_macro_m", "roc_auc_macro", "ece_macro"]:
            long_df = pd.DataFrame(long_rows[metric])
            f_stat, p_val = nested_anova(long_df)
            rec_mean = long_df[long_df["family"] == "recurrent"]["value"].mean()
            trans_mean = long_df[long_df["family"] == "transformer"]["value"].mean()
            print(f"  {metric:15s} recurrent={rec_mean:.4f}  transformer={trans_mean:.4f}  "
                  f"F(1,4)={f_stat:.4f}  p={p_val:.5f}")
            rows.append({"bin": bin_name, "n_comments": int(n_in_bin), "metric": metric,
                        "recurrent_mean": rec_mean, "transformer_mean": trans_mean,
                        "diff": trans_mean - rec_mean, "F_family_1_4": f_stat, "p_family_1_4": p_val})

    df = pd.DataFrame(rows)
    os.makedirs("results_proposed/report", exist_ok=True)
    out_path = "results_proposed/report/table_h7_length_control_nested_anova.csv"
    df.round(5).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print("\n=== Does the H1 family conclusion hold in every bin? ===")
    for metric in ["f1_macro_m", "roc_auc_macro"]:
        sub = df[df["metric"] == metric]
        n_sig = (sub["p_family_1_4"] < 0.05).sum()
        n_right_direction = (sub["diff"] > 0).sum()
        print(f"  {metric}: transformer > recurrent in {n_right_direction}/5 bins, "
              f"significant (p<0.05) in {n_sig}/5 bins")


if __name__ == "__main__":
    main()

"""
H2, full pre-registered methodology (preregistration.md H2): one-way ANOVA + Tukey HSD,
DeLong's test per label (median p over seeds), and paired bootstrap over test comments --
all Holm-corrected over the 9 cross-family pairs.

Written independently for this project (same general statistical methods as are standard for
this kind of pairwise model comparison, not copied from any other project's code). Reads only
already-saved results_proposed/runs/<model>_p4/seed_<seed>/{metrics.json,test_outputs.npz};
no new training.

Usage: python -m src.h2_full_methods
"""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

RECURRENT = ["lstm", "bilstm", "gru"]
TRANSFORMER = ["bert", "distilbert", "roberta"]
MODELS = RECURRENT + TRANSFORMER
SEEDS = [42, 43, 44]  # primary-analysis seeds only -- recurrent 45-46 are supplementary, excluded here
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
RESULTS_DIR = "results_proposed/runs"
PAIRS = [(r, t) for r in RECURRENT for t in TRANSFORMER]


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


def load_metrics(model, seed):
    with open(os.path.join(RESULTS_DIR, f"{model}_p4", f"seed_{seed}", "metrics.json"), encoding="utf-8") as f:
        return json.load(f)["test"]


def load_outputs(model, seed):
    d = np.load(os.path.join(RESULTS_DIR, f"{model}_p4", f"seed_{seed}", "test_outputs.npz"))
    return d["probs"], d["y_true"]


# ---------------- 1. One-way ANOVA + Tukey HSD ----------------
def tukey_hsd(metric):
    labels, values = [], []
    for m in MODELS:
        for s in SEEDS:
            labels.append(m)
            values.append(load_metrics(m, s)[metric])
    f_stat, anova_p = stats.f_oneway(*[
        [load_metrics(m, s)[metric] for s in SEEDS] for m in MODELS
    ])
    res = pairwise_tukeyhsd(np.array(values), np.array(labels), alpha=0.05)
    df = pd.DataFrame(res._results_table.data[1:], columns=res._results_table.data[0])
    sub = df[df.apply(lambda r: (r["group1"], r["group2"]) in PAIRS or (r["group2"], r["group1"]) in PAIRS, axis=1)]
    return anova_p, sub.copy()


# ---------------- 2. DeLong's test (per label, median p over seeds) ----------------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong_test_buggy(probs_a, probs_b, y_true):
    """ORIGINAL implementation -- kept for the audit trail only. DO NOT USE.

    Confirmed buggy 3 Nov 2026 (checks 1a/1b/1c): `tz` is computed on `preds[r]` in its
    original, unsorted-by-label order, but the downstream aucs/v01/v10 formulas (the standard
    "fast DeLong" algorithm, Sun & Xu 2014) require `tz` to be computed on predictions already
    grouped with all positive-class examples first, then all negative-class examples. That
    grouping step is missing here, so the AUC and variance estimate are silently corrupted
    whenever the input data is not already sorted by label (i.e. always, in practice). Verified:
    this function's own internal AUC did not match sklearn's `roc_auc_score` on the same data,
    including sign flips on 2 of 6 labels (LSTM vs BERT, seed 42).
    """
    pos = y_true == 1
    neg = y_true == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan, np.nan, np.nan
    preds = np.vstack([probs_a, probs_b])
    m, n = pos.sum(), neg.sum()
    k = 2
    pos_preds = preds[:, pos]
    neg_preds = preds[:, neg]
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _compute_midrank(pos_preds[r])
        ty[r] = _compute_midrank(neg_preds[r])
        tz[r] = _compute_midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    ld = np.array([[1, -1]])
    var = ld.dot(delongcov).dot(ld.T)
    auc_diff = aucs[0] - aucs[1]
    if var[0, 0] <= 0:
        return auc_diff, np.nan, np.nan
    z = auc_diff / np.sqrt(var[0, 0])
    p = 2 * stats.norm.sf(abs(z))
    return auc_diff, z, p


def delong_test(probs_a, probs_b, y_true):
    """Two-sided DeLong test for two correlated ROC-AUC estimates on the same binary labels.

    FIXED 3 Nov 2026 (Category A technical fix, approved by the author): predictions are now
    explicitly grouped with positive-class examples first, then negative-class examples, before
    the combined midrank `tz` is computed -- required by the aucs/v01/v10 formulas below (the
    standard "fast DeLong" algorithm, Sun & Xu 2014). See `delong_test_buggy` above for the
    original implementation and the bug description. Pre-registered aggregation rules (median
    across labels, then seeds; Holm correction) are unchanged by this fix.
    """
    pos = y_true == 1
    neg = y_true == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan, np.nan, np.nan
    preds = np.vstack([probs_a, probs_b])
    m, n = pos.sum(), neg.sum()
    k = 2
    pos_preds = preds[:, pos]
    neg_preds = preds[:, neg]
    grouped_preds = np.concatenate([pos_preds, neg_preds], axis=1)  # positives first, then negatives
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _compute_midrank(pos_preds[r])
        ty[r] = _compute_midrank(neg_preds[r])
        tz[r] = _compute_midrank(grouped_preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    ld = np.array([[1, -1]])
    var = ld.dot(delongcov).dot(ld.T)
    auc_diff = aucs[0] - aucs[1]
    if var[0, 0] <= 0:
        return auc_diff, np.nan, np.nan
    z = auc_diff / np.sqrt(var[0, 0])
    p = 2 * stats.norm.sf(abs(z))
    return auc_diff, z, p


def delong_per_pair():
    rows = []
    for rec, trans in PAIRS:
        for label_idx, label in enumerate(LABELS):
            seed_ps = []
            for s in SEEDS:
                probs_r, y_r = load_outputs(rec, s)
                probs_t, y_t = load_outputs(trans, s)
                assert np.array_equal(y_r, y_t)
                _, _, p = delong_test(probs_t[:, label_idx], probs_r[:, label_idx], y_r[:, label_idx])
                seed_ps.append(p)
            median_p = float(np.nanmedian(seed_ps))
            rows.append({"recurrent": rec, "transformer": trans, "label": label,
                         "seed_ps": seed_ps, "median_p": median_p})
    return pd.DataFrame(rows)


# ---------------- 3. Paired bootstrap over test comments ----------------
def paired_bootstrap(metric_fn, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for rec, trans in PAIRS:
        seed_diffs = []
        for s in SEEDS:
            probs_r, y = load_outputs(rec, s)
            probs_t, _ = load_outputs(trans, s)
            n = y.shape[0]
            obs_diff = metric_fn(probs_t, y) - metric_fn(probs_r, y)
            boot_diffs = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.integers(0, n, n)
                boot_diffs[b] = metric_fn(probs_t[idx], y[idx]) - metric_fn(probs_r[idx], y[idx])
            p_boot = 2 * min((boot_diffs <= 0).mean(), (boot_diffs >= 0).mean())
            seed_diffs.append({"seed": s, "obs_diff": obs_diff, "p_boot": p_boot})
        median_p = float(np.median([d["p_boot"] for d in seed_diffs]))
        rows.append({"recurrent": rec, "transformer": trans, "median_p_boot": median_p,
                      "per_seed": seed_diffs})
    return pd.DataFrame(rows)


def macro_auc(probs, y):
    aucs = []
    for j in range(y.shape[1]):
        if y[:, j].sum() == 0 or y[:, j].sum() == len(y):
            continue
        from sklearn.metrics import roc_auc_score
        aucs.append(roc_auc_score(y[:, j], probs[:, j]))
    return float(np.mean(aucs))


def main():
    print("=" * 70)
    print("H2, full pre-registered methods: Tukey HSD, DeLong, paired bootstrap")
    print("=" * 70)

    print("\n--- 1. One-way ANOVA + Tukey HSD (macro ROC-AUC) ---")
    anova_p, tukey_df = tukey_hsd("roc_auc_macro")
    print(f"ANOVA p (6 models) = {anova_p:.5f}")
    print(tukey_df.to_string(index=False))
    tukey_df.to_csv("results_proposed/report/table_h2_tukey_hsd_auc.csv", index=False)

    print("\n--- 1b. One-way ANOVA + Tukey HSD (macro-F1(M), secondary) ---")
    anova_p_f1, tukey_df_f1 = tukey_hsd("f1_macro_m")
    print(f"ANOVA p (6 models) = {anova_p_f1:.5f}")
    print(tukey_df_f1.to_string(index=False))
    tukey_df_f1.to_csv("results_proposed/report/table_h2_tukey_hsd_f1m.csv", index=False)

    print("\n--- 2. DeLong's test per label, median p over 3 seeds ---")
    dl = delong_per_pair()
    dl_summary = dl.groupby(["recurrent", "transformer"])["median_p"].apply(
        lambda s: float(np.nanmedian(s))).reset_index(name="median_p_across_labels")
    pvals = dl_summary["median_p_across_labels"].tolist()
    dl_summary["p_holm_9pairs"] = holm(pvals)
    dl_summary["sig_0.05"] = dl_summary["p_holm_9pairs"] < 0.05
    print(dl_summary.to_string(index=False))
    dl.drop(columns=["seed_ps"]).round(5).to_csv("results_proposed/report/table_h2_delong_per_label.csv", index=False)
    dl_summary.round(5).to_csv("results_proposed/report/table_h2_delong_summary.csv", index=False)

    print("\n--- 3. Paired bootstrap over test comments (macro ROC-AUC, 2000 resamples) ---")
    boot = paired_bootstrap(macro_auc, n_boot=2000)
    boot_out = boot[["recurrent", "transformer", "median_p_boot"]].copy()
    boot_out["p_holm_9pairs"] = holm(boot_out["median_p_boot"].tolist())
    boot_out["sig_0.05"] = boot_out["p_holm_9pairs"] < 0.05
    print(boot_out.to_string(index=False))
    boot_out.round(5).to_csv("results_proposed/report/table_h2_bootstrap.csv", index=False)

    print("\nAll three pre-registered H2 methods saved to results_proposed/report/.")


if __name__ == "__main__":
    main()

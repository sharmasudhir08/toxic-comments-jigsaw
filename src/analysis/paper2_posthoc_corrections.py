"""
Paper-2 post-hoc corrections and additional checks, verifying specific claims raised in
`paper/AI Expert Review Comments How Reliable Are Architecture Rankings.docx` against saved
results. Read-only: no training, no new model inference, no rerun of DeLong/bootstrap/permutation.
All inputs are already-saved CSVs or already-saved test_outputs.npz/duplicate_screen.npz arrays.

What this computes, and from which saved files:

1. H7 length-bin Holm correction over the 10 pre-existing per-bin tests (5 bins x
   {f1_macro_m, roc_auc_macro}) in results_proposed/report/table_h7_length_control_nested_anova.csv.
   -> results_proposed/report/table_h7_length_control_holm10.csv

2. Duplicate-screened H1 nested ANOVA F(1,4), same method as H1 (src/family_comparison.py's
   nested_anova), fed from the already-saved per-seed screened metrics in
   results_proposed/report/table_h7_dup_screened_per_seed.csv.
   -> results_proposed/report/table_h1_nested_anova_screened.csv

2a. Same nested ANOVA for PR-AUC (macro), which has no existing per-seed screened column --
    computed here from the already-saved results_proposed/runs/<model>_p4/seed_<seed>/
    test_outputs.npz (probs, y_true) masked by data/duplicate_screen.npz, using sklearn's
    average_precision_score per label (no new predictions, only re-aggregation).
    -> results_proposed/report/table_h1_nested_anova_screened_v2.csv

5. H3 Holm correction computed separately within each benchmark family (3 recurrent, 3
   transformer), per preregistration.md Sec. 4 ("Holm correction is applied within each family
   of tests"), vs. the currently-saved pooled-6 correction in
   results_proposed/report/table_h3_vs_maslej_tuned_comparator.csv.
   -> results_proposed/report/table_h3_within_family_holm.csv

2b. H4 per-label margins over the benchmark, per model, from the already-saved
    results_proposed/report/table8_per_label_vs_maslej.csv (tuned rows), with an added
    "minority margin without threat" column (mean of severe_toxic + identity_hate only).
    -> results_proposed/report/table_h4_per_label_margins.csv

Usage: python -m src.paper2_posthoc_corrections   (run from the project root)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from statsmodels.stats.multitest import multipletests

from src.data import load_official_test
from src.analysis.family_comparison import nested_anova, RECURRENT, TRANSFORMER

REPORT_DIR = "results_proposed/report"
RUNS_DIR = "results_proposed/runs"
MODELS = ["lstm", "bilstm", "gru", "bert", "distilbert", "roberta"]
SEEDS = [42, 43, 44]


def item1_h7_holm10():
    h7 = pd.read_csv(f"{REPORT_DIR}/table_h7_length_control_nested_anova.csv")
    sub = h7[h7["metric"].isin(["f1_macro_m", "roc_auc_macro"])].copy()
    assert len(sub) == 10, f"expected 10 tests, got {len(sub)}"
    reject, p_holm, _, _ = multipletests(sub["p_family_1_4"].values, alpha=0.05, method="holm")
    sub["p_holm_10tests"] = p_holm
    sub["sig_holm"] = reject
    sub = sub.sort_values("p_family_1_4")
    out = f"{REPORT_DIR}/table_h7_length_control_holm10.csv"
    sub.round(5).to_csv(out, index=False)
    print("=== Item 1: H7 Holm over 10 tests ===")
    print(sub[["bin", "metric", "p_family_1_4", "p_holm_10tests", "sig_holm"]].to_string(index=False))
    print("Saved:", out)
    return sub


def item2_screened_nested_anova():
    per_seed = pd.read_csv(f"{REPORT_DIR}/table_h7_dup_screened_per_seed.csv")
    rows = []
    for metric_screened, metric_name in [("f1_macro_m_screened", "f1_macro_m"),
                                          ("roc_auc_macro_screened", "roc_auc_macro"),
                                          ("ece_macro_screened", "ece_macro")]:
        long_df = per_seed[["model", "seed", metric_screened]].rename(columns={metric_screened: "value"})
        long_df["family"] = long_df["model"].apply(lambda m: "recurrent" if m in RECURRENT else "transformer")
        res = nested_anova(long_df, metric_name)
        rows.append({"metric": metric_name, **res})
    df = pd.DataFrame(rows)
    out = f"{REPORT_DIR}/table_h1_nested_anova_screened.csv"
    df.round(5).to_csv(out, index=False)
    print("\n=== Item 2: duplicate-screened H1 nested ANOVA F(1,4) ===")
    print(df.to_string(index=False))
    print("Saved:", out)
    return df


def item2a_screened_pr_auc():
    """PR-AUC has no existing per-seed screened column, unlike f1/roc_auc/ece -- recompute from
    the saved test_outputs.npz + duplicate_screen.npz mask (same mask-building method as
    src/recompute_on_duplicate_screened_test.py), then run the same nested ANOVA."""
    test_df = load_official_test("data")
    test_ids = test_df["id"].values
    d = np.load("data/duplicate_screen.npz", allow_pickle=True)
    id_to_dup = dict(zip(d["test_id"], d["is_near_duplicate"]))
    dup_mask = np.array([bool(id_to_dup[i]) for i in test_ids])
    keep_mask = ~dup_mask

    rows = []
    for m in MODELS:
        for s in SEEDS:
            dd = np.load(f"{RUNS_DIR}/{m}_p4/seed_{s}/test_outputs.npz")
            probs, y = dd["probs"], dd["y_true"]
            assert len(probs) == len(test_ids), f"{m} seed {s}: row mismatch"
            probs_s, y_s = probs[keep_mask], y[keep_mask]
            aps = [average_precision_score(y_s[:, j], probs_s[:, j]) for j in range(y_s.shape[1])
                   if 0 < y_s[:, j].sum() < len(y_s)]
            rows.append({"model": m, "seed": s, "pr_auc_macro_screened": float(np.mean(aps))})
    per_seed = pd.DataFrame(rows)

    long_df = per_seed.rename(columns={"pr_auc_macro_screened": "value"})
    long_df["family"] = long_df["model"].apply(lambda mm: "recurrent" if mm in RECURRENT else "transformer")
    res = nested_anova(long_df, "pr_auc_macro")

    base = pd.read_csv(f"{REPORT_DIR}/table_h1_nested_anova_screened.csv")
    df = pd.concat([base, pd.DataFrame([{"metric": "pr_auc_macro", **res}])], ignore_index=True)
    out = f"{REPORT_DIR}/table_h1_nested_anova_screened_v2.csv"
    df.round(5).to_csv(out, index=False)
    print("\n=== Item 2a: duplicate-screened PR-AUC nested ANOVA F(1,4) ===")
    print(df.to_string(index=False))
    print("Saved:", out)
    return df


def item5_h3_within_family_holm():
    h3 = pd.read_csv(f"{REPORT_DIR}/table_h3_vs_maslej_tuned_comparator.csv").copy()
    h3["p_holm_within_family"] = np.nan
    for fam in ["recurrent", "transformer"]:
        mask = h3["family"] == fam
        _, p_adj, _, _ = multipletests(h3.loc[mask, "p_one_sided"].values, alpha=0.05, method="holm")
        h3.loc[mask, "p_holm_within_family"] = p_adj
    h3["sig_within_family_0.05"] = h3["p_holm_within_family"] < 0.05
    out = f"{REPORT_DIR}/table_h3_within_family_holm.csv"
    h3.round(5).to_csv(out, index=False)
    print("\n=== Item 5: H3 Holm within-family (3+3) vs. pooled-6 ===")
    print(h3[["model", "family", "p_one_sided", "p_holm_6models", "p_holm_within_family",
              "sig_within_family_0.05"]].to_string(index=False))
    print("Saved:", out)
    return h3


def item2b_h4_per_label_margins():
    per_label = pd.read_csv(f"{REPORT_DIR}/table8_per_label_vs_maslej.csv")
    tuned = per_label[per_label["model"].str.endswith("(tuned)")].copy()
    tuned["model"] = tuned["model"].str.replace(" (tuned)", "", regex=False)
    minority = {"severe_toxic", "threat", "identity_hate"}
    frequent = {"toxic", "obscene", "insult"}

    rows = []
    for model, sub in tuned.groupby("model", sort=False):
        d = dict(zip(sub["label"], sub["difference"]))
        min_margin = np.mean([d[l] for l in minority])
        min_margin_no_threat = np.mean([d[l] for l in minority if l != "threat"])
        freq_margin = np.mean([d[l] for l in frequent])
        rows.append({"model": model, "minority_margin": min_margin,
                     "minority_margin_without_threat": min_margin_no_threat,
                     "frequent_margin": freq_margin, "threat_margin": d["threat"]})
    df = pd.DataFrame(rows)
    out = f"{REPORT_DIR}/table_h4_per_label_margins.csv"
    df.round(5).to_csv(out, index=False)
    print("\n=== Item 2b: H4 per-label margins (with/without threat) ===")
    print(df.to_string(index=False))
    print("Saved:", out)
    return df


if __name__ == "__main__":
    item1_h7_holm10()
    item2_screened_nested_anova()
    item2a_screened_pr_auc()
    item5_h3_within_family_holm()
    item2b_h4_per_label_margins()

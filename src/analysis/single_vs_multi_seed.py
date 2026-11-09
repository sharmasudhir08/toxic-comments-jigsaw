"""
H5 (RQ4): do single-seed pairwise conclusions differ from multi-seed (3-seed) conclusions?

For each of the 9 cross-family model pairs (recurrent x transformer, tuned thresholds), and for
each label, computes per-comment correctness for both models at each seed and runs McNemar's test
(paired, on discordant predictions) -- this is what a researcher would conclude from a SINGLE seed.
Combines the 3 per-seed p-values with Fisher's method to get the MULTI-seed conclusion. Reports how
often the single-seed and multi-seed conclusions disagree at alpha=0.05, and how often the
better-performing model in the pair flips rank across seeds.

No new training or inference -- reads results_proposed/runs/<model>_p4/seed_<seed>/test_outputs.npz
(predictions already saved from the original runs). Requires all 6 models to have 3 seeds.

Usage: python -m src.single_vs_multi_seed
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues
from statsmodels.stats.contingency_tables import mcnemar

RECURRENT = ["lstm", "bilstm", "gru"]
TRANSFORMER = ["bert", "distilbert", "roberta"]
SEEDS = [42, 43, 44]
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
RESULTS_DIR = "results_proposed/runs"


def load(model, seed):
    path = os.path.join(RESULTS_DIR, f"{model}_p4", f"seed_{seed}", "test_outputs.npz")
    d = np.load(path)
    return d["probs"], d["y_true"]


def correctness(probs, y_true, label_idx, threshold=0.5):
    pred = (probs[:, label_idx] >= threshold).astype(int)
    return (pred == y_true[:, label_idx]).astype(int)


def main():
    print("=" * 70)
    print("H5 (RQ4): single-seed vs multi-seed pairwise conclusions")
    print("=" * 70)

    pairs = [(r, t) for r in RECURRENT for t in TRANSFORMER]
    rows = []

    for model_a, model_b in pairs:
        data_a = {s: load(model_a, s) for s in SEEDS}
        data_b = {s: load(model_b, s) for s in SEEDS}
        for li, label in enumerate(LABELS):
            seed_pvals = []
            seed_better = []  # which model has higher accuracy on this label, per seed
            for s in SEEDS:
                probs_a, y_a = data_a[s]
                probs_b, y_b = data_b[s]
                assert np.array_equal(y_a, y_b), "mismatched test sets between models/seeds"
                corr_a = correctness(probs_a, y_a, li)
                corr_b = correctness(probs_b, y_b, li)
                # 2x2 contingency: both correct, a-only, b-only, both wrong
                both_correct = int(((corr_a == 1) & (corr_b == 1)).sum())
                a_only = int(((corr_a == 1) & (corr_b == 0)).sum())
                b_only = int(((corr_a == 0) & (corr_b == 1)).sum())
                both_wrong = int(((corr_a == 0) & (corr_b == 0)).sum())
                table = [[both_correct, a_only], [b_only, both_wrong]]
                result = mcnemar(table, exact=False, correction=True)
                seed_pvals.append(result.pvalue)
                seed_better.append(model_a if corr_a.mean() > corr_b.mean() else model_b)

            _, fisher_p = combine_pvalues(seed_pvals, method="fisher")
            single_seed_sig = [p < 0.05 for p in seed_pvals]
            multi_seed_sig = fisher_p < 0.05
            rank_flips = len(set(seed_better)) > 1
            disagreement = any(s != multi_seed_sig for s in single_seed_sig)

            rows.append({
                "model_a": model_a, "model_b": model_b, "label": label,
                "p_seed42": seed_pvals[0], "p_seed43": seed_pvals[1], "p_seed44": seed_pvals[2],
                "sig_seed42": single_seed_sig[0], "sig_seed43": single_seed_sig[1], "sig_seed44": single_seed_sig[2],
                "p_fisher_combined": fisher_p, "sig_multi_seed": multi_seed_sig,
                "single_vs_multi_disagree": disagreement,
                "better_model_per_seed": ",".join(seed_better), "rank_flips_across_seeds": rank_flips,
            })

    df = pd.DataFrame(rows)
    out_dir = "results_proposed/report"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "table_h5_single_vs_multi_seed.csv")
    df.round(5).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df)} rows = 9 pairs x 6 labels)")

    n_disagree = df["single_vs_multi_disagree"].sum()
    n_flips = df["rank_flips_across_seeds"].sum()
    print(f"\nSingle-seed vs multi-seed conclusion disagreement: {n_disagree}/{len(df)} "
          f"(pair,label) combinations ({100*n_disagree/len(df):.1f}%)")
    print(f"Better-model rank flips across at least one seed: {n_flips}/{len(df)} "
          f"({100*n_flips/len(df):.1f}%)")

    print("\n=== Any single seed significant, but multi-seed (Fisher-combined) not significant? ===")
    sub = df[df["single_vs_multi_disagree"]]
    if len(sub):
        print(sub[["model_a", "model_b", "label", "sig_seed42", "sig_seed43", "sig_seed44",
                    "p_fisher_combined", "sig_multi_seed"]].to_string(index=False))
    else:
        print("  None.")

    print("\n=== Rank (better model) flips across seeds ===")
    sub2 = df[df["rank_flips_across_seeds"]]
    if len(sub2):
        print(sub2[["model_a", "model_b", "label", "better_model_per_seed"]].to_string(index=False))
    else:
        print("  None.")


if __name__ == "__main__":
    main()

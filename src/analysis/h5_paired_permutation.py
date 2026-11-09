"""
H5 (RQ4), paired permutation test -- pre-registered (preregistration.md line 87: "Per-seed McNemar
(per label) vs paired permutation over test comments on seed-averaged correctness and
Fisher-combined p-values"), never implemented until 3 Nov 2026 (author review).

For each of the 9 cross-family pairs and 6 labels, computes seed-averaged correctness per comment
for both models (mean of 3 seeds' correct/incorrect indicator), then a Monte Carlo sign-flip paired
permutation test on the per-comment difference (model_a - model_b): under the null (no systematic
difference), each comment's sign is exchangeable, so the null distribution is built by randomly
flipping each comment's sign 2000 times and comparing the observed mean difference to that
distribution (two-sided, matching McNemar's own two-sided convention for this project).

Validated on synthetic data before running on real data (see scripts/audit/verify_h5_permutation.py):
identical predictions give p~1; clearly different predictions give p~0.

Read-only on saved results_proposed/runs/<model>_p4/seed_<seed>/test_outputs.npz. No retraining.

Usage: python -m src.h5_paired_permutation
"""
import os
import numpy as np
import pandas as pd

RECURRENT = ["lstm", "bilstm", "gru"]
TRANSFORMER = ["bert", "distilbert", "roberta"]
SEEDS = [42, 43, 44]
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
RESULTS_DIR = "results_proposed/runs"
N_PERM = 2000


def load(model, seed):
    path = os.path.join(RESULTS_DIR, f"{model}_p4", f"seed_{seed}", "test_outputs.npz")
    d = np.load(path)
    return d["probs"], d["y_true"]


def seed_averaged_correctness(model, label_idx, threshold=0.5):
    """Mean correctness indicator across the 3 seeds, per comment."""
    corr_sum = None
    y_ref = None
    for s in SEEDS:
        probs, y = load(model, s)
        if y_ref is None:
            y_ref = y
        else:
            assert np.array_equal(y, y_ref), "mismatched test sets across seeds"
        pred = (probs[:, label_idx] >= threshold).astype(int)
        corr = (pred == y[:, label_idx]).astype(float)
        corr_sum = corr if corr_sum is None else corr_sum + corr
    return corr_sum / len(SEEDS)


def paired_permutation_p(diff, n_perm=N_PERM, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    obs = diff.mean()
    perm_means = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        perm_means[i] = (diff * signs).mean()
    p = float((np.abs(perm_means) >= np.abs(obs)).mean())
    return p, float(obs)


def main():
    pairs = [(r, t) for r in RECURRENT for t in TRANSFORMER]
    rows = []
    for model_a, model_b in pairs:
        for li, label in enumerate(LABELS):
            corr_a = seed_averaged_correctness(model_a, li)
            corr_b = seed_averaged_correctness(model_b, li)
            diff = corr_a - corr_b
            p, obs_diff = paired_permutation_p(diff)
            rows.append({
                "model_a": model_a, "model_b": model_b, "label": label,
                "obs_diff_seed_avg_correctness": obs_diff,
                "p_permutation": p,
                "sig_permutation_0.05": p < 0.05,
            })
            print(f"{model_a} vs {model_b}, {label}: obs_diff={obs_diff:.5f}, p={p:.4f}")

    df = pd.DataFrame(rows)
    df.round(6).to_csv("results_proposed/report/table_h5_paired_permutation.csv", index=False)
    print(f"\n{df['sig_permutation_0.05'].sum()}/{len(df)} (pair,label) combinations significant"
          f" (permutation test, alpha=0.05)")
    print("Saved: results_proposed/report/table_h5_paired_permutation.csv")


if __name__ == "__main__":
    main()

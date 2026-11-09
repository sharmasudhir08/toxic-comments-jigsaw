"""
Author-approved additional analysis, 4 Nov 2026 (read-only on saved predictions/metrics; no
training). Prompted by an external methodological suggestion (a paired-by-seed delta protocol,
as in arXiv:2511.19794) and by the reviewer's own observation that `preregistration.md` confirms
all six models share identical per-seed train/val splits -- so a seed is a matched unit across
every model, which the pre-registered Tukey HSD test (pooled variance across all 18 runs) does
not exploit.

For each of the 9 pre-registered cross-family pairs, computes the per-seed paired delta in macro
ROC-AUC (transformer minus recurrent, same seed), then:
  1. A paired t-test (df=2) on the 3 deltas -- mean, 95% CI, one-sided p (H2 is directional).
  2. An exact one-sided sign-flip permutation test on the 3 deltas (2**3 = 8 equally likely sign
     patterns -- the smallest achievable one-sided p-value is 1/8 = 0.125, an inherent floor of
     this design at n=3, not a flaw in the implementation).
Both p-value sets are Holm-corrected over the 9 pairs, matching this project's existing
convention for every other H2 test.

This is a secondary/supplementary analysis. It does not replace the pre-registered primary result
(Tukey HSD + Holm, Table 5); it tests whether exploiting the seed-pairing structure changes the
practical conclusion.

Usage: python -m scripts.audit.h2_paired_by_seed
"""
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

RECURRENT = ["lstm", "bilstm", "gru"]
TRANSFORMER = ["bert", "distilbert", "roberta"]
SEEDS = [42, 43, 44]
RESULTS_DIR = "results_proposed/runs"
PAIRS = [(r, t) for r in RECURRENT for t in TRANSFORMER]


def load_metric(model, seed, metric="roc_auc_macro"):
    path = os.path.join(RESULTS_DIR, f"{model}_p4", f"seed_{seed}", "metrics.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["test"][metric]


def holm(pvals):
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * pvals[idx]
        running_max = max(running_max, val)
        adj[idx] = min(running_max, 1.0)
    return adj


def sign_flip_p_one_sided(deltas):
    """Exact permutation p-value: P(permuted mean >= observed mean) over all 2^3 sign patterns."""
    observed_mean = np.mean(deltas)
    n = len(deltas)
    count_ge = 0
    total = 0
    for signs in itertools.product([1, -1], repeat=n):
        permuted_mean = np.mean(np.array(signs) * np.array(deltas))
        if permuted_mean >= observed_mean - 1e-12:
            count_ge += 1
        total += 1
    return count_ge / total


def main():
    rows = []
    for r, t in PAIRS:
        deltas = np.array(
            [load_metric(t, s) - load_metric(r, s) for s in SEEDS]
        )
        mean_delta = deltas.mean()
        sd_delta = deltas.std(ddof=1)
        se = sd_delta / np.sqrt(len(deltas))
        df = len(deltas) - 1

        t_stat = mean_delta / se if se > 0 else np.inf
        p_one_sided_ttest = stats.t.sf(t_stat, df)
        t_crit = stats.t.ppf(0.975, df)
        ci_low = mean_delta - t_crit * se
        ci_high = mean_delta + t_crit * se

        p_perm = sign_flip_p_one_sided(deltas)

        rows.append(
            {
                "recurrent": r,
                "transformer": t,
                "deltas": list(np.round(deltas, 5)),
                "mean_delta": mean_delta,
                "sd_delta": sd_delta,
                "ci95_low_ttest": ci_low,
                "ci95_high_ttest": ci_high,
                "t_stat": t_stat,
                "p_one_sided_ttest": p_one_sided_ttest,
                "p_one_sided_permutation": p_perm,
            }
        )

    df = pd.DataFrame(rows)
    df["p_ttest_holm_9pairs"] = holm(df["p_one_sided_ttest"].tolist())
    df["p_perm_holm_9pairs"] = holm(df["p_one_sided_permutation"].tolist())
    df["sig_ttest_0.05"] = df["p_ttest_holm_9pairs"] < 0.05
    df["sig_perm_0.05"] = df["p_perm_holm_9pairs"] < 0.05

    out_path = "results_proposed/report/table_h2_paired_by_seed.csv"
    df.round(5).to_csv(out_path, index=False)
    print(df.round(4).to_string())
    print(f"\nWrote {out_path}")
    print(f"\nPaired t-test significant (Holm, 9 pairs): {df['sig_ttest_0.05'].sum()}/9")
    print(f"Sign-flip permutation significant (Holm, 9 pairs): {df['sig_perm_0.05'].sum()}/9")
    print(f"Smallest possible one-sided permutation p at n=3: {1/8} (inherent floor)")


if __name__ == "__main__":
    main()

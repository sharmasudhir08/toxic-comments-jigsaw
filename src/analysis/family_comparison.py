"""
Family-level comparison: recurrent (LSTM, Bi-LSTM, GRU) vs transformer (BERT, DistilBERT,
RoBERTa) -- the central test for H1 (preregistration.md).

Unit of analysis is the SIX MODELS (not seeds): each model contributes one mean value per
metric (averaged over its 3 primary seeds). Primary test per preregistration.md H1: nested
ANOVA (models nested within families), F(1,4) -- the family mean square tested against the
model-within-family mean square, not the seed-level residual, since models (not individual
seed-runs) are the real independent replicates for the family-level claim. Robustness checks:
Welch's t-test on the 3-vs-3 model means (illustration of the same model-level unit), the exact
permutation test over all 20 ways of splitting 6 models into two groups of 3 (per
preregistration.md H1's own wording -- not deduplicated to 10 complementary pairs), and a
crossed-random-effects mixed model (random intercepts for model and seed).

Written independently for this project (same general nested-ANOVA logic as is standard for this
family/model/seed design, not copied from any other project's code).

No new training. Reads results_proposed/runs/<model>_p4/seed_<seed>/metrics.json only.
Requires all 6 models to have 3 seeds each -- refuses to run a family-level test otherwise.

Usage: python -m src.family_comparison
"""
import json
import os
import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats

RECURRENT = ["lstm", "bilstm", "gru"]
TRANSFORMER = ["bert", "distilbert", "roberta"]
MODELS = RECURRENT + TRANSFORMER
SEEDS = [42, 43, 44]  # primary-analysis seeds only -- recurrent 45-46 are supplementary, excluded here
RESULTS_DIR = "results_proposed/runs"
METRICS = ["f1_macro_m", "roc_auc_macro", "pr_auc_macro", "ece_macro"]


def load_seeds(model):
    out = {}
    for s in SEEDS:
        path = os.path.join(RESULTS_DIR, f"{model}_p4", f"seed_{s}", "metrics.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                out[s] = json.load(f)["test"]
    return out


def nested_anova(long_df, metric):
    """Family nested ANOVA: F(1,4) = MS_Family / MS_Model(Family).
    long_df has one row per (model, seed) with columns 'family', 'model', 'value'."""
    d = long_df[["family", "model", "value"]].copy()
    grand_mean = d["value"].mean()
    n_total = len(d)

    fam_means = d.groupby("family")["value"].mean()
    model_means = d.groupby(["family", "model"])["value"].mean()
    n_seeds_per_model = d.groupby("model").size().iloc[0]
    n_models_per_family = d.groupby("family")["model"].nunique()
    n_families = fam_means.shape[0]
    n_models = model_means.shape[0]

    # SS_Family: each family's contribution, weighted by its total observation count
    ss_family = 0.0
    for fam, fmean in fam_means.items():
        n_fam_obs = n_models_per_family[fam] * n_seeds_per_model
        ss_family += n_fam_obs * (fmean - grand_mean) ** 2
    df_family = n_families - 1

    # SS_Model(Family): each model's deviation from its own family mean, weighted by its seed count
    ss_model_in_family = 0.0
    for (fam, model), mmean in model_means.items():
        ss_model_in_family += n_seeds_per_model * (mmean - fam_means[fam]) ** 2
    df_model_in_family = n_models - n_families

    # SS_Error: seed-level residual within each model
    ss_error = 0.0
    for model, grp in d.groupby("model"):
        mmean = grp["value"].mean()
        ss_error += ((grp["value"] - mmean) ** 2).sum()
    df_error = n_total - n_models

    ms_family = ss_family / df_family
    ms_model_in_family = ss_model_in_family / df_model_in_family
    ms_error = ss_error / df_error

    f_family = ms_family / ms_model_in_family if ms_model_in_family > 0 else np.nan
    p_family = float(stats.f.sf(f_family, df_family, df_model_in_family)) if not np.isnan(f_family) else np.nan

    f_model_in_family = ms_model_in_family / ms_error if ms_error > 0 else np.nan
    p_model_in_family = float(stats.f.sf(f_model_in_family, df_model_in_family, df_error)) if not np.isnan(f_model_in_family) else np.nan

    return {
        "metric": metric,
        "ss_family": ss_family, "df_family": df_family,
        "ss_model_in_family": ss_model_in_family, "df_model_in_family": df_model_in_family,
        "ss_error": ss_error, "df_error": df_error,
        "ms_family": ms_family, "ms_model_in_family": ms_model_in_family, "ms_error": ms_error,
        "F_family_1_4": f_family, "p_family_1_4": p_family,
        "F_model_in_family_4_12": f_model_in_family, "p_model_in_family_4_12": p_model_in_family,
    }


def mixed_effects_check(long_df, metric):
    """Crossed random-effects model: fixed effect = family, random intercepts for model and seed
    (crossed, since each seed value 42/43/44 appears once per model, not nested)."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {"metric": metric, "mixed_model_error": "statsmodels not available"}

    d = long_df.copy()
    d["const_group"] = 1  # trivial primary grouping; model and seed are both variance components
    try:
        md = smf.mixedlm(
            "value ~ family", d, groups="const_group",
            vc_formula={"model": "0 + C(model)", "seed_re": "0 + C(seed)"},
        )
        res = md.fit(reml=True)
        family_coef = [c for c in res.params.index if c.startswith("family")]
        if not family_coef:
            return {"metric": metric, "mixed_model_error": "no family coefficient found"}
        coef_name = family_coef[0]
        return {
            "metric": metric,
            "family_coef": float(res.params[coef_name]),
            "family_coef_se": float(res.bse[coef_name]),
            "family_coef_p": float(res.pvalues[coef_name]),
            "converged": bool(res.converged),
        }
    except Exception as e:
        return {"metric": metric, "mixed_model_error": str(e)}


def main():
    data = {m: load_seeds(m) for m in MODELS}
    incomplete = [m for m in MODELS if len(data[m]) != 3]
    print("=" * 70)
    print("FAMILY-LEVEL COMPARISON: recurrent vs transformer (H1)")
    print("=" * 70)
    print("\nSeed completeness:", {m: sorted(data[m].keys()) for m in MODELS})
    if incomplete:
        print(f"\nREFUSING to run family-level test: {incomplete} do not have all 3 seeds.")
        return

    # Build long-format dataframe: one row per (model, seed, metric)
    rows = []
    for m in MODELS:
        fam = "recurrent" if m in RECURRENT else "transformer"
        for s in SEEDS:
            for metric in METRICS:
                rows.append({"family": fam, "model": m, "seed": s, "metric": metric,
                             "value": data[m][s][metric]})
    long_all = pd.DataFrame(rows)

    model_means = {m: {} for m in MODELS}
    for m in MODELS:
        for metric in METRICS:
            model_means[m][metric] = np.mean([data[m][s][metric] for s in SEEDS])

    print("\n=== Per-model means (3 seeds each) ===")
    for m in MODELS:
        fam = "recurrent" if m in RECURRENT else "transformer"
        print(f"  {m:12s} ({fam:11s}) F1_M={model_means[m]['f1_macro_m']:.4f}  "
              f"AUC={model_means[m]['roc_auc_macro']:.4f}  "
              f"PR-AUC={model_means[m]['pr_auc_macro']:.4f}  "
              f"ECE={model_means[m]['ece_macro']:.4f}")

    print("\n=== PRIMARY TEST: Nested ANOVA, F(1,4), family tested against model(family) ===")
    nested_rows = []
    for metric in METRICS:
        long_metric = long_all[long_all["metric"] == metric]
        res = nested_anova(long_metric, metric)
        nested_rows.append(res)
        print(f"  {metric:15s} F(1,4)={res['F_family_1_4']:.4f}  p={res['p_family_1_4']:.5f}  "
              f"[model(family) F(4,12)={res['F_model_in_family_4_12']:.4f}  "
              f"p={res['p_model_in_family_4_12']:.5f}]")
    nested_df = pd.DataFrame(nested_rows)

    print("\n=== Robustness 1: Welch's t-test on model-level means (3 vs 3) ===")
    h1_rows = []
    for metric in METRICS:
        rec_vals = np.array([model_means[m][metric] for m in RECURRENT])
        trans_vals = np.array([model_means[m][metric] for m in TRANSFORMER])
        t, p = stats.ttest_ind(trans_vals, rec_vals, equal_var=False)
        diff = trans_vals.mean() - rec_vals.mean()
        direction = "transformer higher" if diff > 0 else "recurrent higher"
        print(f"  {metric:15s} recurrent_mean={rec_vals.mean():.4f}  transformer_mean={trans_vals.mean():.4f}  "
              f"diff={diff:+.4f} ({direction})  Welch_t(3v3) p={p:.4f}")
        h1_rows.append({"metric": metric, "recurrent_mean": rec_vals.mean(), "transformer_mean": trans_vals.mean(),
                         "diff": diff, "direction": direction, "welch_t_3v3_p": p})

    print("\n=== Robustness 2: exact permutation over all 20 ways to split 6 models into 3-vs-3 ===")
    print("    (per preregistration.md H1 wording -- NOT deduplicated to 10 complementary pairs)")
    all_models = MODELS
    for metric in METRICS:
        vals = {m: model_means[m][metric] for m in all_models}
        observed_diff = abs(np.mean([vals[m] for m in TRANSFORMER]) - np.mean([vals[m] for m in RECURRENT]))
        all_splits = list(combinations(all_models, 3))  # 20 splits, NOT deduplicated
        diffs = []
        for combo in all_splits:
            rest = tuple(m for m in all_models if m not in combo)
            d_val = abs(np.mean([vals[m] for m in combo]) - np.mean([vals[m] for m in rest]))
            diffs.append(d_val)
        diffs = np.array(diffs)
        rank = (diffs >= observed_diff - 1e-12).sum()
        perm_p = rank / len(diffs)
        print(f"    {metric:15s} observed_diff={observed_diff:.4f}  "
              f"rank={rank}/{len(diffs)} splits >= observed  exact_p={perm_p:.4f}")
        row = next(r for r in h1_rows if r["metric"] == metric)
        row["exact_permutation_p_20splits"] = perm_p
        row["permutation_splits_total"] = len(diffs)

    print("\n=== Robustness 3: crossed-random-effects mixed model (family fixed; random model + seed) ===")
    mixed_rows = []
    for metric in METRICS:
        long_metric = long_all[long_all["metric"] == metric]
        res = mixed_effects_check(long_metric, metric)
        mixed_rows.append(res)
        if "mixed_model_error" in res:
            print(f"  {metric:15s} ERROR: {res['mixed_model_error']}")
        else:
            print(f"  {metric:15s} family_coef={res['family_coef']:+.4f}  se={res['family_coef_se']:.4f}  "
                  f"p={res['family_coef_p']:.5f}  converged={res['converged']}")
    mixed_df = pd.DataFrame(mixed_rows)

    print("\n  WARNING, illustration only -- treating all 18 runs as independent overstates evidence:")
    for metric in METRICS:
        rec_runs = np.array([data[m][s][metric] for m in RECURRENT for s in SEEDS])
        trans_runs = np.array([data[m][s][metric] for m in TRANSFORMER for s in SEEDS])
        t, p = stats.ttest_ind(trans_runs, rec_runs, equal_var=False)
        print(f"    {metric:15s} Welch over 18 runs (WRONG unit of analysis): p={p:.4f}")
        row = next(r for r in h1_rows if r["metric"] == metric)
        row["welch_18runs_WRONG_unit_p"] = p

    os.makedirs("results_proposed/report", exist_ok=True)
    h1_df = pd.DataFrame(h1_rows)
    h1_df.round(5).to_csv("results_proposed/report/table_h1_family_comparison.csv", index=False)
    nested_df.round(5).to_csv("results_proposed/report/table_h1_nested_anova.csv", index=False)
    mixed_df.round(5).to_csv("results_proposed/report/table_h1_mixed_effects.csv", index=False)
    print("\nSaved: table_h1_family_comparison.csv, table_h1_nested_anova.csv, table_h1_mixed_effects.csv")


if __name__ == "__main__":
    main()

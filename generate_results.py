"""
Aggregate the finished runs into the tables and figures for the paper.

    python generate_results.py                       # reads results/
    python generate_results.py --results_dir results_smoke
    python generate_results.py --results_dir results_proposed --extra_results_dirs results_full
        (one report for several result folders; written to <results_dir>/report/)

Output (in <results_dir>/report/):
    table1_overall_mean_std.csv     overall metrics, mean +- std over the runs (per model)
    table1_overall_numeric.csv      same, numeric mean/std columns
    table2_per_label_f1.csv         per-label F1  (mean +- std)
    table3_per_label_precision_recall.csv
    table4_vs_paper.csv             paper's published LSTM/Bi-LSTM/GRU next to our re-run recurrent
                                    models and the transformers (mean +- std)
    table5_significance.csv         pairwise Welch t-test on macro-F1 across runs (+ Holm)
    table6_thresholds.csv           per-label thresholds tuned on validation (proposed P4 models)
    table7_vs_maslej.csv            our models (mean +- std) next to Maslej-Krešňáková et al. (2020), same test set
    table7b_vs_maslej_tests.csv     one-sided one-sample t-tests vs the family target, seeds above, Holm
    table8_per_label_vs_maslej.csv  per-label F1 vs their Tables 15-18
    all_runs.csv                    one row per (model, seed)
    fig_macro_f1_comparison.png  fig_per_label_f1.png  fig_runs_boxplot.png
    fig_confusion_<model>.png    fig_roc_curves.png
"""

import argparse
import glob
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import average_precision_score, roc_curve

from src.config import LABELS
from src.maslej_baselines import (MASLEJ_OVERALL, MASLEJ_PER_LABEL_F1, MASLEJ_PER_LABEL_TUNED_F1,
                                  MASLEJ_TARGET)
from src.model_registry import MODELS, RNN_MODELS, RNN_PROPOSED, TRANSFORMER_COMMON
from src.paper_baselines import PAPER_COUNTS, paper_metrics

sns.set_theme(style="whitegrid")
OVERALL = ["exact_match_accuracy", "precision_micro", "recall_micro", "f1_micro",
           "precision_macro", "recall_macro", "f1_macro", "f1_macro_m", "roc_auc_macro", "pr_auc_macro",
           "mcc_macro", "ece_macro", "brier_macro", "hamming_loss"]
NICE = {"exact_match_accuracy": "Exact-match Acc", "precision_micro": "Precision (micro)",
        "recall_micro": "Recall (micro)", "f1_micro": "F1 (micro)",
        "precision_macro": "Precision (macro)", "recall_macro": "Recall (macro)",
        "f1_macro": "F1 (macro, mean per label)", "f1_macro_m": "F1 (macro, from macro P/R)",
        "roc_auc_macro": "ROC-AUC (macro)", "pr_auc_macro": "PR-AUC (macro)", "mcc_macro": "MCC (macro)",
        "ece_macro": "ECE (macro)", "brier_macro": "Brier (macro)", "hamming_loss": "Hamming loss"}
# base-paper recurrent re-runs, proposed recurrent models, transformers, transformers under the common protocol
ALL_MODELS = {**RNN_MODELS, **RNN_PROPOSED, **MODELS, **TRANSFORMER_COMMON}
RECURRENT_KEYS = set(RNN_MODELS) | set(RNN_PROPOSED)
BLUE = {"BERT": "#1f4e8c", "DistilBERT": "#4c8bd6", "RoBERTa": "#0b2a52",
        "LSTM": "#f4a582", "Bi-LSTM": "#d6604d", "GRU": "#b2182b"}
STAGE_GREEN = {"p1": "#c7e9c0", "p2": "#74c476", "p3": "#31a354", "p4": "#006d2c"}
STAGE_BLUE = {"p3": "#6baed6", "p4": "#08519c"}
BLUE.update({v["label"]: STAGE_GREEN[v["stage"]] for v in RNN_PROPOSED.values()})
BLUE.update({v["label"]: STAGE_BLUE[v["stage"]] for v in TRANSFORMER_COMMON.values()})


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (nan stays nan)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    running = 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok])]):
        running = max(running, min(1.0, (len(ok) - rank) * p[i]))
        out[i] = running
    return out


def pm(mean, std):
    return f"{mean:.4f} ± {std:.4f}" if not np.isnan(std) else f"{mean:.4f}"


def load_runs(results_dir):
    runs = []
    for p in sorted(glob.glob(os.path.join(results_dir, "runs", "*", "seed_*", "metrics.json"))):
        with open(p) as f:
            r = json.load(f)
        r["_dir"] = os.path.dirname(p)
        npz = os.path.join(r["_dir"], "test_outputs.npz")
        if "pr_auc_macro" not in r["test"] and os.path.isfile(npz):     # runs made before PR-AUC was added
            d = np.load(npz)
            aps = [float(average_precision_score(d["y_true"][:, j], d["probs"][:, j]))
                   for j in range(len(LABELS)) if d["y_true"][:, j].sum()]
            r["test"]["pr_auc_macro"] = float(np.mean(aps)) if aps else float("nan")
        t = r["test"]
        if "f1_macro_m" not in t:                                        # runs made before it was added
            pm_, rm_ = t["precision_macro"], t["recall_macro"]
            t["f1_macro_m"] = 2 * pm_ * rm_ / (pm_ + rm_) if (pm_ + rm_) else 0.0
        runs.append(r)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--extra_results_dirs", nargs="*", default=[],
                    help="also include the runs of these folders (read only); report goes to --results_dir")
    ap.add_argument("--include_base_paper", action="store_true",
                    help="add the earlier conference paper's LSTM/Bi-LSTM/GRU rows and bars (off by default, scope v3)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="restrict to these seeds only (e.g. --seeds 42 43 44 for the primary "
                         "analysis). Default: every seed found, which for the recurrent models "
                         "includes the 45-46 supplementary seeds -- almost never what you want "
                         "for primary tables. preregistration.md's primary analysis uses 42-44 "
                         "for all six models.")
    a = ap.parse_args()
    paper_names = list(PAPER_COUNTS) if a.include_base_paper else []

    runs = load_runs(a.results_dir)
    for d in a.extra_results_dirs:
        runs += load_runs(d)
    if a.seeds is not None:
        before = len(runs)
        runs = [r for r in runs if r["seed"] in a.seeds]
        print(f"--seeds filter {a.seeds}: kept {len(runs)}/{before} runs")
    if not runs:
        raise SystemExit(f"No finished runs found in {a.results_dir}/runs. "
                         "Run  python -m src.run_experiments  first.")
    report = os.path.join(a.results_dir, "report")
    os.makedirs(report, exist_ok=True)

    keys = [k for k in ALL_MODELS if any(r["model"] == k for r in runs)]
    label_of = {k: ALL_MODELS[k]["label"] for k in keys}
    by_model = {k: [r for r in runs if r["model"] == k] for k in keys}
    sources = sorted({r["test_source"] for r in runs})
    print("Test set:", "; ".join(sources))
    print({label_of[k]: len(v) for k, v in by_model.items()}, "finished runs per model")

    # ---- all runs --------------------------------------------------------
    rows = []
    for r in runs:
        row = {"model": r["model_label"], "seed": r["seed"], "train_minutes": r["train_minutes"]}
        row.update({m: r["test"].get(m, float("nan")) for m in OVERALL})
        for lab in LABELS:
            row[f"f1_{lab}"] = r["test"]["per_label"][lab]["f1"]
        rows.append(row)
    runs_df = pd.DataFrame(rows)
    runs_df.round(4).to_csv(os.path.join(report, "all_runs.csv"), index=False)

    # ---- table 1: overall --------------------------------------------------
    fmt_rows, num_rows = {}, []
    for k in keys:
        sub = runs_df[runs_df.model == label_of[k]]
        fr, nr = {"runs": len(sub)}, {"model": label_of[k], "runs": len(sub)}
        for m in OVERALL:
            mean, std = sub[m].mean(), sub[m].std(ddof=1) if len(sub) > 1 else float("nan")
            fr[NICE[m]] = pm(mean, std)
            nr[f"{m}_mean"], nr[f"{m}_std"] = mean, std
        fr["train min/run"] = round(sub["train_minutes"].mean(), 1)
        fmt_rows[label_of[k]] = fr
        num_rows.append(nr)
    t1 = pd.DataFrame(fmt_rows).T
    t1.to_csv(os.path.join(report, "table1_overall_mean_std.csv"))
    pd.DataFrame(num_rows).round(5).to_csv(os.path.join(report, "table1_overall_numeric.csv"), index=False)

    # ---- tables 2 & 3: per label ------------------------------------------
    f1_t, pr_t = {}, {}
    for k in keys:
        f1_t[label_of[k]], pr_t[label_of[k]] = {}, {}
        for lab in LABELS:
            vals = {m: np.array([r["test"]["per_label"][lab][m] for r in by_model[k]])
                    for m in ("f1", "precision", "recall")}
            sd = lambda v: v.std(ddof=1) if len(v) > 1 else float("nan")
            f1_t[label_of[k]][lab] = pm(vals["f1"].mean(), sd(vals["f1"]))
            pr_t[label_of[k]][lab + " P"] = pm(vals["precision"].mean(), sd(vals["precision"]))
            pr_t[label_of[k]][lab + " R"] = pm(vals["recall"].mean(), sd(vals["recall"]))
    pd.DataFrame(f1_t).T.to_csv(os.path.join(report, "table2_per_label_f1.csv"))
    pd.DataFrame(pr_t).T.to_csv(os.path.join(report, "table3_per_label_precision_recall.csv"))

    # ---- table 4: vs the paper --------------------------------------------
    cmp_rows = {}
    for name in paper_names:
        pmx = paper_metrics(name)
        cmp_rows[name] = {"runs": 1, **{c: f"{pmx[c]:.4f}" for c in
                          ["precision_micro", "recall_micro", "f1_micro",
                           "precision_macro", "recall_macro", "f1_macro"]}}
    for k in keys:
        sub = runs_df[runs_df.model == label_of[k]]
        cmp_rows[label_of[k]] = {"runs": len(sub), **{
            c: pm(sub[c].mean(), sub[c].std(ddof=1) if len(sub) > 1 else float("nan"))
            for c in ["precision_micro", "recall_micro", "f1_micro",
                      "precision_macro", "recall_macro", "f1_macro"]}}
    t4 = pd.DataFrame(cmp_rows).T
    t4.to_csv(os.path.join(report, "table4_vs_paper.csv"))

    # ---- table 5: significance --------------------------------------------
    sig = []
    for k1, k2 in itertools.combinations(keys, 2):
        x = runs_df[runs_df.model == label_of[k1]]["f1_macro"]
        y = runs_df[runs_df.model == label_of[k2]]["f1_macro"]
        if len(x) > 1 and len(y) > 1:
            t, p = stats.ttest_ind(x, y, equal_var=False)
            sig.append({"A": label_of[k1], "B": label_of[k2], "mean_diff_macroF1": x.mean() - y.mean(),
                        "t": t, "p_value": p, "significant_at_0.05": bool(p < 0.05)})
    if sig:
        sig_df = pd.DataFrame(sig)
        # Holm-Bonferroni correction over all model pairs (pre-registered, see preregistration.md)
        order = sig_df["p_value"].sort_values().index
        m, running = len(sig_df), 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (m - rank) * sig_df.at[i, "p_value"]))
            sig_df.at[i, "p_holm"] = running
        sig_df["significant_holm_0.05"] = sig_df["p_holm"] < 0.05
        sig_df.round(5).to_csv(os.path.join(report, "table5_significance.csv"), index=False)

    # ---- table 6: tuned thresholds (proposed P4) ---------------------------
    th_rows = {}
    for k in keys:
        ths = [r["thresholds"] for r in by_model[k] if r.get("thresholds")]
        if ths:
            th_rows[label_of[k]] = {lab: pm(np.mean([t[lab] for t in ths]),
                                            np.std([t[lab] for t in ths], ddof=1) if len(ths) > 1 else float("nan"))
                                    for lab in LABELS}
    if th_rows:
        pd.DataFrame(th_rows).T.to_csv(os.path.join(report, "table6_thresholds.csv"))

    # ---- table 7: vs Maslej-Krešňáková et al. (2020), same official test set (preregistration v4, H3) ----
    cmp_cols = ["exact_match_accuracy", "roc_auc_macro", "precision_micro", "recall_micro", "f1_micro",
                "precision_macro", "recall_macro", "f1_macro_m", "f1_macro"]
    t7 = {}
    for name, v in MASLEJ_OVERALL.items():
        t7[name] = {"runs": "1 (published)", "source": v["source"],
                    **{c: (f"{v[c]:.4f}" if c in v else "–") for c in cmp_cols}}
    for k in keys:
        sub = runs_df[runs_df.model == label_of[k]]
        t7[label_of[k]] = {"runs": len(sub), "source": "this study", **{
            c: pm(sub[c].mean(), sub[c].std(ddof=1) if len(sub) > 1 else float("nan")) for c in cmp_cols}}
    pd.DataFrame(t7).T.to_csv(os.path.join(report, "table7_vs_maslej.csv"))

    # tests against the family target (one-sided one-sample t-test, seeds above, Holm over all tests)
    tests = []
    for k in keys:
        fam = "recurrent" if k in RECURRENT_KEYS else "transformer"
        target = MASLEJ_OVERALL[MASLEJ_TARGET[fam]]
        sub = runs_df[runs_df.model == label_of[k]]
        for c in ["roc_auc_macro", "f1_micro", "f1_macro_m", "exact_match_accuracy"]:
            x = sub[c].dropna().values
            p = (stats.ttest_1samp(x, target[c], alternative="greater").pvalue
                 if len(x) > 1 and np.std(x) > 0 else float("nan"))
            tests.append({"model": label_of[k], "family": fam, "metric": NICE[c], "ours_mean": x.mean(),
                          "ours_sd": x.std(ddof=1) if len(x) > 1 else float("nan"),
                          "maslej_target": target[c], "maslej_source": target["source"],
                          "difference": x.mean() - target[c],
                          "seeds_above": f"{int((x > target[c]).sum())}/{len(x)}", "p_one_sided": p})
    if tests:
        tdf = pd.DataFrame(tests)
        tdf["p_holm"] = holm(tdf["p_one_sided"])
        tdf["better_holm_0.05"] = tdf["p_holm"] < 0.05
        tdf.round(5).to_csv(os.path.join(report, "table7b_vs_maslej_tests.csv"), index=False)

    # ---- table 8: per-label F1 vs Maslej (tuned models vs their Tables 17-18; others vs Tables 15-16) ----
    t8 = []
    for k in keys:
        fam = "recurrent" if k in RECURRENT_KEYS else "transformer"
        tuned = ALL_MODELS[k].get("stage") == "p4"
        ref, src = ((MASLEJ_PER_LABEL_TUNED_F1, "Table 17" if fam == "recurrent" else "Table 18") if tuned else
                    (MASLEJ_PER_LABEL_F1, "Table 15" if fam == "recurrent" else "Table 16"))
        for lab in LABELS:
            x = np.array([r["test"]["per_label"][lab]["f1"] for r in by_model[k]])
            t8.append({"model": label_of[k], "label": lab, "ours_mean": x.mean(),
                       "ours_sd": x.std(ddof=1) if len(x) > 1 else float("nan"),
                       "maslej": ref[fam][lab], "maslej_source": src, "difference": x.mean() - ref[fam][lab],
                       "seeds_above": f"{int((x > ref[fam][lab]).sum())}/{len(x)}"})
    if t8:
        pd.DataFrame(t8).round(4).to_csv(os.path.join(report, "table8_per_label_vs_maslej.csv"), index=False)

    print("\n" + "=" * 90)
    print("TABLE I - Overall test performance (mean ± std over runs)")
    print("=" * 90)
    print(t1.to_string())
    print("\nTABLE IV - Micro/macro precision, recall, F1" + (" (with the base paper)" if paper_names else ""))
    print(t4.to_string())

    # ---- figures -----------------------------------------------------------
    paper_color ={"LSTM (paper)": "#c9c9c9", "Bi-LSTM (paper)": "#a8a8a8", "GRU (paper)": "#888888"}

    # 1) macro-F1 bars
    names = paper_names + [label_of[k] for k in keys]
    means = [paper_metrics(n)["f1_macro"] for n in paper_names] + \
            [runs_df[runs_df.model == label_of[k]]["f1_macro"].mean() for k in keys]
    errs = [0] * len(paper_names) + \
           [np.nan_to_num(runs_df[runs_df.model == label_of[k]]["f1_macro"].std(ddof=1)) for k in keys]
    cols = [paper_color[n] for n in paper_names] + [BLUE.get(label_of[k], "#1f4e8c") for k in keys]
    plt.figure(figsize=(max(8, 0.9 * len(names)), 4.5))
    bars = plt.bar(names, means, yerr=errs, color=cols, capsize=4)
    for b, v in zip(bars, means):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontweight="bold")
    plt.ylabel("Macro F1 (test)"); plt.ylim(0, 1.0); plt.xticks(rotation=20, ha="right")
    plt.title("Macro-F1 (mean per label): recurrent (green), transformers (blue)"
              + ("; base paper (grey)" if paper_names else "") + "; error bars = std over runs", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(report, "fig_macro_f1_comparison.png"), dpi=200); plt.close()

    # 2) per-label F1 grouped bars
    allnames = paper_names + [label_of[k] for k in keys]
    width = 0.8 / len(allnames)
    plt.figure(figsize=(max(11, 1.2 * len(allnames)), 4.8))
    for i, n in enumerate(allnames):
        if n in PAPER_COUNTS:
            vals = [paper_metrics(n)["per_label"][l]["f1"] for l in LABELS]; err = None
            color = paper_color[n]
        else:
            k = next(kk for kk in keys if label_of[kk] == n)
            arr = np.array([[r["test"]["per_label"][l]["f1"] for l in LABELS] for r in by_model[k]])
            vals = arr.mean(0); err = arr.std(0, ddof=1) if len(arr) > 1 else None
            color = BLUE.get(n, "#1f4e8c")
        plt.bar(np.arange(6) + i * width - 0.4 + width / 2, vals, width, yerr=err,
                capsize=2, label=n, color=color)
    plt.xticks(range(6), LABELS); plt.ylabel("F1 (test)"); plt.ylim(0, 1.0)
    plt.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    plt.title("Per-label F1")
    plt.tight_layout(); plt.savefig(os.path.join(report, "fig_per_label_f1.png"), dpi=200); plt.close()

    # 3) run-to-run variability
    fig, axes = plt.subplots(1, 2, figsize=(max(9, 1.6 * len(keys)), 4))
    for ax, m in zip(axes, ["f1_macro", "f1_micro"]):
        sns.boxplot(data=runs_df, x="model", y=m, ax=ax, color="#dbe7f5", width=0.5)
        sns.stripplot(data=runs_df, x="model", y=m, ax=ax, color="#1f4e8c", size=7)
        ax.set_title(NICE[m] + " across runs"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=20)
    plt.tight_layout(); plt.savefig(os.path.join(report, "fig_runs_boxplot.png"), dpi=200); plt.close()

    # 4) mean confusion matrices (paper style, one figure per model)
    for k in keys:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
        for ax, lab in zip(axes.ravel(), LABELS):
            cm = np.mean([[[r["test"]["per_label"][lab]["tn"], r["test"]["per_label"][lab]["fp"]],
                           [r["test"]["per_label"][lab]["fn"], r["test"]["per_label"][lab]["tp"]]]
                          for r in by_model[k]], axis=0)
            sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["Pred. neg", "Pred. pos"], yticklabels=["Actual neg", "Actual pos"])
            ax.set_title(f"{label_of[k]} - {lab}", fontsize=10)
        plt.suptitle(f"{label_of[k]}: confusion matrices (mean over {len(by_model[k])} runs)")
        plt.tight_layout(); plt.savefig(os.path.join(report, f"fig_confusion_{k}.png"), dpi=200); plt.close()

    # 5) ROC curves (first finished run of each model)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (j, lab) in zip(axes.ravel(), enumerate(LABELS)):
        for k in keys:
            npz = os.path.join(by_model[k][0]["_dir"], "test_outputs.npz")
            if not os.path.isfile(npz):
                continue
            d = np.load(npz)
            y, p = d["y_true"][:, j], d["probs"][:, j]
            if y.min() == y.max():
                continue
            fpr, tpr, _ = roc_curve(y, p)
            auc = by_model[k][0]["test"]["per_label"][lab]["roc_auc"]
            ax.plot(fpr, tpr, label=f"{label_of[k]} (AUC={auc:.3f})", color=BLUE.get(label_of[k]))
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_title(lab); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.legend(fontsize=8, loc="lower right")
    plt.suptitle("Comparative ROC curves (first run of each model)")
    plt.tight_layout(); plt.savefig(os.path.join(report, "fig_roc_curves.png"), dpi=200); plt.close()

    print(f"\nAll tables and figures saved in: {report}/")


if __name__ == "__main__":
    main()

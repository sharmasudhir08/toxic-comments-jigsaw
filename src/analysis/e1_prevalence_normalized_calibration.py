"""
[APPROVED] item 1b from the 3 Nov 2026 task instruction.

Checks whether minority labels are better calibrated than majority labels even
after accounting for base-rate (prevalence) effects.

Method:
  - ECE per label (equal-width bins, 15 bins) -- raw, same as H8
  - Brier Skill Score (BSS) per label: BSS = 1 - BS_model / BS_ref
    BS_ref is the Brier score of always predicting the label prevalence.
    BSS > 0 means model adds skill beyond base-rate prediction.
    BSS near 0 means model's calibration is trivially explained by base rate.
  - Normalized ECE: ECE / prevalence -- removes the artifact that low-prevalence
    labels have low absolute ECE even when poorly calibrated relative to their rate.
  - Ranks labels by prevalence (minority = low prevalence) and checks whether
    the minority-better-calibrated pattern persists under BSS or normalized ECE.

Uses p4 / seeds 42-44 only (primary analysis protocol, same as H2/H8).
Read-only on saved test_outputs.npz.

Usage: python -m src.e1_prevalence_normalized_calibration
"""
import numpy as np
import pandas as pd
import os

MODELS = ["lstm", "bilstm", "gru", "bert", "distilbert", "roberta"]
SEEDS = [42, 43, 44]
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
RESULTS_DIR = "results_proposed/runs"
N_BINS = 15


def load_outputs(model, seed):
    d = np.load(os.path.join(RESULTS_DIR, f"{model}_p4", f"seed_{seed}", "test_outputs.npz"))
    return d["probs"], d["y_true"]


def ece_label(probs_1d, y_1d, n_bins=N_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_1d)
    for i in range(n_bins):
        mask = (probs_1d >= bins[i]) & (probs_1d < bins[i + 1])
        if i == n_bins - 1:
            mask = (probs_1d >= bins[i]) & (probs_1d <= bins[i + 1])
        if mask.sum() == 0:
            continue
        conf = probs_1d[mask].mean()
        acc = y_1d[mask].mean()
        ece += mask.sum() / n * abs(conf - acc)
    return ece


def brier_score(probs_1d, y_1d):
    return float(np.mean((probs_1d - y_1d.astype(float)) ** 2))


def brier_skill_score(probs_1d, y_1d, prevalence):
    bs_model = brier_score(probs_1d, y_1d)
    # Reference: always predict prevalence
    bs_ref = float(np.mean((prevalence - y_1d.astype(float)) ** 2))
    if bs_ref == 0:
        return np.nan
    return 1.0 - bs_model / bs_ref


def main():
    # First compute prevalences from all seeds (consistent test set)
    probs0, y0 = load_outputs("lstm", 42)
    prevalences = y0.mean(axis=0)  # shape (6,)
    print("Label prevalences (test set):")
    for li, label in enumerate(LABELS):
        print(f"  {label}: {prevalences[li]:.4f}")

    rows = []
    for model in MODELS:
        for seed in SEEDS:
            probs, y = load_outputs(model, seed)
            assert np.array_equal(y, y0), f"mismatched test labels for {model} seed {seed}"
            for li, label in enumerate(LABELS):
                prev = float(prevalences[li])
                p1d = probs[:, li].astype(float)
                y1d = y[:, li].astype(int)
                ece = ece_label(p1d, y1d)
                bss = brier_skill_score(p1d, y1d, prev)
                norm_ece = ece / prev if prev > 0 else np.nan
                rows.append({
                    "model": model, "seed": seed, "label": label,
                    "prevalence": round(prev, 5),
                    "ece": round(ece, 6),
                    "norm_ece": round(norm_ece, 5) if not np.isnan(norm_ece) else np.nan,
                    "bss": round(bss, 5),
                })

    df = pd.DataFrame(rows)

    # Aggregate: mean across seeds per model/label
    agg = df.groupby(["model", "label"]).agg(
        prevalence=("prevalence", "first"),
        ece_mean=("ece", "mean"),
        norm_ece_mean=("norm_ece", "mean"),
        bss_mean=("bss", "mean"),
    ).reset_index()
    agg = agg.sort_values(["model", "prevalence"])

    # Classify labels
    median_prev = float(np.median(prevalences))
    agg["is_minority"] = agg["prevalence"] < median_prev

    print("\n--- Per-model: minority vs majority mean BSS ---")
    for model in MODELS:
        sub = agg[agg["model"] == model]
        min_bss = sub[sub["is_minority"]]["bss_mean"].mean()
        maj_bss = sub[~sub["is_minority"]]["bss_mean"].mean()
        print(f"  {model}: minority BSS={min_bss:.4f}  majority BSS={maj_bss:.4f}  "
              f"minority_higher={'YES' if min_bss > maj_bss else 'NO'}")

    print("\n--- Per-model: minority vs majority mean norm_ECE ---")
    for model in MODELS:
        sub = agg[agg["model"] == model]
        min_ne = sub[sub["is_minority"]]["norm_ece_mean"].mean()
        maj_ne = sub[~sub["is_minority"]]["norm_ece_mean"].mean()
        print(f"  {model}: minority norm_ECE={min_ne:.4f}  majority norm_ECE={maj_ne:.4f}  "
              f"minority_lower={'YES' if min_ne < maj_ne else 'NO'}")

    # Summary table: per-label, averaged over models
    label_summary = agg.groupby("label").agg(
        prevalence=("prevalence", "first"),
        ece_mean=("ece_mean", "mean"),
        norm_ece_mean=("norm_ece_mean", "mean"),
        bss_mean=("bss_mean", "mean"),
        is_minority=("is_minority", "first"),
    ).reset_index().sort_values("prevalence")

    print("\n--- Per-label summary (averaged over 6 models) ---")
    print(label_summary.round(5).to_string(index=False))

    # Verdict
    n_minority_better_bss = sum(
        agg[agg["label"] == lab]["bss_mean"].mean() >
        agg[agg["label"] != lab]["bss_mean"].mean()
        for lab in agg[agg["is_minority"]]["label"].unique()
    )
    minority_labels = label_summary[label_summary["is_minority"]]["label"].tolist()
    majority_labels = label_summary[~label_summary["is_minority"]]["label"].tolist()
    print(f"\nMinority labels (prevalence < {median_prev:.4f}): {minority_labels}")
    print(f"Majority labels: {majority_labels}")

    min_bss_overall = label_summary[label_summary["is_minority"]]["bss_mean"].mean()
    maj_bss_overall = label_summary[~label_summary["is_minority"]]["bss_mean"].mean()
    min_ne_overall = label_summary[label_summary["is_minority"]]["norm_ece_mean"].mean()
    maj_ne_overall = label_summary[~label_summary["is_minority"]]["norm_ece_mean"].mean()

    print(f"\nOverall: minority mean BSS={min_bss_overall:.4f}  majority mean BSS={maj_bss_overall:.4f}")
    print(f"Overall: minority mean norm_ECE={min_ne_overall:.4f}  majority mean norm_ECE={maj_ne_overall:.4f}")

    if min_bss_overall > maj_bss_overall:
        print("\nVERDICT: Minority labels show HIGHER BSS than majority labels.")
        print("  => Better calibration is NOT a base-rate artifact.")
        print("  => E1 finding survives prevalence normalization.")
        print("  => Remove base-rate caveat; report normalized numbers.")
    else:
        print("\nVERDICT: Minority labels do NOT show higher BSS than majority labels.")
        print("  => Better raw ECE for minority is a base-rate artifact.")
        print("  => Demote E1 from main findings to caveated minor finding.")

    # Save
    out = "results_proposed/report/table_e1_prevalence_normalized_calibration.csv"
    agg.round(6).to_csv(out, index=False)
    label_summary.round(6).to_csv(
        "results_proposed/report/table_e1_label_summary_calibration.csv", index=False)
    print(f"\nSaved: {out}")
    print("Saved: results_proposed/report/table_e1_label_summary_calibration.csv")


if __name__ == "__main__":
    main()

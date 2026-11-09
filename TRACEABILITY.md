# Traceability: paper item -> result file -> script -> training runs

Every number in `results/` is produced by exactly one script in `src/analysis/` (or
`generate_results.py`), reading already-saved model outputs from the training runs in Step 1 of
`README.md`. No number here is hand-typed or hand-edited. See `results/MAPPING.csv` for the same
information as a machine-readable CSV.

| Paper item | Result file | Script | Training runs needed |
|---|---|---|---|
| Table 1 (overall performance, incl. PR-AUC) | `Table01_overall_performance.csv` | `generate_results.py --results_dir results_proposed` | all 6 models, seeds 42-44 |
| Table 2 (H1 primary, nested ANOVA) | `Table02_h1_nested_anova.csv` | `python -m src.analysis.family_comparison` | all 6 models, seeds 42-44 |
| Table 3 (per-label F1) | `Table03_per_label_f1.csv` | `generate_results.py --results_dir results_proposed` | all 6 models, seeds 42-44 |
| Table 4 (H2 Tukey+Holm, DeLong) | `Table04a/b_...csv` | `python -m src.analysis.h2_full_methods` | all 6 models, seeds 42-44 |
| Table 4 (H2 paired bootstrap) | `Table04c_h2_bootstrap.csv` | `python -m src.analysis.h2_full_methods` | all 6 models, seeds 42-44 |
| Table 5 (H2 paired-by-seed) | `Table05_h2_paired_by_seed.csv` | `python -m src.analysis.h2_paired_by_seed` | all 6 models, seeds 42-44 |
| Table 6 (H3a/H3b vs. benchmark) | `Table06a_h3_vs_benchmark.csv` | `python -m src.analysis.h3_h4_vs_maslej_tuned_comparator` | all 6 models, seeds 42-44 |
| Table 6 (H3 within-family Holm, primary) | `Table06b_h3_within_family_holm.csv` | `python -m src.analysis.paper2_posthoc_corrections` | (reads Table06a's output) |
| Table 6 (H4 per-label input) | `Table06c_h4_per_label_vs_benchmark.csv` | `generate_results.py --results_dir results_proposed` | all 6 models, seeds 42-44 |
| Table 6 (H4 margins, with/without threat) | `Table06d_h4_margins_with_without_threat.csv` | `python -m src.analysis.paper2_posthoc_corrections` | (reads Table06c's output) |
| Table 7 (H5 McNemar+Fisher) | `Table07a_h5_mcnemar_fisher.csv` | `python -m src.analysis.single_vs_multi_seed` | all 6 models, seeds 42-44 |
| Table 7 (H5 permutation) | `Table07b/c_...csv` | `python -m src.analysis.h5_paired_permutation` | all 6 models, seeds 42-44 |
| Table 8 (E1 prevalence-normalised calibration) | `Table08_e1_prevalence_calibration.csv` | `python -m src.analysis.e1_prevalence_normalized_calibration` | all 6 models, seeds 42-44 |
| Fig. 1 (workflow) | `figures/Fig1.png` | extracted directly from the final manuscript; no source script in this repository |
| Fig. 2 (recurrent architecture) | `figures/Fig2.png` | extracted directly from the final manuscript; no source script in this repository |
| Fig. 3 (transformer architecture) | `figures/Fig3.png` | extracted directly from the final manuscript; no source script in this repository |
| Fig. 4 (overall performance) | `figures/Fig4.png` | extracted directly from the final manuscript; no source script in this repository |
| Fig. 5 (calibration) | `figures/Fig5.png` | extracted directly from the final manuscript; no source script in this repository |
| Fig. 6 (length robustness) | `figures/Fig6.png` | extracted directly from the final manuscript; no source script in this repository |

## Prose numbers (not a numbered table, but reported and re-runnable)

| Where in the paper | Result file | Script |
|---|---|---|
| Sec. 4.6 (near-duplicate/near-exact counts) | `Supp_h7_duplicate_screen_summary.csv` | `python -m src.analysis.duplicate_audit` |
| Sec. 4.6 (H1 primary test repeated on the screened set) | `Supp_h1_screened_nested_anova.csv` | `python -m src.analysis.paper2_posthoc_corrections` |
| Sec. 4.6/5.3 (per-length-bin nested ANOVA, raw p) | `Supp_h7_length_control_raw.csv` | `python -m src.analysis.h7_length_control_nested_anova` |
| Sec. 4.6/5.3 (per-length-bin Holm over 10 tests, primary) | `Supp_h7_length_control_holm10.csv` | `python -m src.analysis.paper2_posthoc_corrections` |
| Sec. 5.3 (word/subword truncation mismatch) | `Supp_truncation_mismatch.csv` | `python -m src.analysis.check_truncation_mismatch` |
| Sec. 5.1 (seed-sensitivity CV ratios) | `Supp_h9_seed_stability_cv.csv` | `python -m src.analysis.seed_stability` |

## Honest gaps

Two result files currently have saved data but no standalone, re-runnable script in this
repository:

- **`Supp_h7_family_screened_welch.csv`** — the Welch secondary check on the duplicate-screened
  set (Sec. 4.6: screened macro-F1(M) difference +0.0414 vs. +0.0415 on the full set, Welch
  p=0.0004). The primary nested-ANOVA repeat on the same screened set (`Supp_h1_screened_nested_anova.csv`)
  is reproducible; this specific Welch-test file's generating script was not found anywhere in the
  project during this package's preparation.
- **`Supp_h8_raw_ece_minority_vs_frequent.csv`** — the raw-ECE (not prevalence-normalised)
  minority-vs-frequent one-sided test, Holm-corrected (Sec. 5.1: ratio 3.80-5.44x, minority labels
  better calibrated by raw ECE; this is the finding Sec. 5.1 goes on to qualify using Table 8's
  prevalence-normalised BSS instead). No generating script was found anywhere in the project.

Both are flagged here rather than silently omitted or backed by a script written after the fact
that might not exactly match the saved numbers — the same disclosure convention used throughout
this project's own result logs.

## No reproduction claim without the saved model outputs

None of the scripts above can be run without the saved per-model, per-seed test-set probabilities
(and, for the duplicate screen, a precomputed near-duplicate mask against the training set). These
are not included in this repository (see `README.md`); request them from the authors, or retrain
from Step 1 of `README.md`, to actually execute the Step 2 reproduction.

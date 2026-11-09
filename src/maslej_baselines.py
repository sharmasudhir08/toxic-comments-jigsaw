"""
Published results of Maslej-Krešňáková, Sarnovský, Butka & Machová (2020), "Comparison of Deep Learning Models
and Various Text Pre-Processing Techniques for the Toxic Comments Classification", Applied Sciences 10(23):8631,
doi: 10.3390/app10238631. Same official Jigsaw test set as this project (63,978 scored comments), single runs.

Their macro-F1 is F1 of macro-precision and macro-recall (paper Sec. 5) = our "f1_macro_m".
"accuracy" is the accuracy they report (it matches exact-match accuracy in magnitude).
The benchmark targets are fixed in preregistration.md v4, Sec. 4.1. Full extraction: reference.md Sec. A2.1.
"""

from src.config import LABELS

MASLEJ_REF = "Maslej-Krešňáková et al. (2020), Appl. Sci. 10:8631"

# Overall results (untuned unless stated). Keys follow src/metrics.py.
MASLEJ_OVERALL = {
    "BiLSTM+CNN, GloVe Twitter (Maslej best recurrent)": {
        "family": "recurrent", "source": "Table 9", "exact_match_accuracy": 0.8787, "roc_auc_macro": 0.9798,
        "precision_micro": 0.59, "recall_micro": 0.77, "f1_micro": 0.67,
        "precision_macro": 0.57, "recall_macro": 0.59, "f1_macro_m": 0.56},
    "BiLSTM+CNN, fastText (Maslej)": {
        "family": "recurrent", "source": "Table 9", "exact_match_accuracy": 0.8839, "roc_auc_macro": 0.9787,
        "precision_micro": 0.61, "recall_micro": 0.73, "f1_micro": 0.66,
        "precision_macro": 0.58, "recall_macro": 0.52, "f1_macro_m": 0.51},
    "GRU 128, GloVe (Maslej)": {
        "family": "recurrent", "source": "Table 4", "exact_match_accuracy": 0.872, "f1_micro": 0.66,
        "f1_macro_m": 0.56},
    "LSTM 128, GloVe (Maslej)": {
        "family": "recurrent", "source": "Table 4", "exact_match_accuracy": 0.873, "f1_micro": 0.65,
        "f1_macro_m": 0.55},
    "BERT-base uncased, fine-tuned (Maslej best transformer)": {
        "family": "transformer", "source": "Table 13", "exact_match_accuracy": 0.8971, "roc_auc_macro": 0.9842,
        "precision_micro": 0.67, "recall_micro": 0.72, "f1_micro": 0.69,
        "precision_macro": 0.69, "recall_macro": 0.51, "f1_macro_m": 0.55},
    "DistilBERT uncased (Maslej)": {
        "family": "transformer", "source": "Table 10", "exact_match_accuracy": 0.8649, "roc_auc_macro": 0.9781,
        "precision_micro": 0.59, "recall_micro": 0.73, "f1_micro": 0.65,
        "precision_macro": 0.48, "recall_macro": 0.39, "f1_macro_m": 0.34},
}

# Benchmark targets per family (preregistration v4, Sec. 4.1)
MASLEJ_TARGET = {"recurrent": "BiLSTM+CNN, GloVe Twitter (Maslej best recurrent)",
                 "transformer": "BERT-base uncased, fine-tuned (Maslej best transformer)"}

# Per-label F1 after per-class threshold optimisation (Tables 17 and 18) -> compare with our tuned (P4) models
MASLEJ_PER_LABEL_TUNED_F1 = {
    "recurrent": dict(zip(LABELS, [0.70, 0.44, 0.69, 0.34, 0.68, 0.55])),     # Table 17
    "transformer": dict(zip(LABELS, [0.71, 0.41, 0.71, 0.47, 0.69, 0.65])),   # Table 18
}
# Per-label F1 at threshold 0.5 (Tables 15 and 16) -> compare with our threshold-0.5 (P3) models
MASLEJ_PER_LABEL_F1 = {
    "recurrent": dict(zip(LABELS, [0.68, 0.41, 0.69, 0.34, 0.68, 0.54])),     # Table 15
    "transformer": dict(zip(LABELS, [0.70, 0.29, 0.69, 0.47, 0.69, 0.60])),   # Table 16
}

"""
Results of the BASE PAPER (LSTM / Bi-LSTM / GRU with Word2Vec), taken from its
confusion matrices (Fig. 4-6). Counts are (TN, FP, FN, TP) on the 63,978 scored
Jigsaw test comments, threshold 0.5. These are single runs (no std available).
Used only to put the transformer results next to the paper's in the tables/figures.
"""

from src.config import LABELS

PAPER_COUNTS = {
    "LSTM (paper)": {
        "toxic": (54452, 3436, 1318, 4772), "severe_toxic": (63497, 114, 271, 96),
        "obscene": (59002, 1285, 1130, 2561), "threat": (63765, 2, 211, 0),
        "insult": (59449, 1102, 1368, 2059), "identity_hate": (63244, 22, 693, 19),
    },
    "Bi-LSTM (paper)": {
        "toxic": (54437, 3451, 1207, 4883), "severe_toxic": (63375, 236, 242, 125),
        "obscene": (58917, 1370, 1090, 2601), "threat": (63703, 64, 135, 76),
        "insult": (59551, 1000, 1354, 2073), "identity_hate": (63141, 125, 497, 215),
    },
    "GRU (paper)": {
        "toxic": (54661, 3227, 1309, 4781), "severe_toxic": (63447, 164, 252, 115),
        "obscene": (58525, 1762, 905, 2786), "threat": (63669, 98, 141, 70),
        "insult": (59472, 1079, 1294, 2133), "identity_hate": (63165, 101, 514, 198),
    },
}


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def paper_metrics(name):
    counts = PAPER_COUNTS[name]
    per_label, ps, rs, fs = {}, [], [], []
    TP = FP = FN = 0
    for lab in LABELS:
        tn, fp, fn, tp = counts[lab]
        p, r, f = _prf(tp, fp, fn)
        per_label[lab] = {"precision": p, "recall": r, "f1": f,
                          "accuracy": (tp + tn) / (tp + tn + fp + fn),
                          "tn": tn, "fp": fp, "fn": fn, "tp": tp}
        ps.append(p); rs.append(r); fs.append(f)
        TP += tp; FP += fp; FN += fn
    pm, rm, fm = _prf(TP, FP, FN)
    return {"precision_micro": pm, "recall_micro": rm, "f1_micro": fm,
            "precision_macro": sum(ps) / 6, "recall_macro": sum(rs) / 6, "f1_macro": sum(fs) / 6,
            "per_label": per_label}

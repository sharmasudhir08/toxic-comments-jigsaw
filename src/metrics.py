"""Multi-label metrics (no torch needed)."""

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, hamming_loss,
                             precision_recall_fscore_support, roc_auc_score)

from src.config import LABELS, THRESHOLD


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _auc(y, p, average=None):
    try:
        return float(roc_auc_score(y, p, average=average))
    except ValueError:          # only one class present (tiny smoke-test subsets)
        return float("nan")


def _mcc(tp, tn, fp, fn):
    den = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    return float((tp * tn - fp * fn) / den) if den else 0.0


def _ece(y, p, bins=10):
    """Expected calibration error, equal-width bins: sum_m |B_m|/n * |mean(y in B_m) - mean(p in B_m)|."""
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == m].mean() - p[idx == m].mean()) * (idx == m).mean()
                     for m in range(bins) if (idx == m).any()))


def _pr_auc(y, p):
    """Area under the precision-recall curve (average precision); nan if the label has no positives."""
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if y.sum() else float("nan")


def multilabel_metrics(y_true, probs, threshold=THRESHOLD):
    """
    y_true: (n, 6) 0/1 array.  probs: (n, 6) sigmoid probabilities.
    threshold: one value, or one value per label (array of 6, e.g. tuned on validation).
    Returns overall + per-label metrics as plain Python floats.
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    y_pred = (probs >= threshold).astype(int)

    out = {}
    for avg in ("micro", "macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0)
        out[f"precision_{avg}"], out[f"recall_{avg}"], out[f"f1_{avg}"] = float(p), float(r), float(f)

    out["exact_match_accuracy"] = float((y_true == y_pred).all(axis=1).mean())  # strict, all 6 labels
    out["label_accuracy_mean"] = float((y_true == y_pred).mean())
    out["hamming_loss"] = float(hamming_loss(y_true, y_pred))

    per_label, aucs, pr_aucs = {}, [], []
    for j, name in enumerate(LABELS):
        tn, fp, fn, tp = confusion_matrix(y_true[:, j], y_pred[:, j], labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        auc = _auc(y_true[:, j], probs[:, j])
        aucs.append(auc)
        pr_auc = _pr_auc(y_true[:, j], probs[:, j])
        pr_aucs.append(pr_auc)
        per_label[name] = {
            "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "roc_auc": auc, "pr_auc": pr_auc, "mcc": _mcc(tp, tn, fp, fn),
            "ece": _ece(y_true[:, j], probs[:, j]),
            "brier": float(np.mean((probs[:, j] - y_true[:, j]) ** 2)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "support": int(tp + fn),
        }
    out["per_label"] = per_label
    valid = [a for a in aucs if not np.isnan(a)]
    out["roc_auc_macro"] = float(np.mean(valid)) if valid else float("nan")
    out["roc_auc_micro"] = _auc(y_true, probs, average="micro") if y_true.sum() else float("nan")
    valid_pr = [a for a in pr_aucs if not np.isnan(a)]
    out["pr_auc_macro"] = float(np.mean(valid_pr)) if valid_pr else float("nan")
    # macro-F1 (M): F1 of macro-precision and macro-recall, the definition used by
    # Maslej-Krešňáková et al. (2020) and van Aken et al. (2018); f1_macro above = mean of per-label F1
    pm_, rm_ = out["precision_macro"], out["recall_macro"]
    out["f1_macro_m"] = float(2 * pm_ * rm_ / (pm_ + rm_)) if (pm_ + rm_) else 0.0
    for k in ("mcc", "ece", "brier"):
        out[f"{k}_macro"] = float(np.mean([per_label[l][k] for l in LABELS]))
    return out

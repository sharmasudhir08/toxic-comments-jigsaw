"""
Central list of the transformer models used in this project.
To add another one (e.g. ALBERT), add one line to MODELS -- training,
prediction, the web app and the results script all pick it up.
"""

import os

MODELS = {
    "bert":       {"checkpoint": "bert-base-uncased",       "label": "BERT"},
    "distilbert": {"checkpoint": "distilbert-base-uncased", "label": "DistilBERT"},
    "roberta":    {"checkpoint": "roberta-base",            "label": "RoBERTa"},
}

# Recurrent models of the base paper (Word2Vec -> frozen Embedding -> RNN(64) -> Dense(6, sigmoid)),
# trained by src.train_lstm / src.train_bilstm / src.train_gru. Epochs as in the paper (Sec. III-E).
RNN_MODELS = {
    "lstm":   {"label": "LSTM",    "cell": "lstm", "bidirectional": False, "epochs": 15},
    "bilstm": {"label": "Bi-LSTM", "cell": "lstm", "bidirectional": True,  "epochs": 10},
    "gru":    {"label": "GRU",     "cell": "gru",  "bidirectional": False, "epochs": 10},
}

# Proposed improved recurrent models (experiment_plan.md, Sec. 9), trained by src.train_proposed_*.
# Ablation stages: P1 representation, P2 + attention/pooling head, P3 + class-weighted loss,
# P4 = P3 with per-label thresholds tuned on validation (no extra training).
PROPOSED_STAGES = ["p1", "p2", "p3", "p4"]
# Report labels (scope v3): the final models are shown as "LSTM (0.5)" / "LSTM (tuned)" etc.;
# P1/P2 (optional ablation, not used by default) keep their stage names.
STAGE_SUFFIX = {"p1": "-P1", "p2": "-P2", "p3": " (0.5)", "p4": " (tuned)"}
RNN_PROPOSED = {f"{k}_{s}": {"label": f"{v['label']}{STAGE_SUFFIX[s]}", "base": k, "stage": s}
                for k, v in RNN_MODELS.items() for s in PROPOSED_STAGES}

# Transformers under the common protocol (src/train.py --protocol common): P3 = threshold 0.5,
# P4 = per-label thresholds tuned on validation (same training run), as for the recurrent models.
TRANSFORMER_COMMON = {f"{k}_{s}": {"label": f"{v['label']}{STAGE_SUFFIX[s]}", "base": k, "stage": s}
                      for k, v in MODELS.items() for s in ["p3", "p4"]}


def model_dir(key: str) -> str:
    """Folder where the fine-tuned model for `key` is saved (with --save_model)."""
    return os.path.join("models", key)


def is_trained(key: str) -> bool:
    d = model_dir(key)
    return os.path.isdir(d) and any(f.endswith((".safetensors", ".bin")) for f in os.listdir(d))


def available_models() -> list:
    return [k for k in MODELS if is_trained(k)]

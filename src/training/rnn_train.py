"""
Train ONE recurrent model of the base paper (LSTM, Bi-LSTM or GRU) for ONE seed.
Normally you do not call this directly -- use  python -m src.train_lstm / train_bilstm / train_gru.

Architecture and training, as in the paper (Sec. III-E), written in PyTorch to match Keras:
    Embedding(vocab_size, 100, weights=[Word2Vec matrix], trainable=False)
    LSTM(64) | Bidirectional(LSTM(64)) | GRU(64),  return_sequences=False
    Dense(6, activation="sigmoid")
    Adam (Keras defaults: lr=1e-3, eps=1e-7), binary cross-entropy, batch_size=64,
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    epochs 15 (LSTM) / 10 (Bi-LSTM, GRU), threshold 0.5.
Keras initialisation is reproduced (glorot_uniform kernels, orthogonal recurrent weights,
zero biases, LSTM forget-gate bias 1).

The train/validation/test split comes from src.data.get_splits, so every seed uses exactly
the same comments as the transformer run with that seed.

Per run it saves, under  <out_dir>/runs/<model>/seed_<seed>/  (same format as src/train.py):
    metrics.json        validation + test metrics (overall and per label), config, history, timing
    test_outputs.npz    test probabilities + true labels (for ROC curves / confusion matrices)
"""

import argparse
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import LABELS, MAX_LEN, SEEDS
from src.data import get_splits
from src.metrics import multilabel_metrics, sigmoid
from src.model_registry import RNN_MODELS
from src.training.rnn_preprocessing import load_cache


class RecurrentClassifier(nn.Module):
    def __init__(self, embedding_matrix, cell="lstm", units=64, bidirectional=False, n_labels=len(LABELS)):
        super().__init__()
        self.cell = cell
        self.embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(embedding_matrix, dtype=torch.float32), freeze=True)   # trainable=False
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(embedding_matrix.shape[1], units, batch_first=True, bidirectional=bidirectional)
        self.out = nn.Linear(units * (2 if bidirectional else 1), n_labels)
        self._keras_init(units)

    def _keras_init(self, units):
        for name, p in self.rnn.named_parameters():
            if "weight_ih" in name:
                for g in p.data.chunk(p.shape[0] // units):     # one glorot matrix per gate
                    nn.init.xavier_uniform_(g)
            elif "weight_hh" in name:
                for g in p.data.chunk(p.shape[0] // units):
                    nn.init.orthogonal_(g)
            elif "bias" in name:
                nn.init.zeros_(p)
                if self.cell == "lstm" and "bias_ih" in name:
                    p.data[units:2 * units] = 1.0            # unit_forget_bias=True (gates i, f, g, o)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        _, h = self.rnn(self.embedding(x))
        if self.cell == "lstm":
            h = h[0]
        # h: (directions, batch, units) -> final forward state [+ final backward state], like Keras
        return self.out(torch.cat(list(h), dim=-1))


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    # Apple MPS was measured slower than CPU for these small (64-unit) recurrent layers, so it is opt-in
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, X, device, batch_size=512):
    """Logits for all rows of X (sigmoid -> probabilities)."""
    model.eval()
    logits = [model(torch.from_numpy(X[i:i + batch_size]).long().to(device)).float().cpu()
              for i in range(0, len(X), batch_size)]
    return torch.cat(logits).numpy() if logits else np.zeros((0, len(LABELS)), np.float32)


def bce(logits, y):
    return float(nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(logits), torch.from_numpy(y.astype(np.float32))))


def rows_for(ids, id_to_row, what):
    missing = [i for i in ids if i not in id_to_row]
    if missing:
        raise ValueError(f"{len(missing)} {what} comments are not in the preprocessing cache "
                         "(was it built from a different data/ folder?). Rebuild it with --overwrite.")
    return np.array([id_to_row[i] for i in ids])


def run_one(key, seed, data_dir="data", cache_dir="data/rnn_cache", out_dir="results", epochs=None,
            batch_size=64, lr=1e-3, max_len=MAX_LEN, val_size=0.1, patience=3, max_train_samples=None,
            max_test_samples=None, overwrite=False, device="auto"):
    cfg = RNN_MODELS[key]
    epochs = epochs or cfg["epochs"]
    run_dir = os.path.join(out_dir, "runs", key, f"seed_{seed}")
    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.isfile(metrics_path) and not overwrite:
        print(f"[skip] {cfg['label']} seed {seed}: already done ({metrics_path})")
        with open(metrics_path) as f:
            return json.load(f)
    os.makedirs(run_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"{cfg['label']} (Word2Vec + {cfg['label']}, 64 units)  |  seed {seed}")
    print("=" * 72)
    t0 = time.time()
    set_all_seeds(seed)
    dev = pick_device(device)

    seq, matrix, meta = load_cache(cache_dir)
    if max_len > meta["max_len"]:
        raise ValueError(f"--max_len {max_len} > cached length {meta['max_len']}; rebuild the cache "
                         f"with  python -m src.rnn_preprocessing --max_len {max_len} --overwrite")
    train_df, val_df, test_df, test_source = get_splits(
        data_dir, seed, val_size, max_train_samples, max_test_samples)
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}  [{test_source}]  device={dev}")

    # sequences are pre-padded/pre-truncated, so the last max_len columns == pad_sequences(maxlen=max_len)
    train_rows = {i: r for r, i in enumerate(seq["train_ids"])}
    test_rows = {i: r for r, i in enumerate(seq["test_ids"])}
    X_all, X_test_all = seq["X_train"][:, -max_len:], seq["X_test"][:, -max_len:]
    X_tr = X_all[rows_for(train_df["id"].astype(str), train_rows, "train")]
    X_va = X_all[rows_for(val_df["id"].astype(str), train_rows, "validation")]
    if test_source.startswith("official"):
        X_te = X_test_all[rows_for(test_df["id"].astype(str), test_rows, "test")]
    else:                                  # hold-out of train.csv
        X_te = X_all[rows_for(test_df["id"].astype(str), train_rows, "test")]
    y_tr, y_va = train_df[LABELS].values.astype(np.float32), val_df[LABELS].values.astype(np.float32)
    y_te = test_df[LABELS].values.astype(int)

    model = RecurrentClassifier(matrix, cfg["cell"], 64, cfg["bidirectional"]).to(dev)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, eps=1e-7)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)

    best_loss, best_state, best_epoch, wait, history = float("inf"), None, 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        perm, total, n = rng.permutation(len(X_tr)), 0.0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb = torch.from_numpy(X_tr[idx]).long().to(dev)
            yb = torch.from_numpy(y_tr[idx]).to(dev)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx); n += len(idx)
        val_loss = bce(predict(model, X_va, dev), y_va)
        history.append({"epoch": epoch, "loss": total / n, "val_loss": val_loss})
        print(f"  epoch {epoch:2d}/{epochs}: loss={total / n:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stopping (no val_loss improvement for {patience} epochs)")
                break
    model.load_state_dict(best_state)                 # restore_best_weights=True
    print(f"  restored best epoch {best_epoch} (val_loss={best_loss:.4f})")

    val_logits = predict(model, X_va, dev)
    val_m = multilabel_metrics(y_va, sigmoid(val_logits))
    probs = sigmoid(predict(model, X_te, dev))
    test_metrics = multilabel_metrics(y_te, probs)

    result = {
        "model": key, "model_label": cfg["label"],
        "checkpoint": f"word2vec-{meta['word2vec']['backend']}+{key}", "seed": seed,
        "test_source": test_source,
        "sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "config": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "max_len": max_len,
                   "val_size": val_size, "patience": patience, "units": 64,
                   "bidirectional": cfg["bidirectional"], "embedding": "Word2Vec (frozen)",
                   "word2vec": meta["word2vec"], "vocab_size": meta["vocab_size"],
                   "cache_dir": cache_dir, "device": str(dev)},
        "val": {"loss": bce(val_logits, y_va), "best_epoch": best_epoch,
                **{k: val_m[k] for k in ("f1_micro", "f1_macro", "precision_macro", "recall_macro",
                                         "exact_match_accuracy", "roc_auc_macro")}},
        "history": history,
        "test": test_metrics,
        "train_minutes": round((time.time() - t0) / 60, 2),
    }
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(os.path.join(run_dir, "test_outputs.npz"),
                        probs=probs.astype(np.float32), y_true=y_te.astype(np.int8))

    print(f"\n{cfg['label']} seed {seed}: TEST  micro-F1={test_metrics['f1_micro']:.4f}  "
          f"macro-F1={test_metrics['f1_macro']:.4f}  AUC={test_metrics['roc_auc_macro']:.4f}  "
          f"exact-match={test_metrics['exact_match_accuracy']:.4f}  "
          f"({result['train_minutes']} min)")
    return result


def main(key):
    """Command line shared by train_lstm.py, train_bilstm.py and train_gru.py."""
    cfg = RNN_MODELS[key]
    ap = argparse.ArgumentParser(description=f"{cfg['label']} (base paper) on Jigsaw, multi-label, 5 seeds")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="same seeds as the transformers")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--cache_dir", default=None, help="default: data/rnn_cache (data/rnn_cache_smoke with --smoke)")
    ap.add_argument("--out_dir", default=None, help="default: results (results_smoke with --smoke)")
    ap.add_argument("--epochs", type=int, default=cfg["epochs"], help="paper: 15 LSTM, 10 Bi-LSTM/GRU")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3, help="Keras Adam default")
    ap.add_argument("--max_len", type=int, default=MAX_LEN)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=3, help="early-stopping patience (epochs), paper: 3")
    ap.add_argument("--max_train_samples", type=int, default=None,
                    help="use only N training comments (must match the transformer runs you compare with)")
    ap.add_argument("--max_test_samples", type=int, default=None)
    ap.add_argument("--device", default="auto", help="auto (cuda, else cpu) | cpu | mps | cuda")
    ap.add_argument("--overwrite", action="store_true", help="redo runs that already finished")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny test: 1500 train / 500 test comments, 1 epoch, 2 seeds")
    a = ap.parse_args()

    if a.smoke:
        a.max_train_samples, a.max_test_samples = a.max_train_samples or 1500, a.max_test_samples or 500
        a.epochs, a.max_len = 1, 64
        if a.seeds == SEEDS:
            a.seeds = SEEDS[:2]
    cache_dir = a.cache_dir or os.path.join(a.data_dir, "rnn_cache_smoke" if a.smoke else "rnn_cache")
    out_dir = a.out_dir or ("results_smoke" if a.smoke else "results")
    print(f"{cfg['label']}   seeds={a.seeds}   cache={cache_dir}   out={out_dir}")

    for seed in a.seeds:
        run_one(key, seed, a.data_dir, cache_dir, out_dir, a.epochs, a.batch_size, a.lr, a.max_len,
                a.val_size, a.patience, a.max_train_samples, a.max_test_samples, a.overwrite, a.device)

    print(f"\n{cfg['label']}: all runs finished. Tables/figures:  python generate_results.py"
          + (f" --results_dir {out_dir}" if out_dir != "results" else ""))

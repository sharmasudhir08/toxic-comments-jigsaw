"""
Proposed recurrent models (experiment_plan.md, Sec. 9-11; preregistration.md v3).
Standard LSTM / Bi-LSTM / GRU, 1 layer, 128 units (--units), max length 128,
keep the first 128 words, pre-padding, no mask. Input = the common light cleaning
(src.preprocessing.clean_text_light, shared with the transformers); --cleaning v2 is still available.

Ablation stages (each adds one group to the previous one):
    P1  cleaning v2 + FastText crawl-300d-2M (300-d, fine-tuned) + SpatialDropout 0.3
        + Adam/ReduceLROnPlateau/clipping, batch 128, max 15 epochs, early stopping on val macro-F1
    P2  P1 + attention + global max + global average pooling, Dense(64, ReLU) + Dropout 0.2 head
    P3  P2 + class-weighted BCE (pos_weight = sqrt(neg/pos), capped at 10)
    P4  P3 models with per-label thresholds tuned on the validation set (no extra training)

Step 1 - preprocessing (once; needs data/embeddings/crawl-300d-2M.vec):
    python -m src.rnn_proposed            # -> data/rnn_cache_proposed/
    python -m src.rnn_proposed --smoke    # -> data/rnn_cache_proposed_smoke/ (first 200k vectors only)
Step 2 - training: python -m src.train_proposed_lstm / train_proposed_bilstm / train_proposed_gru

Per run, under <out_dir>/runs/<model>_<stage>/seed_<seed>/ (same format as src/rnn_train.py):
    metrics.json, test_outputs.npz, and val_outputs.npz (validation probabilities, used for P4)
"""

import argparse
import copy
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import LABELS, MAX_LEN, SEEDS
from src.data import get_splits, load_official_test, load_train
from src.metrics import multilabel_metrics, sigmoid
from src.model_registry import RNN_MODELS, RNN_PROPOSED, TRANSFORMER_COMMON
from src.preprocessing import clean_text_light
from src.training.rnn_preprocessing import fit_tokenizer, load_cache
from src.training.rnn_train import bce, pick_device, predict, rows_for, set_all_seeds

LABEL_OF = {k: v["label"] for k, v in {**RNN_PROPOSED, **TRANSFORMER_COMMON}.items()}

FASTTEXT_PATH = os.path.join("data", "embeddings", "crawl-300d-2M.vec")
EMB_DIM = 300

# ---------------------------------------------------------------- cleaning v2 (row 1)
_CONTRACTIONS = [(r"won't", "will not"), (r"can't", "can not"), (r"n't\b", " not"), (r"'re\b", " are"),
                 (r"'m\b", " am"), (r"'ll\b", " will"), (r"'ve\b", " have"), (r"'d\b", " would"),
                 (r"'s\b", "")]


def clean_text_v2(text: str) -> str:
    """Lowercase, expand contractions, keep runs of ! / ? as one token, punctuation/numbers -> space."""
    t = str(text).lower().replace("’", "'")
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    for pat, rep in _CONTRACTIONS:
        t = re.sub(pat, rep, t)
    t = re.sub(r"!+", " ! ", t)
    t = re.sub(r"\?+", " ? ", t)
    t = re.sub(r"[^a-z!?\s]", " ", t)      # punctuation, special characters and numbers -> space (not deleted)
    return re.sub(r"\s+", " ", t).strip()


def tokenize_v2(texts) -> list:
    from nltk.tokenize import word_tokenize
    return [word_tokenize(clean_text_v2(t), preserve_line=True) for t in texts]


def tokenize_texts(texts, cleaning="light") -> list:
    """light (default, common protocol): URLs removed, case/punctuation/numbers kept, NLTK tokens."""
    if cleaning == "v2":
        return tokenize_v2(texts)
    from nltk.tokenize import word_tokenize
    return [word_tokenize(clean_text_light(t), preserve_line=True) for t in texts]


def texts_to_padded_first(token_lists, word_index, max_len) -> np.ndarray:
    """Keep the FIRST max_len known words (as the transformers' head truncation), pre-padding, no mask."""
    X = np.zeros((len(token_lists), max_len), dtype=np.int32)
    for i, toks in enumerate(token_lists):
        seq = [word_index[w] for w in toks if w in word_index][:max_len]
        if seq:
            X[i, -len(seq):] = seq
    return X


def _candidates(word):
    """FastText lookup order for a (cased) token: exact, lowercase, Capitalised."""
    return [word, word.lower(), word.capitalize()]


# ---------------------------------------------------------------- FastText embedding (rows 2-3)
def load_fasttext_cased(vocab, path=FASTTEXT_PATH, max_vectors=None):
    """Read the FastText vectors needed for a cased vocabulary (all lookup candidates of each word)."""
    if not os.path.isfile(path):
        load_fasttext(vocab, path)                      # raises the download instructions
    wanted = {c for w in vocab for c in _candidates(w)}
    vectors = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for n, line in enumerate(f):
            if max_vectors and n > max_vectors:
                break
            word, _, rest = line.rstrip().partition(" ")
            if word not in wanted or word in vectors or not rest:
                continue
            vec = np.asarray(rest.split(" "), dtype=np.float32)
            if vec.shape[0] == EMB_DIM:
                vectors[word] = vec
    return {w: next(vectors[c] for c in _candidates(w) if c in vectors)
            for w in vocab if any(c in vectors for c in _candidates(w))}


def load_fasttext(vocab, path=FASTTEXT_PATH, max_vectors=None):
    """Read only the vectors of words in `vocab`. The file is cased: the lowercase form wins,
    otherwise the first cased form seen (e.g. 'Wikipedia' for 'wikipedia')."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Download FastText crawl-300d-2M (see how_to_run.md):\n"
            "  curl -L -o data/embeddings/crawl-300d-2M.vec.zip "
            "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip\n"
            "  unzip data/embeddings/crawl-300d-2M.vec.zip -d data/embeddings/")
    wanted, vectors, exact = set(vocab), {}, set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        for n, line in enumerate(f):
            if max_vectors and n > max_vectors:
                break
            word, _, rest = line.rstrip().partition(" ")
            key = word.lower()
            if key not in wanted or key in exact or not rest:
                continue
            vec = np.asarray(rest.split(" "), dtype=np.float32)
            if vec.shape[0] != EMB_DIM:                       # header line "2000000 300"
                continue
            vectors[key] = vec
            if word == key:
                exact.add(key)
    return vectors


def build_matrix(vocab, vectors, seed):
    """Row 0 = padding (zeros). Words without a FastText vector: random normal with FastText's mean/std."""
    found = np.stack(list(vectors.values())) if vectors else np.zeros((1, EMB_DIM), np.float32)
    rng = np.random.default_rng(seed)
    matrix = rng.normal(found.mean(0), found.std(0), size=(len(vocab) + 1, EMB_DIM)).astype(np.float32)
    matrix[0] = 0.0
    for i, w in enumerate(vocab, start=1):
        if w in vectors:
            matrix[i] = vectors[w]
    return matrix


def prepare(a):
    cache_dir = a.cache_dir or os.path.join(a.data_dir, "rnn_cache_proposed_smoke" if a.smoke else "rnn_cache_proposed")
    if os.path.isfile(os.path.join(cache_dir, "meta.json")) and not a.overwrite:
        print(f"[skip] {cache_dir} already exists (use --overwrite to rebuild)")
        return
    os.makedirs(cache_dir, exist_ok=True)
    t0 = time.time()
    train_df, test_df = load_train(a.data_dir), load_official_test(a.data_dir)
    print(f"train={len(train_df)}  test={0 if test_df is None else len(test_df)}\n"
          f"Cleaning ({a.cleaning}) + tokenising ...")
    train_tokens = tokenize_texts(train_df["comment_text"], a.cleaning)
    test_tokens = [] if test_df is None else tokenize_texts(test_df["comment_text"], a.cleaning)

    vocab = fit_tokenizer(train_tokens)
    word_index = {w: i for i, w in enumerate(vocab, start=1)}
    max_vectors = a.max_vectors or (200_000 if a.smoke else None)
    print(f"Reading FastText vectors from {a.fasttext} {'(first %d lines)' % max_vectors if max_vectors else ''} ...")
    vectors = (load_fasttext(vocab, a.fasttext, max_vectors) if a.cleaning == "v2"
               else load_fasttext_cased(vocab, a.fasttext, max_vectors))
    matrix = build_matrix(vocab, vectors, a.seed)

    counts = {}
    for toks in train_tokens:
        for w in toks:
            counts[w] = counts.get(w, 0) + 1
    tok_total = sum(counts.values())
    tok_cov = sum(c for w, c in counts.items() if w in vectors) / max(tok_total, 1)
    np.savez_compressed(os.path.join(cache_dir, "sequences.npz"),
                        train_ids=np.asarray(train_df["id"].astype(str), dtype=str),
                        X_train=texts_to_padded_first(train_tokens, word_index, a.max_len),
                        test_ids=np.asarray([] if test_df is None else test_df["id"].astype(str), dtype=str),
                        X_test=texts_to_padded_first(test_tokens, word_index, a.max_len))
    np.save(os.path.join(cache_dir, "embedding_matrix.npy"), matrix)
    with open(os.path.join(cache_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    meta = {"max_len": a.max_len, "vocab_size": len(vocab) + 1, "embedding_dim": EMB_DIM,
            "word2vec": {"backend": "fasttext-crawl-300d-2M", "file": a.fasttext, "max_vectors_read": max_vectors},
            "cleaning": a.cleaning, "truncation": "keep first max_len words", "padding": "pre, no mask",
            "words_with_vector": len(vectors),
            "vocab_coverage": len(vectors) / len(vocab), "token_coverage": tok_cov, "smoke": a.smoke,
            "n_train": len(train_df), "n_test": len(test_tokens), "minutes": round((time.time() - t0) / 60, 2)}
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nvocab_size={meta['vocab_size']:,}  FastText coverage: {meta['vocab_coverage']:.1%} of words, "
          f"{tok_cov:.1%} of tokens")
    print(f"Saved to {cache_dir}/  ({meta['minutes']} min)")


# ---------------------------------------------------------------- model (rows 4-7)
def keras_init_rnn(rnn, cell, units):
    """Keras defaults: glorot_uniform kernels, orthogonal recurrent weights, zero biases, LSTM forget bias 1."""
    for name, p in rnn.named_parameters():
        if "weight_ih" in name:
            for g in p.data.chunk(p.shape[0] // units):
                nn.init.xavier_uniform_(g)
        elif "weight_hh" in name:
            for g in p.data.chunk(p.shape[0] // units):
                nn.init.orthogonal_(g)
        elif "bias" in name:
            nn.init.zeros_(p)
            if cell == "lstm" and "bias_ih" in name:
                p.data[units:2 * units] = 1.0


class ProposedRecurrent(nn.Module):
    def __init__(self, embedding_matrix, cell="lstm", bidirectional=False, head="last", units=64,
                 spatial_dropout=0.3, n_labels=len(LABELS)):
        super().__init__()
        self.cell, self.head, self.spatial_dropout = cell, head, spatial_dropout
        self.embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(embedding_matrix, dtype=torch.float32), freeze=False, padding_idx=0)  # fine-tuned
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(embedding_matrix.shape[1], units, batch_first=True, bidirectional=bidirectional)
        keras_init_rnn(self.rnn, cell, units)
        h = units * (2 if bidirectional else 1)
        if head == "last":                                  # P1: as in the base paper
            self.out = nn.Linear(h, n_labels)
            nn.init.xavier_uniform_(self.out.weight); nn.init.zeros_(self.out.bias)
        else:                                               # P2+: attention + max + mean pooling
            self.att = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1, bias=False))
            self.out = nn.Sequential(nn.Linear(3 * h, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_labels))
            for m in list(self.att) + list(self.out):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        e = self.embedding(x)
        if self.training and self.spatial_dropout > 0:    # SpatialDropout1D: drop whole embedding channels
            keep = 1.0 - self.spatial_dropout
            e = e * torch.bernoulli(torch.full((e.shape[0], 1, e.shape[2]), keep, device=e.device)) / keep
        seq, h = self.rnn(e)
        if self.head == "last":
            if self.cell == "lstm":
                h = h[0]
            return self.out(torch.cat(list(h), dim=-1))
        w = torch.softmax(self.att(seq).squeeze(-1), dim=1)          # no mask, as decided (row 8)
        att = (seq * w.unsqueeze(-1)).sum(1)
        return self.out(torch.cat([att, seq.max(1).values, seq.mean(1)], dim=-1))


# ---------------------------------------------------------------- training helpers
def pos_weights(y, cap=10.0):
    """Row 9: sqrt(neg/pos) per label, capped (labels without positives get the cap)."""
    pos = y.sum(0)
    w = np.where(pos > 0, np.sqrt((len(y) - pos) / np.maximum(pos, 1)), cap)
    return np.minimum(w, cap).astype(np.float32)


def tune_thresholds(y_val, p_val, grid=np.round(np.arange(0.05, 0.951, 0.01), 2)):
    """Row 10: per-label threshold maximising F1 on the validation set (0.5 if a label has no positives)."""
    th = []
    for j in range(y_val.shape[1]):
        y, p = y_val[:, j].astype(bool), p_val[:, j]
        if not y.any():
            th.append(0.5); continue
        best_t, best_f = 0.5, -1.0
        for t in grid:
            pred = p >= t
            tp = (pred & y).sum()
            f = 2 * tp / (pred.sum() + y.sum()) if (pred.sum() + y.sum()) else 0.0
            if f > best_f:
                best_t, best_f = float(t), f
        th.append(best_t)
    return np.array(th)


def _summary(m):
    return {k: m[k] for k in ("f1_micro", "f1_macro", "precision_macro", "recall_macro",
                              "exact_match_accuracy", "roc_auc_macro", "pr_auc_macro")}


def write_p4(key, seed, out_dir):
    """P4 = the P3 model of the same seed, evaluated with thresholds tuned on its validation outputs."""
    p3_dir = os.path.join(out_dir, "runs", f"{key}_p3", f"seed_{seed}")
    p4_dir = os.path.join(out_dir, "runs", f"{key}_p4", f"seed_{seed}")
    if os.path.isfile(os.path.join(p4_dir, "metrics.json")):
        print(f"[skip] {LABEL_OF[key + '_p4']} seed {seed}: already done")
        return
    with open(os.path.join(p3_dir, "metrics.json")) as f:
        r = json.load(f)
    val, test = np.load(os.path.join(p3_dir, "val_outputs.npz")), np.load(os.path.join(p3_dir, "test_outputs.npz"))
    th = tune_thresholds(val["y_true"], val["probs"])
    test_m = multilabel_metrics(test["y_true"], test["probs"], threshold=th)
    val_m = multilabel_metrics(val["y_true"], val["probs"], threshold=th)
    r.update({"model": f"{key}_p4", "model_label": LABEL_OF[f"{key}_p4"], "stage": "p4",
              "derived_from": p3_dir, "thresholds": dict(zip(LABELS, th.tolist())),
              "val": {**r["val"], **{f"tuned_{k}": v for k, v in _summary(val_m).items()}},
              "test": test_m})
    r["config"] = {**r["config"], "threshold": "per-label, tuned on validation"}
    os.makedirs(p4_dir, exist_ok=True)
    with open(os.path.join(p4_dir, "metrics.json"), "w") as f:
        json.dump(r, f, indent=2)
    np.savez_compressed(os.path.join(p4_dir, "test_outputs.npz"), probs=test["probs"], y_true=test["y_true"])
    print(f"{r['model_label']} seed {seed}: thresholds {np.round(th, 2).tolist()}  "
          f"TEST micro-F1={test_m['f1_micro']:.4f}  macro-F1={test_m['f1_macro']:.4f}")


def run_one(key, stage, seed, data_dir="data", cache_dir="data/rnn_cache_proposed", out_dir="results_proposed",
            epochs=15, batch_size=128, lr=1e-3, max_len=MAX_LEN, val_size=0.1, patience=3,
            max_train_samples=None, max_test_samples=None, overwrite=False, device="auto", units=128):
    cfg = RNN_MODELS[key]
    name = f"{key}_{stage}"
    label = RNN_PROPOSED[name]["label"]
    run_dir = os.path.join(out_dir, "runs", name, f"seed_{seed}")
    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.isfile(metrics_path) and not overwrite:
        print(f"[skip] {label} seed {seed}: already done ({metrics_path})")
        return
    os.makedirs(run_dir, exist_ok=True)
    print("\n" + "=" * 72 + f"\n{label}  |  seed {seed}\n" + "=" * 72)
    t0 = time.time()
    set_all_seeds(seed)
    dev = pick_device(device)

    if not os.path.isfile(os.path.join(cache_dir, "meta.json")):
        raise FileNotFoundError(f"{cache_dir} not found. Run  python -m src.rnn_proposed"
                                f"{' --smoke' if 'smoke' in cache_dir else ''}  first.")
    seq, matrix, meta = load_cache(cache_dir)
    if max_len != meta["max_len"]:          # keep-first sequences cannot be re-cut to another length
        raise ValueError(f"--max_len {max_len} differs from the cache ({meta['max_len']}). Rebuild it with "
                         f"python -m src.rnn_proposed --max_len {max_len} --overwrite")
    train_df, val_df, test_df, test_source = get_splits(data_dir, seed, val_size, max_train_samples, max_test_samples)
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}  [{test_source}]  device={dev}")
    train_rows = {i: r for r, i in enumerate(seq["train_ids"])}
    test_rows = {i: r for r, i in enumerate(seq["test_ids"])}
    X_all, X_test_all = seq["X_train"][:, -max_len:], seq["X_test"][:, -max_len:]
    X_tr = X_all[rows_for(train_df["id"].astype(str), train_rows, "train")]
    X_va = X_all[rows_for(val_df["id"].astype(str), train_rows, "validation")]
    X_te = (X_test_all[rows_for(test_df["id"].astype(str), test_rows, "test")] if test_source.startswith("official")
            else X_all[rows_for(test_df["id"].astype(str), train_rows, "test")])
    y_tr, y_va = train_df[LABELS].values.astype(np.float32), val_df[LABELS].values.astype(int)
    y_te = test_df[LABELS].values.astype(int)

    head = "last" if stage == "p1" else "attention_pooling"
    weighted = stage == "p3"
    model = ProposedRecurrent(matrix, cfg["cell"], cfg["bidirectional"], head, units=units).to(dev)
    pw = pos_weights(y_tr) if weighted else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pw, device=dev) if weighted else None)
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    rng = np.random.default_rng(seed)

    best_f1, best_state, best_epoch, wait, history = -1.0, None, 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        perm, total, n = rng.permutation(len(X_tr)), 0.0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb = torch.from_numpy(X_tr[idx]).long().to(dev)
            yb = torch.from_numpy(y_tr[idx]).to(dev)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx); n += len(idx)
        val_logits = predict(model, X_va, dev)
        val_f1 = multilabel_metrics(y_va, sigmoid(val_logits))["f1_macro"]          # threshold 0.5
        sched.step(val_f1)
        history.append({"epoch": epoch, "loss": total / n, "val_loss": bce(val_logits, y_va),
                        "val_f1_macro": val_f1, "lr": opt.param_groups[0]["lr"]})
        print(f"  epoch {epoch:2d}/{epochs}: loss={total / n:.4f}  val_loss={history[-1]['val_loss']:.4f}  "
              f"val_macroF1={val_f1:.4f}  lr={opt.param_groups[0]['lr']:.1e}")
        if val_f1 > best_f1:
            best_f1, best_epoch, wait = val_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stopping (no val macro-F1 improvement for {patience} epochs)")
                break
    model.load_state_dict(best_state)
    print(f"  restored best epoch {best_epoch} (val macro-F1={best_f1:.4f})")

    val_probs = sigmoid(predict(model, X_va, dev))
    probs = sigmoid(predict(model, X_te, dev))
    test_metrics = multilabel_metrics(y_te, probs)
    result = {
        "model": name, "model_label": label, "stage": stage, "base_model": key,
        "checkpoint": f"fasttext-crawl-300d-2M+{key}+{stage}", "seed": seed, "test_source": test_source,
        "sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "config": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "max_len": max_len,
                   "val_size": val_size, "patience": patience, "units": units,
                   "bidirectional": cfg["bidirectional"], "embedding": "FastText crawl-300d-2M (fine-tuned)",
                   "spatial_dropout": 0.3, "head": head, "weighted_loss": weighted,
                   "pos_weight": dict(zip(LABELS, pw.tolist())) if weighted else None,
                   "early_stopping": "val macro-F1 @0.5", "scheduler": "ReduceLROnPlateau(0.5, patience 1)",
                   "grad_clip": 1.0, "padding": "pre, no mask", "truncation": meta.get("truncation"),
                   "threshold": 0.5, "cleaning": meta.get("cleaning"),
                   "word2vec": meta["word2vec"], "vocab_size": meta["vocab_size"],
                   "embedding_coverage": {"vocab": meta.get("vocab_coverage"), "tokens": meta.get("token_coverage")},
                   "cache_dir": cache_dir, "device": str(dev), "torch": torch.__version__},
        "val": {"loss": bce(predict(model, X_va, dev), y_va), "best_epoch": best_epoch,
                **_summary(multilabel_metrics(y_va, val_probs))},
        "history": history,
        "test": test_metrics,
        "train_minutes": round((time.time() - t0) / 60, 2),
    }
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(os.path.join(run_dir, "test_outputs.npz"), probs=probs.astype(np.float32), y_true=y_te.astype(np.int8))
    np.savez_compressed(os.path.join(run_dir, "val_outputs.npz"), probs=val_probs.astype(np.float32), y_true=y_va.astype(np.int8))
    print(f"\n{label} seed {seed}: TEST  micro-F1={test_metrics['f1_micro']:.4f}  macro-F1={test_metrics['f1_macro']:.4f}  "
          f"ROC-AUC={test_metrics['roc_auc_macro']:.4f}  PR-AUC={test_metrics['pr_auc_macro']:.4f}  "
          f"({result['train_minutes']} min)")


def main(key):
    """Command line shared by train_proposed_lstm.py, train_proposed_bilstm.py and train_proposed_gru.py."""
    ap = argparse.ArgumentParser(description=f"Proposed {RNN_MODELS[key]['label']} (ablation P1-P4), 5 seeds")
    ap.add_argument("--stages", nargs="+", default=["p3"], choices=["p1", "p2", "p3"],
                    help="default: the final configuration only (P3 = threshold 0.5; P4 = tuned thresholds, written "
                         "automatically). P1/P2 = optional ablation stages")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="same seeds as all other models")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--cache_dir", default=None, help="default: data/rnn_cache_proposed (_smoke with --smoke)")
    ap.add_argument("--out_dir", default=None, help="default: results_proposed (results_proposed_smoke with --smoke)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=MAX_LEN)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--units", type=int, default=128, help="recurrent units (per direction for Bi-LSTM)")
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_test_samples", type=int, default=None)
    ap.add_argument("--device", default="auto", help="auto (cuda, else cpu) | cpu | mps | cuda")
    ap.add_argument("--overwrite", action="store_true", help="redo runs that already finished")
    ap.add_argument("--smoke", action="store_true", help="tiny test: 1500 train / 500 test, 1 epoch, 2 seeds")
    a = ap.parse_args()
    if a.smoke:
        a.max_train_samples, a.max_test_samples = a.max_train_samples or 1500, a.max_test_samples or 500
        a.epochs = 1
        if a.seeds == SEEDS:
            a.seeds = SEEDS[:2]
    cache_dir = a.cache_dir or os.path.join(a.data_dir, "rnn_cache_proposed_smoke" if a.smoke else "rnn_cache_proposed")
    out_dir = a.out_dir or ("results_proposed_smoke" if a.smoke else "results_proposed")
    print(f"Proposed {RNN_MODELS[key]['label']}   stages={a.stages}   seeds={a.seeds}   cache={cache_dir}   out={out_dir}")
    for seed in a.seeds:
        for stage in a.stages:
            run_one(key, stage, seed, a.data_dir, cache_dir, out_dir, a.epochs, a.batch_size, a.lr, a.max_len,
                    a.val_size, a.patience, a.max_train_samples, a.max_test_samples, a.overwrite, a.device,
                    a.units)
            if stage == "p3":
                write_p4(key, seed, out_dir)
    print(f"\nDone. Tables/figures:  python generate_results.py --results_dir {out_dir}")


def cli_prepare():
    ap = argparse.ArgumentParser(description="Cleaning v2 + FastText embedding cache for the proposed models")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--fasttext", default=FASTTEXT_PATH)
    ap.add_argument("--cleaning", choices=["light", "v2"], default="light",
                    help="light = common protocol (URLs removed, case/punctuation kept); v2 = earlier cleaning")
    ap.add_argument("--max_len", type=int, default=MAX_LEN)
    ap.add_argument("--max_vectors", type=int, default=None, help="read only the first N FastText lines")
    ap.add_argument("--seed", type=int, default=42, help="seed for the random vectors of unknown words")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="read only the first 200k FastText vectors")
    prepare(ap.parse_args())


if __name__ == "__main__":
    cli_prepare()

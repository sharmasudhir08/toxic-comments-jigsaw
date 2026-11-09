"""
Preprocessing + feature engineering for the recurrent models, exactly as in the base paper
("Empirical Comparison of Recurrent Neural Architectures for Multi-Label Toxic Comment
Classification", Sec. III-B to III-F):

    1. lowercase  2. remove punctuation  3. remove special characters and numbers (regex)
    4. NLTK word_tokenize
    5. Word2Vec on the tokenised training comments (vector_size=100, window=5, min_count=1, workers=4)
    6. tokenizer fitted on the training comments -> integer sequences -> padding
    7. embedding matrix (vocab_size x 100) for the frozen Embedding layer

It runs once and caches everything; the three model scripts (train_lstm / train_bilstm /
train_gru) read the cache. The whole train.csv is processed, so every seed's split
(made by src.data.get_splits, the same as for the transformers) is found by comment id.

    python -m src.rnn_preprocessing            # -> data/rnn_cache/
    python -m src.rnn_preprocessing --smoke    # tiny Word2Vec, -> data/rnn_cache_smoke/

Word2Vec backend: gensim (as in the paper) when it can be imported. gensim does not build
on Python 3.14 yet; in that case a built-in PyTorch Word2Vec with the same algorithm as
gensim's default (CBOW, negative sampling, frequent-word sub-sampling) and the paper's
settings is used. The backend is written to meta.json and to every run's metrics.json.

Cache contents (<cache_dir>/):
    sequences.npz          train_ids, X_train, test_ids, X_test  (int32, pre-padded to max_len)
    embedding_matrix.npy   float32 (vocab_size, 100); row 0 = padding
    vocab.json             words in index order (index = position + 1)
    meta.json              settings, vocabulary size, backend, statistics
"""

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import LABELS, MAX_LEN
from src.data import load_official_test, load_train

W2V_PARAMS = {"vector_size": 100, "window": 5, "min_count": 1, "workers": 4}   # paper, Sec. III-D
_PUNCT = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------- text cleaning (Sec. III-B)
def preprocess_text(text: str) -> str:
    text = str(text).lower()                       # 1) lowercase
    text = text.translate(_PUNCT)                  # 2) punctuation
    text = re.sub(r"[^a-z\s]", "", text)           # 3) special characters and numbers
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list:
    """4) NLTK word_tokenize. After step 3 only letters and spaces remain, so sentence
    splitting has nothing to split on; preserve_line=True gives identical tokens and
    does not need the punkt download."""
    from nltk.tokenize import word_tokenize
    return word_tokenize(text, preserve_line=True)


def preprocess_and_tokenize(texts) -> list:
    return [tokenize(preprocess_text(t)) for t in texts]


# ---------------------------------------------------------------- tokenizer + padding
def fit_tokenizer(token_lists) -> list:
    """Keras Tokenizer.fit_on_texts equivalent: words sorted by frequency, index 1 = most frequent."""
    counts = Counter(w for toks in token_lists for w in toks)
    return [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def texts_to_padded(token_lists, word_index, max_len) -> np.ndarray:
    """Keras texts_to_sequences + pad_sequences defaults: unknown words dropped,
    padding='pre', truncating='pre' (the last max_len tokens are kept)."""
    X = np.zeros((len(token_lists), max_len), dtype=np.int32)
    for i, toks in enumerate(token_lists):
        seq = [word_index[w] for w in toks if w in word_index][-max_len:]
        if seq:
            X[i, -len(seq):] = seq
    return X


# ---------------------------------------------------------------- Word2Vec
def train_word2vec_gensim(sentences, epochs, seed):
    from gensim.models import Word2Vec
    model = Word2Vec(sentences=sentences, epochs=epochs, seed=seed, **W2V_PARAMS)
    return {w: model.wv[w] for w in model.wv.index_to_key}


def train_word2vec_torch(sentences, epochs, seed, negative=5, sample=1e-3, batch_size=4096,
                         lr=2e-3, device=None):
    """CBOW + negative sampling (gensim's default algorithm) with the paper's vector_size,
    window and min_count. Optimised with mini-batch Adam instead of gensim's per-word SGD."""
    import torch
    import torch.nn.functional as F

    dim, window = W2V_PARAMS["vector_size"], W2V_PARAMS["window"]
    device = device or ("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    counts = Counter(w for s in sentences for w in s)
    vocab = [w for w, c in counts.most_common() if c >= W2V_PARAMS["min_count"]]
    w2i = {w: i for i, w in enumerate(vocab)}
    freq = np.array([counts[w] for w in vocab], dtype=np.float64)
    ids = np.fromiter((w2i[w] for s in sentences for w in s if w in w2i), dtype=np.int64)
    lens = np.array([sum(w in w2i for w in s) for s in sentences])
    sent = np.repeat(np.arange(len(lens)), lens)

    thr = sample * freq.sum()                                 # gensim's sub-sampling formula
    keep_p = np.minimum(1.0, (np.sqrt(freq / thr) + 1) * thr / freq)
    noise = torch.tensor(freq ** 0.75, dtype=torch.float32, device=device)

    emb_in = torch.empty(len(vocab), dim, device=device).uniform_(-0.5 / dim, 0.5 / dim)
    emb_out = torch.zeros(len(vocab), dim, device=device)
    emb_in.requires_grad_(); emb_out.requires_grad_()
    opt = torch.optim.Adam([emb_in, emb_out], lr=lr)
    offs = np.concatenate([np.arange(-window, 0), np.arange(1, window + 1)])

    for ep in range(epochs):
        keep = rng.random(len(ids)) < keep_p[ids]
        e_ids, e_sent = ids[keep], sent[keep]
        n, total, steps = len(e_ids), 0.0, 0
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            pos = perm[start:start + batch_size]
            ctx_pos = pos[:, None] + offs[None, :]
            inside = (ctx_pos >= 0) & (ctx_pos < n)
            ctx_pos = np.clip(ctx_pos, 0, n - 1)
            reduced = rng.integers(1, window + 1, size=len(pos))     # gensim's random window
            valid = inside & (e_sent[ctx_pos] == e_sent[pos][:, None]) \
                & (np.abs(offs)[None, :] <= reduced[:, None])
            has_ctx = valid.any(1)
            if not has_ctx.any():
                continue
            ctx = torch.from_numpy(e_ids[ctx_pos[has_ctx]]).to(device)
            msk = torch.from_numpy(valid[has_ctx]).to(device, torch.float32)
            tgt = torch.from_numpy(e_ids[pos[has_ctx]]).to(device)
            h = (emb_in[ctx] * msk[..., None]).sum(1) / msk.sum(1, keepdim=True)   # cbow_mean=1
            negs = torch.multinomial(noise, len(tgt) * negative, replacement=True).view(len(tgt), negative)
            pos_score = (h * emb_out[tgt]).sum(-1)
            neg_score = torch.bmm(emb_out[negs], h[:, :, None]).squeeze(-1)
            loss = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); steps += 1
        print(f"  word2vec epoch {ep + 1}/{epochs}: loss={total / max(steps, 1):.4f}  "
              f"({n:,} tokens after sub-sampling)")
    vecs = emb_in.detach().cpu().numpy()
    return {w: vecs[i] for i, w in enumerate(vocab)}


def train_word2vec(sentences, epochs, seed, backend="auto"):
    if backend in ("auto", "gensim"):
        try:
            return train_word2vec_gensim(sentences, epochs, seed), "gensim"
        except ImportError:
            if backend == "gensim":
                raise
            print("  gensim not installed -> using the built-in PyTorch Word2Vec (same settings)")
    return train_word2vec_torch(sentences, epochs, seed), "torch"


def build_embedding_matrix(vocab, vectors) -> np.ndarray:
    matrix = np.zeros((len(vocab) + 1, W2V_PARAMS["vector_size"]), dtype=np.float32)
    for i, w in enumerate(vocab, start=1):
        if w in vectors:
            matrix[i] = vectors[w]
    return matrix


# ---------------------------------------------------------------- EDA (Sec. III-C, Fig. 2-3)
def run_eda(train_df, train_tokens, eda_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(eda_dir, exist_ok=True)
    print("\n--- EDA (training set) ---")
    train_df.info()
    print(train_df[LABELS].describe().round(4).to_string())
    print("Missing values per column:", train_df.isna().sum().to_dict())
    counts = train_df[LABELS].sum()
    lengths = np.array([len(t) for t in train_tokens])
    print("Label counts:", counts.to_dict())
    print("Comment length (tokens after preprocessing):",
          {k: round(float(v), 1) for k, v in
           zip(["mean", "median", "p95", "max"],
               [lengths.mean(), np.median(lengths), np.percentile(lengths, 95), lengths.max()])})

    plt.figure(figsize=(7, 4))
    plt.bar(LABELS, counts.values, color="#c0504d")
    plt.ylabel("Number of comments"); plt.title("Distribution of toxicity labels (train.csv)")
    plt.xticks(rotation=20); plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "fig_label_distribution.png"), dpi=200); plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(np.clip(lengths, 0, 500), bins=60, color="#4f81bd")
    plt.xlabel("Tokens per comment (clipped at 500)"); plt.ylabel("Number of comments")
    plt.title("Distribution of comment lengths after preprocessing"); plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "fig_comment_length.png"), dpi=200); plt.close()
    return {"label_counts": {k: int(v) for k, v in counts.items()},
            "length_mean": float(lengths.mean()), "length_median": float(np.median(lengths)),
            "length_p95": float(np.percentile(lengths, 95)),
            f"share_longer_than_{MAX_LEN}": float((lengths > MAX_LEN).mean())}


# ---------------------------------------------------------------- cache I/O
def load_cache(cache_dir):
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"{cache_dir} has no preprocessed data. Run  "
                                f"python -m src.rnn_preprocessing{' --smoke' if 'smoke' in cache_dir else ''}  first.")
    with open(meta_path) as f:
        meta = json.load(f)
    seq = np.load(os.path.join(cache_dir, "sequences.npz"))
    matrix = np.load(os.path.join(cache_dir, "embedding_matrix.npy"))
    return seq, matrix, meta


def main():
    ap = argparse.ArgumentParser(description="Paper preprocessing + Word2Vec for LSTM/Bi-LSTM/GRU")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--cache_dir", default=None, help="default: data/rnn_cache (data/rnn_cache_smoke with --smoke)")
    ap.add_argument("--eda_dir", default=None, help="default: results/eda (results_smoke/eda with --smoke)")
    ap.add_argument("--max_len", type=int, default=MAX_LEN, help="padding length of the cached sequences")
    ap.add_argument("--w2v_epochs", type=int, default=5, help="gensim's default (the paper does not change it)")
    ap.add_argument("--w2v_backend", choices=["auto", "gensim", "torch"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny test: Word2Vec on 5,000 comments for 1 epoch")
    a = ap.parse_args()

    cache_dir = a.cache_dir or os.path.join(a.data_dir, "rnn_cache_smoke" if a.smoke else "rnn_cache")
    eda_dir = a.eda_dir or os.path.join("results_smoke" if a.smoke else "results", "eda")
    if os.path.isfile(os.path.join(cache_dir, "meta.json")) and not a.overwrite:
        print(f"[skip] {cache_dir} already exists (use --overwrite to rebuild)")
        return
    os.makedirs(cache_dir, exist_ok=True)
    t0 = time.time()

    train_df = load_train(a.data_dir)
    test_df = load_official_test(a.data_dir)       # rows with -1 labels are already removed
    if test_df is None:
        print("Official test files not found: only train.csv is processed "
              "(the model scripts then use a hold-out of train.csv as test set).")
    print(f"train={len(train_df)}  test={0 if test_df is None else len(test_df)}")

    print("Cleaning + tokenising ...")
    train_tokens = preprocess_and_tokenize(train_df["comment_text"])
    test_tokens = [] if test_df is None else preprocess_and_tokenize(test_df["comment_text"])
    eda = run_eda(train_df, train_tokens, eda_dir)

    w2v_sentences, w2v_epochs = train_tokens, a.w2v_epochs
    if a.smoke:
        w2v_sentences, w2v_epochs = train_tokens[:5000], 1
    print(f"\nTraining Word2Vec on {len(w2v_sentences):,} comments ({w2v_epochs} epochs) ...")
    vectors, backend = train_word2vec(w2v_sentences, w2v_epochs, a.seed, a.w2v_backend)

    vocab = fit_tokenizer(train_tokens)
    word_index = {w: i for i, w in enumerate(vocab, start=1)}
    matrix = build_embedding_matrix(vocab, vectors)
    X_train = texts_to_padded(train_tokens, word_index, a.max_len)
    X_test = texts_to_padded(test_tokens, word_index, a.max_len)
    covered = int((np.abs(matrix[1:]).sum(1) > 0).sum())

    np.savez_compressed(os.path.join(cache_dir, "sequences.npz"),
                        train_ids=np.asarray(train_df["id"].astype(str), dtype=str), X_train=X_train,
                        test_ids=np.asarray([] if test_df is None else test_df["id"].astype(str), dtype=str),
                        X_test=X_test)
    np.save(os.path.join(cache_dir, "embedding_matrix.npy"), matrix)
    with open(os.path.join(cache_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    meta = {"max_len": a.max_len, "vocab_size": len(vocab) + 1, "embedding_dim": W2V_PARAMS["vector_size"],
            "word2vec": {**W2V_PARAMS, "epochs": w2v_epochs, "backend": backend,
                         "trained_on_comments": len(w2v_sentences)},
            "words_with_vector": covered, "smoke": a.smoke,
            "n_train": len(train_df), "n_test": len(test_tokens), "eda": eda,
            "minutes": round((time.time() - t0) / 60, 2)}
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nvocab_size={meta['vocab_size']:,}  words with a Word2Vec vector={covered:,}  backend={backend}")
    print(f"Saved to {cache_dir}/  ({meta['minutes']} min).  EDA figures in {eda_dir}/")


if __name__ == "__main__":
    main()

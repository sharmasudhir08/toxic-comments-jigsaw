"""
Read-only analysis: measures the actual word-vs-subword-token mismatch at the 128 limit, using
the real light-cleaned text and the real tokenizers this project uses (NLTK for recurrent models,
each transformer's own tokenizer). No training, no new predictions -- just tokenizes the already-
saved test set text.

Usage: python -m src.check_truncation_mismatch
"""
import os
import numpy as np
import pandas as pd
from src.data import load_official_test
from src.preprocessing import clean_text_light


def main():
    test_df = load_official_test("data")
    texts = test_df["comment_text"].apply(clean_text_light).tolist()
    print(f"Test set: {len(texts)} comments (light-cleaned, matching the real pipeline)")

    from nltk.tokenize import word_tokenize
    word_counts = np.array([len(word_tokenize(t)) for t in texts])
    print(f"\nWord-token counts (NLTK, recurrent-model tokenizer): "
          f"mean={word_counts.mean():.1f}  median={np.median(word_counts):.0f}  "
          f"max={word_counts.max()}")
    pct_over_128_words = (word_counts > 128).mean() * 100
    print(f"Comments exceeding 128 words (truncated for recurrent models): {pct_over_128_words:.2f}%")

    rows = [{"tokenizer": "NLTK (recurrent)", "checkpoint": "-", "n_sampled": len(texts),
             "mean_word_count": word_counts.mean(), "median_word_count": float(np.median(word_counts)),
             "max_word_count": int(word_counts.max()), "mean_subword_per_word_ratio": np.nan,
             "pct_over_128": pct_over_128_words, "words_covered_by_128_budget": 128.0}]

    from transformers import AutoTokenizer
    for ckpt, label in [("bert-base-uncased", "BERT"), ("distilbert-base-uncased", "DistilBERT"),
                         ("roberta-base", "RoBERTa")]:
        tok = AutoTokenizer.from_pretrained(ckpt)
        # Sample for speed -- tokenizing all 63,978 comments with 3 tokenizers is unnecessary;
        # a random sample of 5000 gives a stable estimate of the ratio.
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(len(texts), size=5000, replace=False)
        sample_texts = [texts[i] for i in sample_idx]
        sample_word_counts = word_counts[sample_idx]

        subword_counts = np.array([len(tok.tokenize(t)) for t in sample_texts])
        ratio = subword_counts / np.maximum(sample_word_counts, 1)
        pct_over_128_subword = (subword_counts > 128).mean() * 100

        # How many words does 128 subword tokens cover, on average, for comments long enough to
        # be truncated by either scheme?
        long_enough = sample_word_counts >= 128
        if long_enough.sum() > 0:
            avg_ratio_long = ratio[long_enough].mean()
            words_covered_by_128_subwords = 128 / avg_ratio_long
        else:
            words_covered_by_128_subwords = float("nan")

        print(f"\n{label} ({ckpt}) tokenizer, n={len(sample_texts)} sampled comments:")
        print(f"  mean subword-tokens-per-word ratio: {ratio.mean():.3f}")
        print(f"  comments exceeding 128 subword tokens (truncated for {label}): {pct_over_128_subword:.2f}%")
        print(f"  for comments >=128 words long: 128 subword tokens covers ~{words_covered_by_128_subwords:.1f} words "
              f"(vs. the full 128 words the recurrent models see)")

        rows.append({"tokenizer": label, "checkpoint": ckpt, "n_sampled": len(sample_texts),
                     "mean_word_count": sample_word_counts.mean(),
                     "median_word_count": float(np.median(sample_word_counts)),
                     "max_word_count": int(sample_word_counts.max()),
                     "mean_subword_per_word_ratio": ratio.mean(),
                     "pct_over_128": pct_over_128_subword,
                     "words_covered_by_128_budget": words_covered_by_128_subwords})

    df = pd.DataFrame(rows)
    os.makedirs("results_proposed/report", exist_ok=True)
    out_path = "results_proposed/report/table_truncation_mismatch.csv"
    df.round(4).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

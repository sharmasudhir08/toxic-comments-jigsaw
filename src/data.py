"""
Jigsaw Toxic Comment data loading + splitting (multi-label, six labels).

Files expected in data/:  train.csv  [+ test.csv + test_labels.csv]
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import LABELS
from src.preprocessing import clean_batch

OFFICIAL_TEST = "official Jigsaw test set (test.csv + test_labels.csv, scored rows only)"
HOLDOUT_TEST = "10% hold-out of train.csv (official test files not found)"


def load_train(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "train.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Download the Jigsaw Toxic Comment Classification "
            "Challenge data from Kaggle and put train.csv in the data/ folder."
        )
    df = pd.read_csv(path)
    missing = {"comment_text", *LABELS} - set(df.columns)
    if missing:
        raise ValueError(f"train.csv is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["comment_text"]).reset_index(drop=True)
    df[LABELS] = df[LABELS].astype(int)
    return df


def load_official_test(data_dir: str):
    """Returns the scored part of the official test set, or None if files are absent."""
    p_test = os.path.join(data_dir, "test.csv")
    p_labels = os.path.join(data_dir, "test_labels.csv")
    if not (os.path.isfile(p_test) and os.path.isfile(p_labels)):
        return None
    test = pd.read_csv(p_test)
    labels = pd.read_csv(p_labels)
    df = test.merge(labels, on="id", how="inner")
    df = df.dropna(subset=["comment_text"])
    df = df[(df[LABELS] != -1).all(axis=1)].reset_index(drop=True)   # -1 = not scored
    df[LABELS] = df[LABELS].astype(int)
    return df


def get_splits(data_dir, seed, val_size=0.1, max_train_samples=None, max_test_samples=None):
    """
    Returns (train_df, val_df, test_df, test_source).
    The split depends on `seed`, so the 5 runs use 5 different train/val splits.
    The official test set (when present) is identical for every run.
    """
    all_train = load_train(data_dir)
    test_df = load_official_test(data_dir)

    if test_df is None:
        trainval, test_df = train_test_split(
            all_train, test_size=0.1, random_state=seed,
            stratify=all_train[LABELS].max(axis=1))
        source = HOLDOUT_TEST
    else:
        trainval, source = all_train, OFFICIAL_TEST

    train_df, val_df = train_test_split(
        trainval, test_size=val_size, random_state=seed,
        stratify=trainval[LABELS].max(axis=1))

    if max_train_samples and len(train_df) > max_train_samples:
        train_df = train_df.sample(n=max_train_samples, random_state=seed)
        val_cap = max(200, int(max_train_samples * val_size / (1 - val_size)))
        val_df = val_df.sample(n=min(len(val_df), val_cap), random_state=seed)
    if max_test_samples and len(test_df) > max_test_samples:
        test_df = test_df.sample(n=max_test_samples, random_state=seed)

    out = []
    for d in (train_df, val_df, test_df):
        d = d.copy()
        d["text"] = clean_batch(d["comment_text"])
        out.append(d.reset_index(drop=True))
    return out[0], out[1], out[2], source

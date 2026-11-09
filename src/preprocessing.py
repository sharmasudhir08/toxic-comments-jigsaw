"""Light text cleaning applied before tokenisation (kept from the original project)."""

import re


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/mentions, keep basic punctuation."""
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s#'!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_batch(texts) -> list:
    return [clean_text(t) for t in texts]


def clean_text_light(text: str) -> str:
    """Common-protocol input for ALL six models (preregistration v3): remove URLs, normalise whitespace.
    Case, punctuation and numbers are kept: Maslej-Krešňáková et al. (2020, Table 9) found that standard
    pre-processing lowered macro-F1 with pretrained embeddings."""
    text = re.sub(r"http\S+|www\.\S+", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()

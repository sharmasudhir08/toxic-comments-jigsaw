"""Shared constants for the whole project."""

# The six Jigsaw labels (same order as the base paper).
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Five independent runs. Each seed changes the train/validation split,
# the classifier-head initialisation and the data order.
SEEDS = [42, 43, 44, 45, 46]

MAX_LEN = 128        # max tokens per comment
THRESHOLD = 0.5      # sigmoid threshold (same as the base paper)

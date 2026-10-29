"""Canonical dense batches, including positive-only in-batch-negative training."""

from .collator import RetrievalCollator


class DenseCollator:
    def __init__(self, tokenizer, query_max_length, doc_max_length):
        self.tokenizer = tokenizer
        self.lengths = [query_max_length, doc_max_length, doc_max_length]

    def __call__(self, rows):
        has_negatives = ["negative" in row for row in rows]
        keys = ["query", "positive"] + (["negative"] if any(has_negatives) else [])
        return RetrievalCollator(self.tokenizer, self.lengths[: len(keys)], keys)(rows)

"""Canonical dense batches, including positive-only in-batch-negative training."""

from .collator import RetrievalCollator


class DenseCollator:
    def __init__(self, tokenizer, query_max_length, doc_max_length):
        self.tokenizer = tokenizer
        self.lengths = [query_max_length, doc_max_length, doc_max_length]

    def __call__(self, rows):
        has_negatives = ["negative" in row for row in rows]
        if any(has_negatives) and not all(has_negatives):
            raise ValueError("Every row in a batch must either have negatives or use positive-only training")
        keys = ["query", "positive"] + (["negative"] if all(has_negatives) else [])
        return RetrievalCollator(self.tokenizer, self.lengths[: len(keys)], keys)(rows)

"""Deterministic held-out retrieval evaluation independent of training negatives.

Corpus JSONL rows contain ``id`` and ``text``; query rows contain ``query``
and ``relevant_ids`` (a nonempty list of corpus IDs). Relevance is binary.
"""

import json
import math
from pathlib import Path

import torch


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty retrieval evaluation file: {path}")
    return rows


class RetrievalEvaluator:
    def __init__(self, data, task):
        self.data = data
        self.task = task
        self.corpus = read_rows(data.valid_corpus_file)
        self.queries = read_rows(data.valid_query_file)
        ids = [str(row["id"]) for row in self.corpus]
        if len(set(ids)) != len(ids):
            raise ValueError("Evaluation corpus IDs must be unique")
        self.ids = ids
        for index, row in enumerate(self.corpus):
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"Invalid corpus text at row {index}")
        for index, row in enumerate(self.queries):
            relevant = row.get("relevant_ids")
            if (
                not isinstance(row.get("query"), str)
                or not row["query"].strip()
                or not isinstance(relevant, list)
                or not relevant
                or not {str(value) for value in relevant}.issubset(ids)
            ):
                raise ValueError(f"Invalid query/relevance at row {index}")

    @torch.no_grad()
    def __call__(self, model, batch_size, prefix="eval"):
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        def encode(texts, length):
            tokens = model.tokenizer(texts, padding=True, truncation=True, max_length=length, return_tensors="pt").to(
                device
            )
            if self.task == "colbert":
                return (
                    model._encode(tokens["input_ids"], tokens["attention_mask"], normalize_embeddings=True),
                    tokens["attention_mask"][:, 1:],
                )
            return model(tokens), None

        totals = {"recall_at_1": 0.0, "recall_at_10": 0.0, "mrr_at_10": 0.0, "ndcg_at_10": 0.0}
        try:
            for row in self.queries:
                query, query_mask = encode([row["query"]], self.data.query_max_length)
                scores = []
                for offset in range(0, len(self.corpus), batch_size):
                    documents, mask = encode(
                        [item["text"] for item in self.corpus[offset : offset + batch_size]], self.data.doc_max_length
                    )
                    if self.task == "colbert":
                        similarities = torch.einsum("qsh,bth->qbst", query, documents)
                        similarities = similarities.masked_fill(~mask[None, :, None, :].bool(), float("-inf"))
                        maxima = similarities.max(-1).values
                        maxima = maxima.masked_fill(~torch.isfinite(maxima), 0)
                        score = (maxima * query_mask[:, None, :]).sum(-1)[0]
                    else:
                        score = (query @ documents.T)[0]
                    scores.extend(score.float().cpu().tolist())
                ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:10]
                relevant = {str(value) for value in row["relevant_ids"]}
                hits = [self.ids[index] in relevant for index in ranking]
                totals["recall_at_1"] += sum(hits[:1]) / len(relevant)
                totals["recall_at_10"] += sum(hits) / len(relevant)
                totals["mrr_at_10"] += next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0)
                dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
                ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
                totals["ndcg_at_10"] += dcg / ideal
        finally:
            model.train(was_training)
        return {f"{prefix}_{key}": value / len(self.queries) for key, value in totals.items()}

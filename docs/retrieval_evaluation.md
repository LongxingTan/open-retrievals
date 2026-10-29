# Held-out retrieval evaluation

Set both `data.valid_query_file` and `data.valid_corpus_file` to JSONL files.
Corpus rows have unique `id` values and nonempty `text`:

```json
{"id": "doc-1", "text": "A relevant passage"}
```

Query rows contain `query` and a nonempty list of `relevant_ids` referencing that
corpus. Relevance is binary:

```json
{"query": "A question", "relevant_ids": ["doc-1"]}
```

Dense and ColBERT evaluation reports `eval_recall_at_1`, `eval_recall_at_10`,
`eval_mrr_at_10`, and `eval_ndcg_at_10`. With these files configured, the default
best-checkpoint metric becomes `eval_ndcg_at_10` with higher scores preferred.
`data.valid_file` is optional for corpus evaluation. Scores are computed against
the full corpus in bounded document batches, with deterministic ID-order ties;
this exact evaluator prioritizes correctness for small held-out corpora.

Cross-encoder training accepts either labeled pairs or the same
query/positive/negative rows as retrieval training. Grouped rows optimize softmax
cross entropy with the positive as the first candidate; each query has equal
weight even when negative counts differ. Every group needs a negative.

Paths in YAML are relative to the YAML file. Command-line overrides and dictionary
configuration paths are relative to the current working directory.

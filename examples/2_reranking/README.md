# Reranking model

## Cross encoder reranking

Refer to [the fine-tuning code](./train_cross_encoder.py), to train model like:

- BAAI/bge-reranker-base
- BAAI/bge-reranker-v2-m3
- maidalun1020/bce-reranker-base_v1



## LLM reranking

Refer to [the fine-tuning code](./train_llm.py), to train the model like
- BAAI/bge-reranker-v2-gemma


Note
- `AutoModelForRanking.from_pretrained(model_name_or_path, causal_lm=True)`
- Prompt: "Given a query with a relevant body, determine whether the document is pertinent to the query by providing a prediction of either 'Yes' or 'No'."

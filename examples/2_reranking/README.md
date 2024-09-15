# Reranking model

| 模型                                | 微调前   | 微调后    |
|-----------------------------------|-------|--------|
| bge-reranker-base                 | 0.666 | 0.706  |
| bge-m3                            | 0.657 | 0.695  |
| Qwen2-1.5B-Instruct               | -     | 0.699  |
| bge-reranker-v2-gemma             | 0.637 | 0.706  |
| chinese-roberta-wwm-ext (ColBERT) | -     | 0.687  |

- MAP in t2 ranking data


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

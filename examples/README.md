# Open-Retrievals examples

## Embedding

**Data Format**

- Text pair: use in-batch negative fine-tuning
```
{'query': TEXT_TYPE, 'positive': List[TEXT_TYPE]}
...
```

- Text triplet: Hard negative (or mix In-batch negative) fine-tuning
```
{'query': TEXT_TYPE, 'positive': List[TEXT_TYPE], 'negative': List[TEXT_TYPE]}
...
```

- Text scored pair:
```
{(query, positive, label), (query, negative, label), ...}
```

**Fine-tune**

- [embedding-pairwise finetune](./0_embedding/train_pairwise.py)
- [embedding-llm pairwise finetune](./0_embedding/train_llm.py)
  - set `query_instruction`
    - "Given a query and a relevant document, retrieve the document that are pertinent to the query\nQuery: "
  - use the appropriate `pooling_method`
    - `last`
  - maybe we need to reduce the batch_size due to large model size
  - set `use_lora` to `True` if you want to use lora to reduce training memory


## Reranking
- [rerank-cross encoder](./2_reranking/train_cross_encoder.py)
- [rerank-colbert](3_colbert/rerank_colbert.py)
- [rerank-llm finetune](./2_reranking/train_llm.py)
  - `AutoModelForRanking.from_pretrained(model_name_or_path, causal_lm=True)`
  - Prompt: "Given a query with a relevant body, determine whether the document is pertinent to the query by providing a prediction of either 'Yes' or 'No'."


## RAG
- [RAG with Langchain](4_rag/rag_langchain_demo.py)


## Whole pipeline examples
- [t2-ranking dataset](./t2_ranking/README.md)
- [scifact dataset](./scifact/README.md)
- [wikipedia-nq dataset](./wikipedia-nq/README.md)


## FAQ

1. The grad_norm during training is always zero?
- consider to change fp16 or bf16

2. The fine-tuned embedding performance during inference is worse than original?
- check whether the pooling_method is correct
- check whether the prompt is exactly same as training for LLM model

3. How can we fine-tune the `BAAI/bge-m3` ColBERT model?
- open-retrievals support to fine-tune the `BAAI/bge-m3 colbert` directly, just don't set `use_fp16=True` while fine-tuning, and set the learning_rate smaller

# Sentence embedding model

## Fine-tune

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



if you want to train directly use shell, refer to the [document](https://open-retrievals.readthedocs.io/en/master/embed.html)


## LLM embed
- no need to set `causal_lm=True` like LLMRanker for AutoModelForEmbedding, but normally set the pooling_method to `last`
- set `query_instruction`
  - "Given a query and a relevant document, retrieve the document that are pertinent to the query\nQuery: "
- use the appropriate `pooling_method`
  - `last`
- maybe we need to reduce the batch_size due to large model size
- set `use_lora` to `True` if you want to use lora to reduce training memory

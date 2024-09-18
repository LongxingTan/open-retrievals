# Text embedding model

| Model                  | Original | Finetuned |
|------------------------|----------|-----------|
| m3e                    | 0.654    | 0.693     |
| bge-base-zh-v1.5       | 0.657    | 0.703     |
| Qwen2-1.5B-Instruct    | -        | 0.695     |
| e5-mistral-7b-instruct | 0.651    | 0.699     |

- MAP in t2 ranking data


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


Train directly using shell script, refer to the [document](https://open-retrievals.readthedocs.io/en/master/embed.html)

## Transformer encoder embedding

Refer to [the fine-tuning code](./train_pairwise.py) to train the model like


## LLM embedding

Refer to [the fine-tuning code](./train_llm.py), to train the model like


Note
- no need to set `causal_lm=True` like LLMRanker for AutoModelForEmbedding, but normally set the pooling_method to `last`
- set `query_instruction` and `document_instruction` in `RetrievalTrainDataset` during train or add it manually to text directly, set it in `AutoModelForEmbedding` during encode
  - "Given a query and a relevant document, retrieve the document that are pertinent to the query\nQuery: "
- use the appropriate `pooling_method`
  - `last`
- maybe we need to reduce the batch_size due to large model size
- set `use_lora` to `True` if you want to use lora to reduce training memory

```python
from retrievals import AutoModelForEmbedding

model_name = 'intfloat/e5-mistral-7b-instruct'
model = AutoModelForEmbedding.from_pretrained(
  model_name,
  pooling_method='last',
  use_fp16=True,
  query_instruction='Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ',
  document_instruction='',
)
```

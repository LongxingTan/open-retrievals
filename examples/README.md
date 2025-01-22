# 🚀 Open-Retrievals Examples
Welcome to Open-Retrievals, a cutting-edge repository designed to empower your retrieval-augmented generation (RAG) pipelines with state-of-the-art techniques in embedding, reranking, and RAG integration.
- [t2-ranking dataset](./t2_ranking/README.md)
- [scifact dataset](./scifact/README.md)
- [wikipedia-nq dataset](./wikipedia-nq/README.md)


## 🔍 1. Embedding Models
- [embedding-pairwise finetune](./0_embedding/train_pairwise.py)
- [embedding-llm pairwise finetune](./0_embedding/train_llm.py)
  - no need to set `causal_lm=True` like LLMRanker for AutoModelForEmbedding, but normally set the pooling_method to `last`
  - set `query_instruction` and `document_instruction` in `RetrievalTrainDataset` during train or add it manually to text directly, while set it in `AutoModelForEmbedding` during encode
  - "Given a query and a relevant document, retrieve the document that are pertinent to the query\nQuery: "
  - use the appropriate `pooling_method`: `last`
  - maybe we need to reduce the batch_size due to large model size
  - set `use_lora` to `True` if you want to use lora to reduce training memory


| Model                  | Original | Finetuned |
|------------------------|----------|-----------|
| m3e                    | 0.654    | 0.693     |
| bge-base-zh-v1.5       | 0.657    | 0.703     |
| Qwen2-1.5B-Instruct    | -        | 0.695     |
| e5-mistral-7b-instruct | 0.651    | 0.699     |


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

## 📊 2. [Reranking](./2_reranking)
- [rerank-cross encoder](./2_reranking/train_cross_encoder.py)
- [rerank-colbert](3_colbert/train_colbert.py)
- [rerank-llm finetune](./2_reranking/train_llm.py)
  - `AutoModelForRanking.from_pretrained(model_name_or_path, causal_lm=True)`
  - Prompt: "Given a query with a relevant body, determine whether the document is pertinent to the query by providing a prediction of either 'Yes' or 'No'."


| Model                             | Original | Finetuned |
|-----------------------------------|----------|-----------|
| bge-reranker-base                 | 0.666    | 0.706     |
| bge-m3                            | 0.657    | 0.695     |
| Qwen2-1.5B-Instruct               | -        | 0.699     |
| bge-reranker-v2-gemma             | 0.637    | 0.706     |
| chinese-roberta-wwm-ext (ColBERT) | -        | 0.687     |


## 📚 3. [RAG](./4_rag)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fJC-8er-a4NRkdJkwWr4On7lGt9rAO4P?usp=sharing)

For basic rag application, refer to [rag_langchain_demo.py](4_rag/rag_langchain_demo.py)


## 🚀 4. Deployment

speed: `Nvidia TensorRT + Nvidia Triton inference server` > `Microsoft ONNX Runtime + Nvidia Triton inference server` > `Pytorch + FastAPI`

### 4.1 Transfer to onnx
**Prerequisites**
```shell
pip install optimum
pip install onnxruntime
```

```shell
python embed2onnx.py --model_name BAAI/bge-small-en-v1.5 --output_path ./onnx_model
```


## ❓ 5. FAQ

1. The grad_norm during training is always zero?
- consider to change fp16 or bf16
- while training, set `bf16` or `fp16` in `TrainingArguments`; while inference, set `use_fp16=True` in `AutoModelForEmbedding` or `LLMRanker`

2. The fine-tuned embedding performance during inference is worse than original?
- check whether the pooling_method is correct
- check whether the prompt or instruction is exactly same as training for LLM model

3. How can we fine-tune the `BAAI/bge-m3` ColBERT model?
- open-retrievals support to fine-tune the `BAAI/bge-m3 colbert` directly, just don't set `use_fp16=True` while fine-tuning, and set the learning_rate smaller

4. The performance is worse?
- the collator and loss should be aligned, especially for triplet training with negative embeddings. The collator of open-retrievals provided is `{query: value, positive: value, negative: value}`. Another collator is `{query: value, document: positive+negative}`, so the loss function should be treated accordingly

# T2_ranking

| Model              | map   | mrr   |
|--------------------|-------|-------|
| bge-base-zh-v1.5   | 0.657 | 0.768 |
| + **fine-tuning**  | 0.701 | 0.816 |
| bge-reranker-base  | 0.666 | 0.761 |
| + **fine-tuning**  | 0.691 | 0.806 |


## 1. Prepare dataset

- [t2-reranking data](https://huggingface.co/datasets/C-MTEB/T2Reranking)
```shell
python prepare_t2ranking_data.py
```

## 2. Finetune embedding

```shell
sh embed_pairwise_train.sh
```

```shell
sh embed_llm_train.sg
```


## Rerank
```shell
sh rerank_cross_encoder.sh
```

```shell
sh rerank_colbert.sh
```

```shell
sh rerank_llm.sh
```

## Evaluate
```shell

```

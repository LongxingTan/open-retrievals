# T2_ranking

An end-to-end example for text retrieval

## Experiment

bge-base-zh-v1.5
- "map": 0.6569549236524207, "mrr": 0.7683207806932297
- embed/pairwise/infonce: "map": 0.7012381232799435, "mrr": 0.81575288845697

bge-reranker-base
- "map": 0.6660360850586858, "mrr": 0.76091472303207
- rerank/cross-encoder: "map": 0.6906494118852755, "mrr": 0.8064902548320916


## Prepare dataset
```shell
python prepare_t2ranking_data.py
```

## Train

```shell
sh pairwise_embed_train.sh
```

## Indexing
Encode corpus
```shell
sh encode_corpus.sh
```

Encode Query
```shell
sh encode_query.sh
```

## Retrieve
```shell
sh retrieve.sh
```

## Rerank
```shell
sh rerank.sh
```

## Evaluate
```shell

```

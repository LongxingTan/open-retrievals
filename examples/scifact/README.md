# scifact
- [scifact data](https://huggingface.co/datasets/Tevatron/scifact)
- [scifact corpus](https://huggingface.co/datasets/Tevatron/scifact-corpus)

## Fine-tuning Embedding
```shell
sh embed_pairwuse_train.sh
```


## Encoding corpus
- save the pair of `(embedding vector, id)` for each corpus example, support for multiple files

```shell
sh encode_corpus.sh
```


## Encoding query
- save the pair of `(embedding vector, id)` for each query example
- use `Tevatron/scifact/dev` or `Tevatron/scifact/test` so we can choose to encode the dev or test file

```shell
sh encode_query.sh
```

## Retrieval
```shell
sh retrieve.sh
```


Reranking
```shell
sh rerank.sh
```
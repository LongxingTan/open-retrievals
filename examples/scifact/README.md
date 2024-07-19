# scifact

| Model                  | mrr@10 | recall@10 | ndcg@10 |
|------------------------|--------|-----------|---------|
| bge-base-en-v1.5       | 0.703  | 0.862     | 0.744   |
| + **fine-tuning**      | 0.757  | 0.900     | 0.793   |
| e5-mistral-7b-instruct | 0.589  | 0.748     | 0.630   |
| + **fine-tuning**      | 0.763  | 0.940     | 0.806   |


## Fine-tuning embedding
- [scifact data](https://huggingface.co/datasets/Tevatron/scifact)
- [scifact corpus](https://huggingface.co/datasets/Tevatron/scifact-corpus)

```shell
sh embed_pairwuse_train.sh
```

Optional: llm embedding
```shell
sh embed_llm_train.sh
```

## Encoding corpus
- save the pair of `(embedding vector, id)` for each corpus example, support for multiple files
- for llm embed encoding, remember to use the same instruction

```shell
sh encode_corpus.sh
```

Optional: llm encoding
```shell
sh encode_llm_corpus.sh
```

## Encoding query
- save the pair of `(embedding vector, id)` for each query example
- use `Tevatron/scifact/dev` or `Tevatron/scifact/test` so we can choose to encode the dev or test file

```shell
sh encode_query.sh
```

Optional: llm encoding
```shell
sh encode_llm_query.sh
```

## Retrieval
```shell
sh retrieve.sh
```

## Reranking
```shell
sh rerank.sh
```

## Evaluation
- download the `dev_qrels.txt` from [dropbox](https://www.dropbox.com/s/lpq8mfynqzsuyy5/dev_qrels.txt)
```shell
python evaluate.py
```

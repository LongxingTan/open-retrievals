# Retrievals examples


## Finetune
- [Text embeddings finetune with contrastive learning](./embed_pairwise_finetune.py)

```shell
python embed_pointwise_finetune.py
```

```shell
python embed_pairwise_finetune.py
```

```shell
sh embed_llm.sh
```


## Retrieval

```shell
python retrieval_faiss.py
```

## Rerank
- [Cross-encoder Rerank using T2Ranking data](./rerank_cross_encoder.py)

```shell
python rerank_cross_encoder.py
```

## RAG
- [RAG application with retrieval, rerank in langchain](./rag_langchain.py)
- [A RAG app demo](./rag_app/README.md)

```shell
python rag_langchain.py
```

## Trainer

- customer trainer
  - support FGM, AWP
  - support EMA

- transformer trainer
  - support deepspeed

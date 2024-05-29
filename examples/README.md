# Retrievals examples


## Finetune
- [Text embeddings finetune with contrastive learning](0_embeddings/embed_pairwise_finetune.py)

```shell
python embed_pairwise_simple.py
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
- [Cross-encoder Rerank using T2Ranking data](2_rerank/train_cross_encoder.py)

```shell
python train_cross_encoder.py
```

- [ColBERT rerank](2_rerank/train_colbert.py)


## RAG
- [RAG application with retrieval, rerank in langchain](3_rag/rag_langchain.py)
- [A RAG app demo](3_rag/README.md)

```shell
python rag_langchain.py
```

## Trainer

- customer trainer
  - support FGM, AWP
  - support EMA

- transformer trainer
  - support deepspeed

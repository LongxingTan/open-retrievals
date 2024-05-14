# Retrievals examples


## Finetune
- [Text embeddings finetune with contrastive learning](./finetune_pairwise_embed.py)
```shell
sh finetune_pairwise.sh
```

- [LLM text embeddings finetune with contrastive learning](./finetune_llm_embed.py)
```shell
sh finetune_llm.sh
```


## Retrieval
- [Retrieval with faiss](./retrieval_dense.py)


## Rerank
- [Cross-encoder Rerank using T2Ranking data](./rerank_cross_encoder.py)


## RAG
- [RAG application with retrieval, rerank in langchain](./rag_langchain.py)
- [A RAG app demo](./rag_app/README.md)


## Trainer

- customer trainer
  - support FGM, AWP
  - support EMA

- transformer trainer
  - support deepspeed

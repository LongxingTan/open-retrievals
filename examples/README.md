# Open-Retrievals examples

## [Embedding](./0_embedding)
- [embedding-pairwise finetune](./0_embedding/train_pairwise.py)
- [embedding-llm pairwise finetune](./0_embedding/train_llm.py)


## [Reranking](./2_reranking)
- [rerank-cross encoder](./2_reranking/train_cross_encoder.py)
- [rerank-colbert](3_colbert/rerank_colbert.py)
- [rerank-llm finetune](./2_reranking/train_llm.py)


## [RAG](./4_rag)
- [RAG with Langchain](4_rag/rag_langchain_demo.py)


## Whole pipeline examples
- [t2-ranking dataset](./t2_ranking/README.md)
- [scifact dataset](./scifact/README.md)
- [wikipedia-nq dataset](./wikipedia-nq/README.md)


## FAQ

1. The grad_norm during training is always zero?
- consider to change fp16 or bf16

2. The fine-tuned embedding performance during inference is worse than original?
- check whether the pooling_method is correct
- check whether the prompt is exactly same as training for LLM model

3. How can we fine-tune the `BAAI/bge-m3` ColBERT model?
- open-retrievals support to fine-tune the `BAAI/bge-m3 colbert` directly, just don't set `use_fp16=True` while fine-tuning, and set the learning_rate smaller

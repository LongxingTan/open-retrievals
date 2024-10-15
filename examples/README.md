# Open-Retrievals examples

## 1. [Embedding](./0_embedding)
- [embedding-pairwise finetune](./0_embedding/train_pairwise.py)
- [embedding-llm pairwise finetune](./0_embedding/train_llm.py)


## 2. [Reranking](./2_reranking)
- [rerank-cross encoder](./2_reranking/train_cross_encoder.py)
- [rerank-colbert](3_colbert/train_colbert.py)
- [rerank-llm finetune](./2_reranking/train_llm.py)


## 3. [RAG](./4_rag)
- [RAG with Langchain](4_rag/rag_langchain_demo.py)


## 4. Whole pipeline examples
- [t2-ranking dataset](./t2_ranking/README.md)
- [scifact dataset](./scifact/README.md)
- [wikipedia-nq dataset](./wikipedia-nq/README.md)


## 5. FAQ

1. The grad_norm during training is always zero?
- consider to change fp16 or bf16
- while training, set `bf16` or `fp16` in `TrainingArguments`; while inference, set `use_fp16=True` in `AutoModelForEmbedding` or `LLMRanker`

2. The fine-tuned embedding performance during inference is worse than original?
- check whether the pooling_method is correct
- check whether the prompt or instruction is exactly same as training for LLM model

3. How can we fine-tune the `BAAI/bge-m3` ColBERT model?
- open-retrievals support to fine-tune the `BAAI/bge-m3 colbert` directly, just don't set `use_fp16=True` while fine-tuning, and set the learning_rate smaller

4. The performance is worse?
- the collator and loss should be aligned, especially for triplet training with negative embeddings. The collator of open-retrievals provided is `{query: value, positive: value, negative: value}`. Another collator is `{query: value, document: positive+negative}`, if so the loss function should be treated accordingly

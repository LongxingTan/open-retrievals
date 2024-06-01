# Open-Retrievals examples


## Embedding & Retrieval

[Text embeddings finetune with contrastive learning](./0_embeddings/pairwise_finetune2.py)

**Data Format**
Training: Each line of the Train file is a training instance
```
{'query': TEXT_TYPE, 'positives': List[TEXT_TYPE], 'negatives': List[TEXT_TYPE]}
...
```
Inference/Encoding: Each line of the encoding file is a piece of text to be encoded
```
{text_id: "xxx", 'text': TEXT_TYPE}
...
```

**Training**
```shell
cd 1_retrieval
python pairwise_finetune.py
```

```shell
cd 1_retrieval

CUDA_VISIBLE_DEVICES=0 python pairwise_finetune2.py \
    --model_name_or_path bert-base-multilingual-uncased \
    --train_data ./example_data/toy_finetune_data.jsonl \
    --output_dir modeloutput
```

```shell
cd 1_retrieval
sh llm_embed.sh
```


## Rerank
- [Cross-encoder Rerank using T2Ranking data](2_rerank/train_cross_encoder.py)

```shell
cd 2_rerank
python train_cross_encoder.py
```

- [ColBERT rerank](2_rerank/train_colbert.py)


## RAG
- [RAG application with retrieval, rerank in langchain](./3_rag/rag_langchain_demo.py)


```shell
cd 3_rag
python rag_langchain_demo.py
```

# Open-Retrievals examples

## Basic Usage

| Exp                        | Model                   | Performance | Finetune  | Colab                                                                                                                                                               |
|----------------------------|-------------------------|-------------|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| embed pairwise finetune    | bge-base-zh-v1.5        | 0.657       | **0.701** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17KXe2lnNRID-HiVvMtzQnONiO74oGs91?usp=sharing) |
| embed llm finetune (LoRA)  | Qwen2-1.5B-Instruct     | 0.554       |           | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jj1kBQWFcuQ3a7P9ttnl1hgX7H8WA_Za?usp=sharing) |
| rerank cross encoder       | bge-reranker-base       | 0.666       | **0.691** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QvbUkZtG56SXomGYidwI4RQzwODQrWNm?usp=sharing) |
| rerank colbert (zero shot) | chinese-roberta-wwm-ext |             |           | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QVtqhQ080ZMltXoJyODMmvEQYI6oo5kO?usp=sharing) |
| rerank llm finetune (LoRA) | Qwen2-1.5B-Instruct     |             |           |                                                                                                                                                                     |


- [embedding-pairwise finetune](./embedding_pairwise_finetune.py)
- [embedding-llm pairwise finetune](./embedding_llm_finetune.py)
- [rerank-cross encoder](./rerank_cross_encoder.py)
- [rerank-colbert](./rerank_colbert.py)
- [rerank-llm finetune](../reference/rerank_llm_finetune.py)
- [RAG with Langchain](./rag_langchain_demo.py)


## Retrieval

**Data Format**
```
{'query': TEXT_TYPE, 'positive': List[TEXT_TYPE], 'negative': List[TEXT_TYPE]}
...
```

**Pairwise embedding finetune**
```shell
MODEL_NAME="BAAI/bge-base-zh-v1.5"
TRAIN_DATA="/t2_ranking.jsonl"
OUTPUT_DIR="/t2_output"

torchrun --nproc_per_node 1 \
  -m retrievals.pipelines.embed \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --model_name_or_path $MODEL_NAME \
  --do_train \
  --train_data $TRAIN_DATA \
  --positive_key positive \
  --negative_key negative \
  --learning_rate 3e-5 \
  --fp16 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 32 \
  --dataloader_drop_last True \
  --query_max_length 64 \
  --document_max_length 512 \
  --train_group_size 2 \
  --logging_steps 100 \
  --temperature 0.02 \
  --use_inbatch_neg false
```

**Pairwise LLM embedding finetune**
- add query_instruction
  - "Given a query and a relevant document, retrieve the document that are pertinent to the query\nQuery: "
- use the appropriate pooling_method
  - last
- maybe reduce the batch_size due to large model size
- set use_lora to True if you want to use lora

```shell
MODEL_NAME="intfloat/e5-mistral-7b-instruct"
TRAIN_DATA="/t2_ranking.jsonl"
OUTPUT_DIR="/t2_output"

torchrun --nproc_per_node 1 \
  -m retrievals.pipelines.embed \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --model_name_or_path $MODEL_NAME \
  --do_train \
  --train_data $TRAIN_DATA \
  --positive_key positive \
  --negative_key negative \
  --use_lora True \
  --query_instruction "Given a query and a relevant document, retrieve the document that are pertinent to the query\nQuery: " \
  --document_instruction '# Document: ' \
  --learning_rate 3e-5 \
  --bf16 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --dataloader_drop_last True \
  --query_max_length 128 \
  --document_max_length 256 \
  --train_group_size 2 \
  --logging_steps 100 \
  --temperature 0.02 \
  --use_inbatch_neg false
```


## Rerank

**Cross encoder reranking**

```shell
MODEL_NAME="BAAI/bge-reranker-base"
TRAIN_DATA="/t2_ranking.jsonl"
OUTPUT_DIR="/t2_output"

torchrun --nproc_per_node 1 \
  -m retrievals.pipelines.rerank \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --model_name_or_path $MODEL_NAME \
  --model_type cross-encoder \
  --do_train \
  --train_data $TRAIN_DATA \
  --positive_key positive \
  --negative_key negative \
  --learning_rate 3e-5 \
  --fp16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 32 \
  --dataloader_drop_last True \
  --max_length 512 \
  --train_group_size 3 \
  --logging_steps 100
```


**Colbert reranking**

```shell
MODEL_NAME='hfl/chinese-roberta-wwm-ext'
TRAIN_DATA="/t2_ranking.jsonl"
OUTPUT_DIR="/t2_output"

torchrun --nproc_per_node 1 \
  --module retrievals.pipelines.rerank \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --model_name_or_path $MODEL_NAME \
  --tokenizer_name $MODEL_NAME \
  --model_type colbert \
  --do_train \
  --train_data $TRAIN_DATA \
  --positive_key positive \
  --negative_key negative \
  --learning_rate 1e-5 \
  --fp16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --dataloader_drop_last True \
  --max_length 512 \
  --train_group_size 8 \
  --unfold_each_positive false \
  --save_total_limit 2 \
  --logging_steps 100
```


**LLM reranking**

- AutoModelForRanking.from_pretrained(model_name_or_path, causal_lm = True)
- Prompt: "Given a query with a relevant body, determine whether the document is pertinent to the query by providing a prediction of either 'Yes' or 'No'."

```shell
MODEL_NAME="Qwen/Qwen2-1.5B-Instruct"
TRAIN_DATA="/t2_ranking.jsonl"
OUTPUT_DIR="/t2_output"

torchrun --nproc_per_node 1 \
    -m retrievals.pipelines.rerank \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --model_name_or_path $MODEL_NAME \
    --model_type llm \
    --causal_lm True \
    --use_lora True \
    --train_data $TRAIN_DATA \
    --task_prompt "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'." \
    --query_instruction "A: " \
    --document_instruction 'B: ' \
    --positive_key positive \
    --negative_key negative \
    --learning_rate 2e-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --dataloader_drop_last True \
    --max_len 256 \
    --train_group_size 4 \
    --logging_steps 1 \
    --save_steps 2000 \
    --save_total_limit 2 \
    --bf16
```

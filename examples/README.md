# Open-Retrievals examples

## Basic Usage

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
  --train_group_size 4 \
  --logging_steps 100 \
  --temperature 0.02 \
  --use_inbatch_negative false
```

**Pairwise LLM embedding finetune**
- add query_instruction
  - "Given a query and a relevant document, retrieve the document that are pertinent to the query\nQuery: "
- use the appropriate pooling_method
  - `last`
- maybe we need to reduce the batch_size due to large model size
- set `use_lora` to True if you want to use lora

```shell
MODEL_NAME="intfloat/e5-mistral-7b-instruct"
TRAIN_DATA="/t2_ranking.jsonl"
OUTPUT_DIR="/t2_output"

torchrun --nproc_per_node 1 \
  -m retrievals.pipelines.embed \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --model_name_or_path $MODEL_NAME \
  --pooling_method last \
  --do_train \
  --train_data $TRAIN_DATA \
  --positive_key positive \
  --negative_key negative \
  --use_lora True \
  --query_instruction "Retrieve the possible answer for query.\nQuery: " \
  --document_instruction 'Document: ' \
  --learning_rate 2e-4 \
  --bf16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --dataloader_drop_last True \
  --query_max_length 64 \
  --document_max_length 256 \
  --train_group_size 4 \
  --logging_steps 100 \
  --temperature 0.02 \
  --use_inbatch_negative false \
  --save_total_limit 1
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
  --learning_rate 2e-5 \
  --fp16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 64 \
  --dataloader_drop_last True \
  --max_length 512 \
  --save_total_limit 1 \
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
  --learning_rate 1e-4 \
  --bf16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 64 \
  --dataloader_drop_last True \
  --max_length 256 \
  --train_group_size 4 \
  --unfold_each_positive false \
  --save_total_limit 1 \
  --logging_steps 100 \
  --use_inbatch_negative false
```

**LLM reranking**
- `AutoModelForRanking.from_pretrained(model_name_or_path, causal_lm=True)`
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


## Common questions
- If grad_norm during training is always zero, consider to change fp16 or bf16
- If the fine-tuned embedding performance during inference is worse, check whether the pooling_method is correct, and the prompt is the same as training

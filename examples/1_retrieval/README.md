# Retrieval Examples

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


If you want to finetune a LLM for embedding:
- Use causal LM
  - AutoModelForEmbedding.from_pretrained(model_name_or_path, causal_lm=True)
- add query_instruction
  - "Given a query and a relevant document, retrieve the document that are pertinent to the query\nQuery: "

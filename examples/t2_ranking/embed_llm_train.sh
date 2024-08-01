MODEL_NAME="Qwen/Qwen2-1.5B-Instruct"
TRAIN_DATA="/t2_ranking.jsonl"
OUTPUT_DIR="/t2_output"

torchrun --nproc_per_node 1 \
  -m retrievals.pipelines.embed \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --model_name_or_path $MODEL_NAME \
  --pooling_method last \
  --do_train \
  --data_name_or_path $TRAIN_DATA \
  --positive_key positive \
  --negative_key negative \
  --use_lora True \
  --query_instruction "Retrieve the possible answer for query.\nQuery: " \
  --document_instruction 'Document: ' \
  --learning_rate 1e-4 \
  --bf16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --dataloader_drop_last True \
  --query_max_length 64 \
  --document_max_length 256 \
  --train_group_size 4 \
  --logging_strategy steps \
  --logging_steps 100 \
  --temperature 0.02 \
  --use_inbatch_negative false \
  --save_total_limit 1

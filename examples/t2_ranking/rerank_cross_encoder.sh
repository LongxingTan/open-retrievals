MODEL_NAME="BAAI/bge-reranker-base"
TRAIN_DATA="t2_ranking.jsonl"
OUTPUT_DIR="t2_rank_output"

torchrun --nproc_per_node 1 \
  -m retrievals.pipelines.rerank \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --model_name_or_path $MODEL_NAME \
  --model_type cross-encoder \
  --do_train \
  --data_name_or_path $TRAIN_DATA \
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

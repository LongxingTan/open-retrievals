MODEL_NAME="BAAI/bge-base-en-v1.5"
TRAIN_DATA="Tevatron/scifact"
OUTPUT_DIR="./scifact/ft_out"

torchrun --nproc_per_node 1 \
  -m retrievals.pipelines.embed \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --model_name_or_path $MODEL_NAME \
  --do_train \
  --data_name_or_path $TRAIN_DATA \
  --positive_key positive_passages \
  --negative_key negative_passages \
  --pooling_method cls \
  --loss_fn infonce \
  --use_lora False \
  --query_instruction "" \
  --document_instruction "" \
  --learning_rate 1e-5 \
  --fp16 \
  --num_train_epochs 15 \
  --per_device_train_batch_size 64 \
  --dataloader_drop_last True \
  --query_max_length 64 \
  --document_max_length 512 \
  --train_group_size 2 \
  --logging_steps 100 \
  --temperature 0.02 \
  --save_total_limit 1 \
  --use_inbatch_negative false

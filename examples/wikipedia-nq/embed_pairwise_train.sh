MODEL_NAME="BAAI/bge-base-en-v1.5"
TRAIN_DATA="./wikipedia-nq/nq-train-data/train_data.json"
OUTPUT_DIR="./wikipedia-nq/nq_model"


torchrun --nproc_per_node 1 \
  -m retrievals.pipelines.embed \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --model_name_or_path $MODEL_NAME \
  --do_train \
  --data_name_or_path $TRAIN_DATA \
  --query_key query \
  --positive_key positives \
  --negative_key negatives \
  --pooling_method cls \
  --loss_fn infonce \
  --use_lora False \
  --query_instruction "" \
  --document_instruction "" \
  --learning_rate 1e-5 \
  --fp16 \
  --num_train_epochs 10 \
  --per_device_train_batch_size 128 \
  --dataloader_drop_last True \
  --query_max_length 64 \
  --document_max_length 256 \
  --train_group_size 2 \
  --logging_steps 100 \
  --temperature 0.02 \
  --save_total_limit 1 \
  --use_inbatch_negative false

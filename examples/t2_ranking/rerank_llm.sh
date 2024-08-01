MODEL_NAME="Qwen/Qwen2-1.5B-Instruct"
TRAIN_DATA="t2_ranking.jsonl"
OUTPUT_DIR="t2_output"

torchrun --nproc_per_node 1 \
    -m retrievals.pipelines.rerank \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --model_name_or_path $MODEL_NAME \
    --model_type llm \
    --causal_lm True \
    --use_lora True \
    --data_name_or_path $TRAIN_DATA \
    --task_prompt "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'." \
    --query_instruction "A: " \
    --document_instruction 'B: ' \
    --positive_key positive \
    --negative_key negative \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --dataloader_drop_last True \
    --max_len 256 \
    --train_group_size 4 \
    --logging_steps 10 \
    --save_steps 20000 \
    --save_total_limit 1 \
    --bf16

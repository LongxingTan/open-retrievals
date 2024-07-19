ENCODE_QUERY_DIR=./query-embeddings
MODEL_NAME="Qwen/Qwen2-1.5B-Instruct"
LORA_DIR=./ft_llm_out
QUERY=Tevatron/scifact/dev
mkdir -p $ENCODE_QUERY_DIR

python -m retrievals.pipelines.embed \
    --model_name_or_path $MODEL_NAME \
    --lora_path $LORA_DIR \
    --pooling_method last \
    --output_dir $ENCODE_QUERY_DIR \
    --encoding_save_file query.pkl \
    --do_encode \
    --bf16 \
    --per_device_eval_batch_size 256 \
    --data_name_or_path $QUERY \
    --query_key query \
    --query_instruction "Retrieve the possible answer for query.\nQuery: " \
    --query_max_length 64 \
    --is_query true

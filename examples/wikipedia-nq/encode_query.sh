ENCODE_QUERY_DIR=./embeddings-nq-queries
MODEL_DIR=./nq-model
QUERY=./nq-test-queries.json
mkdir $ENCODE_QUERY_DIR

python -m retrievals.pipelines.embed \
    --model_name_or_path $MODEL_DIR \
    --output_dir $ENCODE_QUERY_DIR \
    --encoding_save_file query.pkl \
    --do_encode \
    --fp16 \
    --per_device_eval_batch_size 256 \
    --data_name_or_path $QUERY \
    --query_key text \
    --query_max_length 64 \
    --is_query true

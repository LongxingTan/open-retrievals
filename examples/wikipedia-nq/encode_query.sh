ENCODE_QUERY_DIR=embeddings-nq-queries
OUT_DIR=temp
MODEL_DIR=model-nq
QUERY=nq-test-queries.json
mkdir $ENCODE_QUERY_DIR

python -m run \
    --model_name_or_path $MODEL_DIR \
    --output_dir $OUT_DIR \
    --do_encode \
    --fp16 \
    --per_device_eval_batch_size 256 \
    --encode_in_path $QUERY \
    --is_query true

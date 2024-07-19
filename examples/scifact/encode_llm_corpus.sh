ENCODE_CORPUS_DIR=./scifact/corpus-embeddings
MODEL_NAME="intfloat/e5-mistral-7b-instruct"
LORA_DIR=./ft_llm_out
CORPUS=Tevatron/scifact-corpus
mkdir -p $ENCODE_CORPUS_DIR

python -m retrievals.pipelines.embed \
    --model_name_or_path $MODEL_NAME \
    --lora_path $LORA_DIR \
    --pooling_method last \
    --output_dir $ENCODE_CORPUS_DIR \
    --encoding_save_file corpus.pkl \
    --do_encode \
    --bf16 \
    --per_device_eval_batch_size 128 \
    --data_name_or_path $CORPUS \
    --query_key text \
    --document_instruction "Document: " \
    --document_max_length 256 \
    --is_query false

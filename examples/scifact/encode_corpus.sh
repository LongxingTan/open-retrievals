ENCODE_CORPUS_DIR=./scifact/corpus-embeddings
MODEL_DIR="./scifact/ft_out"
CORPUS=Tevatron/scifact-corpus
mkdir $ENCODE_CORPUS_DIR


python -m retrievals.pipelines.embed \
    --model_name_or_path $MODEL_DIR \
    --output_dir $ENCODE_CORPUS_DIR \
    --encode_save_file corpus.pkl \
    --do_encode \
    --fp16 \
    --per_device_eval_batch_size 256 \
    --data_name_or_path $CORPUS \
    --query_key text \
    --document_max_length 512 \
    --is_query false

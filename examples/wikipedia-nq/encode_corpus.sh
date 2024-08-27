# wiki corpus is a folder with multiple jsons
ENCODE_CORPUS_DIR=./embeddings-nq-corpus
MODEL_DIR=./nq-model
CORPUS_DIR=./wikipedia-corpus

mkdir $ENCODE_CORPUS_DIR
for s in $(seq -f "%02g" 0 21)
do
python -m retrievals.pipelines.embed \
    --model_name_or_path $MODEL_DIR \
    --output_dir $ENCODE_CORPUS_DIR \
    --encoding_save_file $s.pkl \
    --do_encode \
    --fp16 \
    --per_device_eval_batch_size 256 \
    --data_name_or_path $CORPUS/docs$s.json \
    --query_key text \
    --is_query false \
    --document_max_length 128
done

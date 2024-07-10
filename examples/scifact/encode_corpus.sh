# Tevatron/scifact-corpus

ENCODE_DIR=embeddings-nq-corpus
OUT_DIR=temp
MODEL_DIR=model-nq
CORPUS_DIR=wikipedia-corpus

mkdir $ENCODE_DIR
for s in $(seq -f "%02g" 0 21)
do
python -m run \
    --model_name_or_path $MODEL_DIR \
    --output_dir $OUT_DIR \
    --do_encode \
    --fp16 \
    --per_device_eval_batch_size 256 \
    --encode_in_path $CORPUS_DIR/docs$s.json \
    --encode_save_path $ENCODE_DIR/$s.index \
    --is_query false
done

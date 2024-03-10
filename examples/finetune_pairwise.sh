CUDA_VISIBLE_DEVICES=0 python wiki_finetune.py \
    --model_name_or_path bert-base-multilingual-uncased \
    --train_data /root/kaggle101/examples/FlagEmbedding/examples/finetune/toy_finetune_data.jsonl \
    --output_dir modeloutput

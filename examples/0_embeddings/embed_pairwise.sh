CUDA_VISIBLE_DEVICES=0 python embed_pairwise_finetune.py \
    --model_name_or_path bert-base-multilingual-uncased \
    --train_data ./example_data/toy_finetune_data.jsonl \
    --output_dir modeloutput
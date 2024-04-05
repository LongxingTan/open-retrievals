# torchrun --nnodes 1 --nproc-per-node 4
# deepspeed --include localhost:0,1,2,3
# CUDA_VISIBLE_DEVICES=1,2,3 python
# accelerate launch --config_file conf_ds.yaml \

accelerate launch --config_file conf/conf_llm.yaml \
    finetune_llm_embed.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --train_data /root/kaggle101/examples/FlagEmbedding/examples/finetune/toy_finetune_data.jsonl \
    --output_dir modeloutput \

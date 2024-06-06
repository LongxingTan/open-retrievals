# torchrun --nnodes 1 --nproc-per-node 4
# deepspeed --include localhost:0,1,2,3
# CUDA_VISIBLE_DEVICES=1,2,3 python
# accelerate launch --config_file conf_ds.yaml \

accelerate launch \
    --config_file conf_llm.yaml \
    llm_finetune_for_embed.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --train_data  \
    --output_dir modeloutput \

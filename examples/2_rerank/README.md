# Ranking Examples

## prepare data
```python
from datasets import load_dataset

dataset = load_dataset("C-MTEB/T2Reranking", split="dev")
ds = dataset.train_test_split(test_size=0.1, seed=42)

ds_train = (
    ds["train"]
    .filter(lambda x: len(x["positive"]) > 0 and len(x["negative"]) > 0)
)

ds_train.to_json("t2_ranking.jsonl", force_ascii=False)
```

## train
cross encoder
```shell
python train_cross_encoder.py
```


```shell
MODEL_NAME="BAAI/bge-reranker-base"
TRAIN_DATA="/t2_ranking.jsonl"
OUTPUT_DIR="/t2_output"

torchrun --nproc_per_node 1 \
  -m retrievals.pipelines.rerank \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --model_name_or_path $MODEL_NAME \
  --model_type cross-encoder \
  --do_train \
  --train_data $TRAIN_DATA \
  --positive_key positive \
  --negative_key negative \
  --learning_rate 3e-5 \
  --fp16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 32 \
  --dataloader_drop_last True \
  --max_length 512 \
  --train_group_size 3 \
  --logging_steps 100
```


colbert
```shell
python train_colbert.py
```


LLM
- AutoModelForRanking.from_pretrained(model_name_or_path, causal_lm = True)
- Prompt: "Given a query with a relevant body, determine whether the document is pertinent to the query by providing a prediction of either 'Yes' or 'No'."

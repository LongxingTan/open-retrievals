# Ranking Examples

## prepare data
```python
from datasets import load_dataset

dataset = load_dataset("C-MTEB/T2Reranking", split="dev")
ds = dataset.train_test_split(test_size=0.1)

ds_train = (
    ds["train"]
    .filter(lambda x: len(x["positive"]) > 0 and len(x["negative"]) > 0)
)

ds_train.to_json("t2_ranking.jsonl", force_ascii=False)
```

## train
- cross encoder demo
```shell
python train_cross_encoder.py
```

- cross encoder
```shell

```


- colbert
```shell
python train_colbert.py
```

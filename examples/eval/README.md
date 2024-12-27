# Evaluation

**Prerequisites**
```shell
pip install datasets mteb[beir]
pip install open-retrievals[eval]
```

**Eval**
```shell
python eval_retrieval.py  --model_name stella-base-zh  --output_dir ./zh_results/stella-base
```


## Reference

- https://github.com/beir-cellar/beir
- https://github.com/AmenRa/ranx
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

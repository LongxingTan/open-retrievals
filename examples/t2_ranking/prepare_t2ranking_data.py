from datasets import load_dataset

dataset = load_dataset("C-MTEB/T2Reranking", split="dev")
ds = dataset.train_test_split(test_size=0.1, seed=42)

ds_train = ds["train"].filter(lambda x: len(x["positive"]) > 0 and len(x["negative"]) > 0)
ds_train.to_json("t2_ranking.jsonl", force_ascii=False)

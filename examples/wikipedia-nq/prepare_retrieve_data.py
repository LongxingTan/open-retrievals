import json
import os
from argparse import ArgumentParser

from tqdm import tqdm
from transformers import AutoTokenizer

parser = ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--encode", type=bool, required=False, default=False)
parser.add_argument("--tokenizer", type=str, required=False, default="bert-base-uncased")
parser.add_argument("--minimum-negatives", type=int, required=False, default=8)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

data = json.load(open(args.input))

if not os.path.exists(args.output):
    os.makedirs(args.output)

with open(os.path.join(args.output, "train_data.json"), "w") as f:
    for idx, item in enumerate(tqdm(data)):
        group = {}
        query = item["question"]
        if args.encode:
            query = tokenizer.encode(query, add_special_tokens=False, max_length=256, truncation=True)
        group["query"] = query

        positives = item["positive_ctxs"]
        negatives = item["hard_negative_ctxs"]
        group["positives"] = []
        group["negatives"] = []
        for pos in positives:
            text = pos["title"] + " " + pos["text"]
            if args.encode:
                text = tokenizer.encode(text, add_special_tokens=False, max_length=256, truncation=True)
            group["positives"].append(text)
        for neg in negatives:
            text = neg["title"] + " " + neg["text"]
            if args.encode:
                text = tokenizer.encode(text, add_special_tokens=False, max_length=256, truncation=True)
            group["negatives"].append(text)
        if len(group["negatives"]) >= args.minimum_negatives and len(group["positives"]) >= 1:
            f.write(json.dumps(group) + "\n")

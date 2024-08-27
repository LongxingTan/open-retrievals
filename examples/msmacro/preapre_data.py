import argparse
import json
import os

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description='prepare json data for msmacro')
parser.add_argument('--model_name', default='bert-base-uncased', type=str, help='which model to use')
parser.add_argument(
    '--triplet_dir',
    default='./triples.train.small.tsv',
    type=str,
    help='the input path',
)
parser.add_argument('--output_dir', default='./ms_train.jsonl', type=str, help='output directory')
args = parser.parse_args()


def prepare_json():
    with open(args.triplet_dir) as f:
        with open(args.output_dir, 'w') as out_file:
            for line in f:
                query, pos, neg = line.strip().split("\t")
                data = {'query': query, 'positive': [pos], 'negative': [neg]}
                out_file.write(json.dumps(data) + '\n')
    return


if __name__ == "__main__":
    prepare_json()

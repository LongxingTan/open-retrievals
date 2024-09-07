"""Build hard negative for retrieval/reranking"""

import os
import random
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)

args = parser.parse_args()


def load_ranking(rank_file, relevance, num_hn_sample, depth):
    """
    rank_file: the output file from retrieve/rerank, each line with query_id, passage_id, score, rank
    relevance: hash table record the true passage_id for each query_id
    n_sample: numbers of hard negative
    depth: the retrieval/rerank chosen numbers from the rank_file result
    """
    with open(rank_file) as f:
        lines = iter(f)
        qid, pid, _, _ = next(lines).strip().split()

        curr_q = qid
        negatives = [] if pid in relevance[pid] else [pid]

        while True:
            try:
                q, p, _, _ = next(lines).strip().split()
                if q != curr_q:
                    # 开始下一个query_id之前, 首先把之前的返回；再重新从q开始
                    negatives = negatives[:depth]
                    random.shuffle(negatives)
                    yield curr_q, relevance[curr_q], negatives[:num_hn_sample]

                    curr_q = q
                    negatives = [] if p in relevance[q] else [p]
                else:
                    if p not in relevance[qid]:
                        negatives.append(p)

            except StopIteration:
                negatives = negatives[:depth]
                random.shuffle(negatives)
                yield curr_q, relevance[curr_q], negatives[:num_hn_sample]
                return


if __name__ == "__main__":
    os.makedirs(args.output, exist_ok=True)

    with open("hard_negative.json", "w") as f:
        for x in load_ranking():
            f.write(x + "\n")

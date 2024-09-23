import argparse
import json
from collections import defaultdict

from datasets import load_dataset

from retrievals.metrics import get_fbeta, get_mrr, get_ndcg


def transfer_index_to_id(save_path):
    corpus_dataset = load_dataset("Tevatron/scifact-corpus", "default")["train"]
    index_to_docid = {index: example["docid"] for index, example in enumerate(corpus_dataset)}

    query_dataset = load_dataset("Tevatron/scifact", 'default')['dev']
    index_to_queryid = {index: example['query_id'] for index, example in enumerate(query_dataset)}

    with open(args.ranking_path) as f, open(save_path, 'w') as f_out:
        for line in f:
            qidx, pidx, score, rank = [x for x in line.strip().split("\t")]
            qid = index_to_queryid[int(qidx)]
            pid = index_to_docid[int(pidx)]
            f_out.write(f'{qid}\t{pid}\t{score}\t{rank}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qrel_path",
        default="./scifact/dev_qrels.txt",
    )
    parser.add_argument(
        "--ranking_path",
        default="./scifact/recall_out/retrieval.txt",
    )
    parser.add_argument(
        "--save_path",
        default="./scifact/recall_out/retrieval_id.txt",
    )
    args = parser.parse_args()

    transfer_index_to_id(args.save_path)

    qid2positives = defaultdict(list)
    with open(args.qrel_path) as f:
        for line in f:
            qid, _, pid, label = [int(x) for x in line.strip().split()]
            assert label == 1
            qid2positives[qid].append(pid)

    qid2ranking = defaultdict(list)
    with open(args.save_path) as f:
        for line in f:
            qid, pid, score, rank = [x for x in line.strip().split("\t")]
            qid = int(qid)
            pid = int(pid)
            qid2ranking[qid].append(pid)

    results = get_mrr(qid2positives, qid2ranking, cutoff_rank=10)
    results.update(get_fbeta(qid2positives, qid2ranking, cutoff_ranks=[10]))
    results.update(get_ndcg(qid2positives, qid2ranking, cutoff_rank=10))

    print(json.dumps(results, indent=4))

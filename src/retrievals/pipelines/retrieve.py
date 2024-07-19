"""Retrieve pipeline
retrieve: one query match with top_k document: [{'id': 'doc_id', 'similarity': 0.8}, {'id': 'doc_id', "similarity': 0.9]
"""

import glob
import logging
import os
import pickle
from argparse import ArgumentParser

import numpy as np

from ..models.retrieval_auto import AutoModelForRetrieval, FaissRetrieval

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def load_pickle(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup


def save_ranking(dists, indices, ranking_file, query_ids):
    """
    save format: query_id, doc_id, score, rank
    """
    os.makedirs(os.path.dirname(ranking_file), exist_ok=True)

    with open(ranking_file, 'w') as f:
        for qid, score, index in zip(query_ids, dists, indices):
            score_list = [(s, idx) for s, idx in zip(score, index)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for rank, (s, idx) in enumerate(score_list, start=1):
                f.write(f'{qid}\t{idx}\t{s}\t{rank}\n')


def retrieve():
    parser = ArgumentParser()
    parser.add_argument("--query_reps", required=True)
    parser.add_argument("--passage_reps", required=True)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_ranking_file", required=True)
    parser.add_argument("--save_text", action="store_true")

    args = parser.parse_args()

    index_files = glob.glob(args.passage_reps)
    logger.info(f"Pattern match found {len(index_files)} files; loading them into index.")

    passage_reps0, passage_lookup0 = load_pickle(index_files[0])
    retriever = FaissRetrieval(passage_reps0)

    shards = [(passage_reps0, passage_lookup0)]
    for i in range(1, len(index_files)):
        passage_reps, passage_lookup = load_pickle(index_files[i])
        shards.append((passage_reps, passage_lookup))

    lookups = []
    for passage_reps, passage_lookup in shards:
        retriever.add(passage_reps)
        lookups += passage_lookup

    query_reps, query_lookup = load_pickle(args.query_reps)

    dists, indices = retriever.search(query_embeddings=query_reps, top_k=args.top_k)
    doc_ids = np.array([[int(lookups[idx]) for idx in indices] for indices in indices])

    save_ranking(dists, doc_ids, args.save_ranking_file, query_lookup)


if __name__ == "__main__":
    retrieve()

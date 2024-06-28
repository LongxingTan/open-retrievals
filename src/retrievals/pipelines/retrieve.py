import glob
import logging
from argparse import ArgumentParser

import torch

from ..models.retrieval_auto import AutoModelForRetrieval

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def retrieve():
    parser = ArgumentParser()
    parser.add_argument("--query_reps", required=True)
    parser.add_argument("--passage_reps", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--save_ranking_to", required=True)
    parser.add_argument("--save_text", action="store_true")

    args = parser.parse_args()

    index_files = glob.glob(args.passage_reps)
    logger.info(f"Pattern match found {len(index_files)} files; loading them into index.")

    retriever = AutoModelForRetrieval()

    query_embed = torch.load(args.query_reps)

    dists, indices = retriever.similarity_search(query_embed=query_embed, index_path=index_files[0], top_k=args.top_k)
    print(indices.shape)

    retriever.save_ranking(dists, indices, args.save_ranking_to)


if __name__ == "__main__":
    retrieve()

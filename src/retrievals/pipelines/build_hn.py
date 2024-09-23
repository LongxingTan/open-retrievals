"""Build hard negative for retrieval/reranking"""

import json
import logging
import os
import random
from argparse import ArgumentParser

from ..models.embedding_auto import AutoModelForEmbedding
from ..models.retrieval_auto import AutoModelForRetrieval, FaissRetrieval

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument('--candidate_pool', default=None, type=str)
    parser.add_argument('--model_name_or_path', default="BAAI/bge-base-en", type=str)
    parser.add_argument('--range_for_sampling', default="10-210", type=str, help="range to sample negatives")
    parser.add_argument('--negative_number', default=15, type=int, help='the number of negatives')
    parser.add_argument('--query_instruction_for_retrieval', default="")
    parser.add_argument('--positive_key', type=str, default="positive")
    parser.add_argument('--negative_key', type=str, default="negative")
    return parser.parse_args()


def add_neg_for_ranking():
    os.makedirs(args.output_file, exist_ok=True)

    with open("hard_negative.json", "w") as f:
        for x in load_ranking():
            f.write(x + "\n")


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


def get_corpus(candidate_pool):
    corpus = []
    for line in open(candidate_pool):
        line = json.loads(line.strip())
        corpus.append(line['text'])
    return corpus


def add_neg(
    model,
    input_file,
    candidate_pool,
    output_file,
    sample_range,
    negative_number,
    positive_key: str = 'pos',
    negative_key: str = 'neg',
):
    corpus = []
    queries = []
    train_data = []
    for line in open(input_file):
        line = json.loads(line.strip())
        train_data.append(line)
        corpus.extend(line[positive_key])  # if not candidate_pool, all pos will be considered as corpus
        if negative_key in line:
            corpus.extend(line[negative_key])
        queries.append(line['query'])

    if candidate_pool is not None:
        if not isinstance(candidate_pool, list):
            candidate_pool = get_corpus(candidate_pool)
        corpus = list(set(candidate_pool))
    else:
        corpus = list(set(corpus))

    logger.info(f'Encoding for query of {len(queries)}')
    query_embeds = model.encode_queries(queries, batch_size=256)

    logger.info(f'Indexing and search of {len(corpus)}')
    index = model.build_index(inputs=corpus)

    retriever = FaissRetrieval(corpus_index=index)
    _, all_indices = retriever.search(query_embeds, top_k=sample_range[-1])
    assert len(all_indices) == len(train_data)

    for i, data in enumerate(train_data):
        query = data['query']
        idxs = all_indices[i][sample_range[0] : sample_range[1]]
        filtered_idx = []
        for idx in idxs:
            if idx == -1:
                break
            if corpus[idx] not in data[positive_key] and corpus[idx] != query:
                filtered_idx.append(idx)

        if len(filtered_idx) > negative_number:
            filtered_idx = random.sample(filtered_idx, negative_number)

        data[negative_key] = [corpus[idx] for idx in filtered_idx]

    with open(output_file, 'w') as f:
        for data in train_data:
            if len(data[negative_key]) < negative_number:
                samples = random.sample(corpus, negative_number - len(data[negative_number]) + len(data[positive_key]))
                samples = [sent for sent in samples if sent not in data[positive_key]]
                data[negative_key].extend(samples[: negative_number - len(data[negative_key])])
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    logger.info(f"With hard negative file saved in {output_file}")


if __name__ == "__main__":
    args = parse_args()
    sample_range = args.range_for_sampling.split('-')
    sample_range = [int(x) for x in sample_range]

    model = AutoModelForEmbedding.from_pretrained(
        args.model_name_or_path, query_instruction=args.query_instruction_for_retrieval
    )

    add_neg(
        model=model,
        input_file=args.input_file,
        candidate_pool=args.candidate_pool,
        output_file=args.output_file,
        sample_range=sample_range,
        negative_number=args.negative_number,
        positive_key=args.positive_key,
        negative_key=args.negative_key,
    )

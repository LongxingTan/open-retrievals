import logging
from typing import Dict, List, Tuple, Union


def get_mrr(qid2positive: Dict[str, List[str]], qid2ranking: Dict[str, List[str]], cutoff_rank: int = 10):
    """
    qid2positive: {qid: [pos1, pos2]}
    qid2ranking: {qid: [rank1, rank2, rank3]}
    """
    qid2mrr = dict()

    for qid in qid2positive:
        positives = qid2positive[qid]
        ranked_doc_ids = qid2ranking[qid]

        for rank, doc_id in enumerate(ranked_doc_ids, start=1):
            if doc_id in positives:
                if rank <= cutoff_rank:
                    qid2mrr[qid] = 1.0 / rank
                break
    return {f"mrr@{cutoff_rank}": sum(qid2mrr.values()) / len(qid2ranking.keys())}

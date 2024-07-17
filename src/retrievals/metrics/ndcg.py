import math
from typing import Dict, List


def get_ndcg(qid2positive: Dict[str, List[str]], qid2ranking: Dict[str, List[str]], cutoff_rank: int = 10):
    """
    qid2positive: {qid: [pos1_doc_id, pos2_doc_id]}
    qid2ranking: {qid: [rank1_doc_id, rank2_doc_id, rank3_doc_id]}
    """

    def dcg(relevance_scores: List[int], cutoff: int) -> float:
        return sum([rel / math.log2(idx + 2) for idx, rel in enumerate(relevance_scores[:cutoff])])

    qid2ndcg = dict()

    for qid in qid2positive:
        positives = set(qid2positive[qid])
        ranked_doc_ids = qid2ranking[qid]

        relevance_scores = [1 if doc_id in positives else 0 for doc_id in ranked_doc_ids]
        actual_dcg = dcg(relevance_scores, cutoff_rank)

        ideal_relevance_scores = sorted(relevance_scores, reverse=True)
        ideal_dcg = dcg(ideal_relevance_scores, cutoff_rank)

        if ideal_dcg > 0:
            qid2ndcg[qid] = actual_dcg / ideal_dcg
        else:
            qid2ndcg[qid] = 0.0

    return {f"ndcg@{cutoff_rank}": sum(qid2ndcg.values()) / len(qid2ranking.keys())}

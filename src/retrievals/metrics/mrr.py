from typing import Dict, List, Tuple, Union


def get_mrr(qid2positive: Dict[str, List[str]], qid2ranking: Dict[str, List[str]], cutoff_ranks: List[int] = [10]):
    """
    qid2positive: {qid: [pos1_doc_id, pos2_doc_id]}
    qid2ranking: {qid: [rank1_doc_id, rank2_doc_id, rank3_doc_id]}
    """
    qid2mrr = {cutoff_rank: {} for cutoff_rank in cutoff_ranks}

    for qid in qid2positive:
        positives = qid2positive[qid]
        ranked_doc_ids = qid2ranking[qid]

        for cutoff_rank in cutoff_ranks:

            for rank, doc_id in enumerate(ranked_doc_ids, start=1):
                if doc_id in positives:
                    if rank <= cutoff_rank:
                        qid2mrr[cutoff_rank][qid] = 1.0 / rank
                    break
    return {
        f"mrr@{cutoff_rank}": sum(qid2mrr[cutoff_rank].values()) / len(qid2ranking.keys())
        for cutoff_rank in cutoff_ranks
    }

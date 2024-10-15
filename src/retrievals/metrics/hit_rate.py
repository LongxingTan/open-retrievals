from typing import Dict, List


def get_hit_rate(qid2positive: Dict[str, List[str]], qid2ranking: Dict[str, List[str]], cutoff_ranks: List[int] = [10]):
    """
    qid2positive (order doesn't matter): {qid: [pos1_doc_id, pos2_doc_id]}
    qid2ranking (order does matter): {qid: [rank1_doc_id, rank2_doc_id, rank3_doc_id]}
    """

    def hit_rate(positives_ids: List[str], ranked_doc_ids: List[str], cutoff: int) -> float:
        """
        Calculate hit rate at the specified cutoff
        """
        hits = 0

        for doc_id in ranked_doc_ids[:cutoff]:
            if doc_id in positives_ids:
                hits += 1

        return hits / cutoff if cutoff > 0 else 0.0

    qid2hr = {cutoff_rank: {} for cutoff_rank in cutoff_ranks}

    for qid in qid2positive:
        positives_ids = qid2positive[qid]
        ranked_doc_ids = qid2ranking[qid]
        for cutoff_rank in cutoff_ranks:
            qid2hr[cutoff_rank][qid] = hit_rate(positives_ids, ranked_doc_ids, cutoff_rank)

    return {
        f"hit_rate@{cutoff_rank}": sum(qid2hr[cutoff_rank].values()) / len(qid2hr) if qid2hr else 0.0
        for cutoff_rank in cutoff_ranks
    }

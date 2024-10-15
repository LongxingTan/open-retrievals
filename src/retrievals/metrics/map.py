from typing import Dict, List


def get_map(qid2positive: Dict[str, List[str]], qid2ranking: Dict[str, List[str]], cutoff_ranks: List[int] = [10]):
    """
    qid2positive (order doesn't matter): {qid: [pos1_doc_id, pos2_doc_id]}
    qid2ranking (order does matter): {qid: [rank1_doc_id, rank2_doc_id, rank3_doc_id]}
    """

    def average_precision(positives_ids: List[str], ranked_doc_ids: List[str], cutoff: int) -> float:
        """
        Average of precision for each cut_off
        """
        hits = 0
        sum_precisions = 0.0

        for rank, doc_id in enumerate(ranked_doc_ids[:cutoff], start=1):
            if doc_id in positives_ids:
                hits += 1
                sum_precisions += hits / rank

        return sum_precisions / min(len(positives_ids), cutoff) if positives_ids else 0.0

    qid2map = {cutoff_rank: {} for cutoff_rank in cutoff_ranks}

    for qid in qid2positive:
        positives_ids = qid2positive[qid]
        ranked_doc_ids = qid2ranking[qid]
        for cutoff_rank in cutoff_ranks:
            qid2map[cutoff_rank][qid] = average_precision(positives_ids, ranked_doc_ids, cutoff_rank)

    return {
        f"map@{cutoff_rank}": sum(qid2map[cutoff_rank].values()) / len(qid2ranking.keys())
        for cutoff_rank in cutoff_ranks
    }

from typing import Dict, List


def get_map(qid2positive: Dict[str, List[str]], qid2ranking: Dict[str, List[str]], cutoff_rank: int = 10):
    """
    qid2positive: {qid: [pos1_doc_id, pos2_doc_id]}
    qid2ranking: {qid: [rank1_doc_id, rank2_doc_id, rank3_doc_id]}
    """

    def average_precision(positives: List[str], ranked_doc_ids: List[str], cutoff: int) -> float:
        hits = 0
        sum_precisions = 0.0

        for rank, doc_id in enumerate(ranked_doc_ids[:cutoff], start=1):
            if doc_id in positives:
                hits += 1
                sum_precisions += hits / rank

        return sum_precisions / len(positives) if positives else 0.0

    qid2map = dict()

    for qid in qid2positive:
        positives = qid2positive[qid]
        ranked_doc_ids = qid2ranking[qid]
        qid2map[qid] = average_precision(positives, ranked_doc_ids, cutoff_rank)

    return {f"map@{cutoff_rank}": sum(qid2map.values()) / len(qid2ranking.keys())}

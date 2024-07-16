from typing import Dict, List


def get_recall(qid2positive: Dict[str, List[str]], qid2ranking: Dict[str, List[str]], cutoff_ranks: List[int] = [10]):
    qid2recall = {cutoff_rank: {} for cutoff_rank in cutoff_ranks}
    num_samples = len(qid2ranking.keys())

    for qid in qid2positive:
        positives = qid2positive[qid]
        ranked_doc_ids = qid2ranking[qid]

        for rank, doc_id in enumerate(ranked_doc_ids, start=1):
            if doc_id in positives:
                for cutoff_rank in cutoff_ranks:
                    if rank <= cutoff_rank:
                        qid2recall[cutoff_rank][qid] = qid2recall[cutoff_rank].get(qid, 0) + 1.0 / len(positives)

    return {
        f"recall@{cutoff_rank}": sum(qid2recall[cutoff_rank].values()) / num_samples for cutoff_rank in cutoff_ranks
    }

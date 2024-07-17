import math
import unittest
from typing import Dict, List

from src.retrievals.metrics.ndcg import get_ndcg


class TestGetNDCG(unittest.TestCase):
    def test_get_ndcg(self):
        # Define test cases
        qid2positive = {'q1': ['d1', 'd3'], 'q2': ['d1'], 'q3': ['d2', 'd3']}

        qid2ranking = {'q1': ['d1', 'd2', 'd3', 'd4'], 'q2': ['d2', 'd1', 'd3'], 'q3': ['d1', 'd3', 'd2']}

        cutoff_rank = 3

        # Expected NDCG Calculation
        def compute_expected_ndcg(relevance_scores: List[int], cutoff: int) -> float:
            dcg_val = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance_scores[:cutoff]))
            ideal_relevance_scores = sorted(relevance_scores, reverse=True)
            ideal_dcg_val = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_relevance_scores[:cutoff]))
            return dcg_val / ideal_dcg_val if ideal_dcg_val > 0 else 0.0

        expected_ndcg = {
            'q1': compute_expected_ndcg([1, 0, 1, 0], cutoff_rank),
            'q2': compute_expected_ndcg([0, 1, 0], cutoff_rank),
            'q3': compute_expected_ndcg([0, 1, 1], cutoff_rank),
        }

        expected_mean_ndcg = sum(expected_ndcg.values()) / len(expected_ndcg)

        # Call the function
        result = get_ndcg(qid2positive, qid2ranking, cutoff_rank)

        # Check if the result is correct
        self.assertAlmostEqual(result[f'ndcg@{cutoff_rank}'], expected_mean_ndcg, places=4)

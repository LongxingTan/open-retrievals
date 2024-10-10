import unittest

from src.retrievals.metrics.mrr import get_mrr


class TestGetMRR(unittest.TestCase):
    def test_get_mrr(self):
        qid2positive = {'q1': ['d1', 'd3'], 'q2': ['d1'], 'q3': ['d2', 'd3']}
        qid2ranking = {'q1': ['d1', 'd2', 'd3', 'd4'], 'q2': ['d2', 'd1', 'd3'], 'q3': ['d1', 'd3', 'd2']}
        cutoff_ranks = [3]
        expected_mrr = {
            'q1': 1.0 / 1,
            'q2': 1.0 / 2,
            'q3': 1.0 / 2,
        }

        expected_mean_mrr = sum(expected_mrr.values()) / len(expected_mrr)
        result = get_mrr(qid2positive, qid2ranking, cutoff_ranks)
        self.assertAlmostEqual(result[f'mrr@{cutoff_ranks[0]}'], expected_mean_mrr, places=4)

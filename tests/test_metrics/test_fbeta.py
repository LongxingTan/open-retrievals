import unittest

from src.retrievals.metrics.fbeta import get_fbeta


class TestGetFbeta(unittest.TestCase):
    def test_get_fbeta(self):
        qid2positive = {'q1': ['d1', 'd3'], 'q2': ['d1'], 'q3': ['d2', 'd3']}

        qid2ranking = {'q1': ['d1', 'd2', 'd3', 'd4'], 'q2': ['d2', 'd1', 'd3'], 'q3': ['d1', 'd3', 'd2']}

        cutoff_ranks = [2, 3]

        expected_recall_at_2 = {
            'q1': 1.0 / 2,
            'q2': 1.0 / 1,
            'q3': 1.0 / 2,
        }

        expected_recall_at_3 = {
            'q1': 2.0 / 2,  # d1 and d3 are in top 3
            'q2': 1.0 / 1,  # d1 is in top 3
            'q3': 2.0 / 2,  # d3 is in top 3
        }

        expected_mean_recall_at_2 = sum(expected_recall_at_2.values()) / len(expected_recall_at_2)
        expected_mean_recall_at_3 = sum(expected_recall_at_3.values()) / len(expected_recall_at_3)

        result = get_fbeta(qid2positive, qid2ranking, cutoff_ranks)
        print(result)

        self.assertAlmostEqual(result['recall@2'], expected_mean_recall_at_2, places=4)
        self.assertAlmostEqual(result['recall@3'], expected_mean_recall_at_3, places=4)

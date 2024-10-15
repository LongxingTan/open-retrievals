import unittest

from src.retrievals.metrics.map import get_map


class TestGetMap(unittest.TestCase):
    def test_get_map(self):
        qid2positive = {'q1': ['d1', 'd3'], 'q2': ['d1'], 'q3': ['d2', 'd3']}

        qid2ranking = {'q1': ['d1', 'd2', 'd3', 'd4'], 'q2': ['d2', 'd1', 'd3'], 'q3': ['d1', 'd3', 'd2']}

        cutoff_ranks = [3, 5]

        expected_map = {
            'q1': (1 / 1 + 2 / 3) / 2,  # (Precision@1 for d1 + Precision@3 for d3) / 2 positives
            'q2': (1 / 2) / 1,  # Precision@2 for d1
            'q3': (1 / 2 + 2 / 3) / 2,  # (Precision@2 for d3 + Precision@3 for d2) / 2 positives
        }

        expected_mean_map = sum(expected_map.values()) / len(expected_map)

        result = get_map(qid2positive, qid2ranking, cutoff_ranks)

        self.assertAlmostEqual(result[f'map@{cutoff_ranks[0]}'], expected_mean_map, places=4)

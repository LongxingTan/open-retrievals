from unittest import TestCase

from src.retrievals.models.pooling import ClsTokenPooling, MeanPooling


class PoolingTest(TestCase):
    def setUp(self) -> None:
        self.features = [{"label_ids": [0, 1, 2], "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)]

    def test_cls_pooling(self):
        pass

    def test_mean_pooling(self):
        pass

    def test_last_pooling(self):
        pass

    def test_attention_pooling(self):
        pass

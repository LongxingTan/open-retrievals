import unittest

import torch

from src.retrievals.models.pooling import (
    AdaptiveGeM,
    AutoPooling,
    ClsTokenPooling,
    GeM,
    GeMText,
    LastTokenPooling,
    MeanPooling,
    WeightedLayerPooling,
)


class TestPoolingLayers(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_length = 8
        self.hidden_size = 16
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.last_hidden_state = torch.randn(self.batch_size, self.seq_length, self.hidden_size).to(self.device)
        self.attention_mask = torch.ones(self.batch_size, self.seq_length).to(self.device)
        # Set some positions in attention mask to 0 to simulate padding
        self.attention_mask[:, -2:] = 0

    def test_auto_pooling(self):
        pooling_methods = {
            'mean': MeanPooling,
            'cls': ClsTokenPooling,
            'weighted': WeightedLayerPooling,
            'last': LastTokenPooling,
        }

        for method, expected_class in pooling_methods.items():
            pooling = AutoPooling(method)
            if method != 'weighted':  # Skip weighted as it has different input requirements
                output = pooling(self.last_hidden_state, self.attention_mask)
                self.assertEqual(output.shape, (self.batch_size, self.hidden_size))

    def test_mean_pooling(self):
        pooling = MeanPooling()
        output = pooling(self.last_hidden_state, self.attention_mask)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))

        # Test with all-zero attention mask
        zero_mask = torch.zeros_like(self.attention_mask)
        output_zero = pooling(self.last_hidden_state, zero_mask)
        self.assertFalse(torch.isnan(output_zero).any())

    def test_last_token_pooling(self):
        pooling = LastTokenPooling()
        output = pooling(self.last_hidden_state, self.attention_mask)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))

    def test_gem_text(self):
        pooling = GeMText(dim=1)
        output = pooling(self.last_hidden_state, self.attention_mask)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))

        # Test with different p values
        pooling.p.data.fill_(2.0)
        output_p2 = pooling(self.last_hidden_state, self.attention_mask)
        self.assertFalse(torch.equal(output, output_p2))

    def test_gem(self):
        # Create 4D input for GeM
        x = torch.randn(self.batch_size, self.hidden_size, 7, 7).to(self.device)
        pooling = GeM()
        output = pooling(x)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size, 1, 1))

    def test_adaptive_gem(self):
        x = torch.randn(self.batch_size, self.hidden_size, 7, 7).to(self.device)
        pooling = AdaptiveGeM()
        output = pooling(x)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size, 1, 1))

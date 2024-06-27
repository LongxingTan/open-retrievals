import unittest

import torch
import torch.nn as nn

from src.retrievals.losses.token_loss import TokenLoss


class TestTokenLoss(unittest.TestCase):
    def setUp(self):
        self.token_index = 5
        self.train_group_size = 2
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.loss_fn = TokenLoss(
            token_index=self.token_index,
            train_group_size=self.train_group_size,
            criterion=self.criterion,
        )
        self.logits = torch.randn(4, 10, 15)  # Example shapes: [batch_size * train_group_size, seq_len, num_classes]
        self.labels = torch.randint(0, 15, (4, 10))

    def test_loss_calculation(self):
        loss = self.loss_fn(logits=self.logits, labels=self.labels)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss).any())
        self.assertGreaterEqual(loss.item(), 0)

    def test_shifted_targets(self):
        _, max_indices = torch.max(self.labels, dim=1)
        predict_indices = max_indices - 1
        expected_predict_indices = torch.clamp(predict_indices, min=0)  # Ensure no negative indices
        logits = [self.logits[i, expected_predict_indices[i], :] for i in range(self.logits.shape[0])]
        logits = torch.stack(logits, dim=0)
        scores = logits[:, self.token_index]

        self.assertEqual(scores.shape[0], self.logits.shape[0])

    def test_grouped_scores(self):
        scores = self.loss_fn(logits=self.logits, labels=self.labels)
        print(scores)

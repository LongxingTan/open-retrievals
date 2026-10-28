"""CPU numerical guards for the training refactor, not quality/speed benchmarks."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import BertConfig, BertModel

from src.retrievals.losses import ColbertLoss, InfoNCE
from src.retrievals.models import AutoModelForEmbedding
from src.retrievals.trainer.dense import DenseTrainer


class NumericalParityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_dense_loss_and_gradients_match_reference(self):
        for inbatch in (True, False):
            with self.subTest(inbatch=inbatch):
                values = [torch.randn(n, 8, requires_grad=True) for n in (3, 3, 6)]
                q, p, n = [F.normalize(value, dim=-1) for value in values]
                if inbatch:
                    scores = q @ torch.cat([p, n]).T
                    labels = torch.arange(3)
                else:
                    scores = torch.cat(
                        [(q * p).sum(-1, keepdim=True), torch.einsum('bh,bnh->bn', q, n.reshape(3, 2, 8))],
                        dim=-1,
                    )
                    labels = torch.zeros(3, dtype=torch.long)
                expected = F.cross_entropy(scores / 0.05, labels)
                actual = InfoNCE(temperature=0.05, use_inbatch_negative=inbatch)(*values)
                torch.testing.assert_close(actual, expected)
                for got, want in zip(
                    torch.autograd.grad(actual, values, retain_graph=True), torch.autograd.grad(expected, values)
                ):
                    torch.testing.assert_close(got, want)

    def test_unmasked_colbert_matches_previous_objective_and_gradients(self):
        for inbatch in (True, False):
            with self.subTest(inbatch=inbatch):
                q, p, n = [torch.randn(b, 4, 8, requires_grad=True) for b in (3, 3, 6)]
                if inbatch:
                    positive = torch.einsum('bsh,cth->bcst', q, p).max(-1).values.sum(-1)
                    negative = torch.einsum('bsh,cth->bcst', q, n).max(-1).values.sum(-1)
                    labels = torch.arange(3)
                else:
                    positive = torch.einsum('bsh,bnth->bnst', q, p.reshape(3, 1, 4, 8)).max(-1).values.sum(-1)
                    negative = torch.einsum('bsh,bnth->bnst', q, n.reshape(3, 2, 4, 8)).max(-1).values.sum(-1)
                    labels = torch.zeros(3, dtype=torch.long)
                expected = F.cross_entropy(torch.cat([positive, negative], dim=-1) / 0.05, labels)
                actual = ColbertLoss(temperature=0.05, use_inbatch_negative=inbatch)(q, p, n)
                torch.testing.assert_close(actual, expected)
                for got, want in zip(
                    torch.autograd.grad(actual, (q, p, n), retain_graph=True),
                    torch.autograd.grad(expected, (q, p, n)),
                ):
                    torch.testing.assert_close(got, want)

    def test_legacy_embedding_forward_matches_backbone_pooling(self):
        backbone = BertModel(
            BertConfig(
                vocab_size=16,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                hidden_dropout_prob=0,
                attention_probs_dropout_prob=0,
            )
        )
        # No tokenizer or downloads are needed when max_length is explicit.
        features = {
            'input_ids': torch.tensor([[2, 5, 3, 0], [2, 6, 7, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
        }
        for pooling in ('mean', 'cls', 'last'):
            with self.subTest(pooling=pooling):
                model = AutoModelForEmbedding(model=backbone, pooling_method=pooling, max_length=4, device='cpu')
                model.eval()
                hidden = backbone(**features).last_hidden_state
                expected = model.pooling(hidden, attention_mask=features['attention_mask'])
                torch.testing.assert_close(model(features), expected)

    def test_dense_training_does_not_build_unused_scores(self):
        trainer = object.__new__(DenseTrainer)
        trainer.loss_fn = lambda q, p, n: (q - p).square().mean()
        # The loss needs no score matrix; only an explicit output request does.
        inputs = {'query': torch.randn(3, 8), 'positive': torch.randn(3, 8)}
        with patch.object(torch.Tensor, '__matmul__', side_effect=AssertionError('unused scores')):
            loss = trainer.compute_loss(lambda x: x, inputs)
        self.assertTrue(torch.isfinite(loss))
        _, outputs = trainer.compute_loss(lambda x: x, inputs, return_outputs=True)
        torch.testing.assert_close(outputs['scores'], inputs['query'] @ inputs['positive'].T)

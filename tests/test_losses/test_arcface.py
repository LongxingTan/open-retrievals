from unittest import TestCase
from unittest.mock import patch

import torch

from src.retrievals.losses.arcface import ArcFaceAdaptiveMarginLoss

from .test_losses_common import LossTesterMixin


class ArcFaceAdaptiveMarginLossTest(TestCase):
    def setUp(self):
        # Initialize with a simple in_features and out_features configuration
        self.in_features = 10
        self.out_features = 5
        self.arcface_loss = ArcFaceAdaptiveMarginLoss(in_features=self.in_features, out_features=self.out_features)

    def test_init_parameters(self):
        # Ensure parameters are initialized correctly
        self.assertEqual(self.arcface_loss.arc_weight.shape, (self.out_features, self.in_features))
        # Xavier uniform initialization cannot be directly checked for values,
        # but we can check if parameters are registered and of correct type
        self.assertTrue(isinstance(self.arcface_loss.arc_weight, torch.nn.Parameter))

    def test_set_margin(self):
        margin = 0.5
        self.arcface_loss.set_margin(margin=margin)
        # Check if margin related attributes are set correctly
        self.assertEqual(self.arcface_loss.margin, margin)
        self.assertTrue(torch.is_tensor(self.arcface_loss.cos_m))
        self.assertTrue(torch.is_tensor(self.arcface_loss.sin_m))
        self.assertTrue(self.arcface_loss.arc_weight.requires_grad)

    @patch('torch.nn.functional.linear', return_value=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]))
    @patch('torch.nn.functional.normalize', side_effect=lambda x: x)
    def test_forward(self, mock_normalize, mock_linear):
        embeddings = torch.randn(1, self.in_features)
        labels = torch.tensor([1])
        output = self.arcface_loss.forward(embeddings, labels)

        self.assertIn("sentence_embedding", output)
        # self.assertIn("loss", output)  # This assumes that self.criterion is not None

        # Check output shapes and types
        self.assertTrue(isinstance(output["sentence_embedding"], torch.Tensor))

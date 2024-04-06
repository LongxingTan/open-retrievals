from unittest import TestCase

import torch

from src.retrievals.losses.focal_loss import FocalLoss


class FocalLossTest(TestCase):
    def setUp(self):
        self.inputs = torch.randn(10, 5)  # Example: 10 samples, 5 classes
        self.labels = torch.randint(0, 5, (10,))  # Random labels for the 10 samples

    def test_loss_computation(self):
        # Testing default gamma = 0 (should behave like CrossEntropyLoss)
        focal_loss = FocalLoss()
        ce_loss = torch.nn.CrossEntropyLoss()

        fl_loss_val = focal_loss(self.inputs, self.labels)
        ce_loss_val = ce_loss(self.inputs, self.labels)

        # With gamma = 0, FocalLoss should be very close to CrossEntropyLoss
        self.assertTrue(torch.isclose(fl_loss_val, ce_loss_val, atol=1e-7))

    def test_gamma_effect(self):
        # Compare loss values for different gamma values on a difficult example
        focal_loss_low_gamma = FocalLoss(gamma=0.5)
        focal_loss_high_gamma = FocalLoss(gamma=2)

        low_gamma_loss = focal_loss_low_gamma(self.inputs, self.labels)
        high_gamma_loss = focal_loss_high_gamma(self.inputs, self.labels)

        # Generally, we cannot assert the direction of change without knowing the inputs,
        # but we can assert the computation was successful.
        self.assertTrue(torch.isfinite(low_gamma_loss))
        self.assertTrue(torch.isfinite(high_gamma_loss))

    def test_numerical_stability(self):
        # Potentially use very small probabilities to test stability
        small_prob_inputs = torch.log(torch.tensor([[1e-10, 1.0]]))
        labels = torch.tensor([0])

        focal_loss = FocalLoss(gamma=2)
        loss = focal_loss(small_prob_inputs, labels)

        # Simply check if the computation is stable (not NaN or inf)
        self.assertTrue(torch.isfinite(loss))

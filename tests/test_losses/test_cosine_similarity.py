from unittest import TestCase

import torch

from src.retrievals.losses.cosine_similarity import CosineSimilarity


class CosineSimilarityTest(TestCase):
    def setUp(self):
        # Setup can adjust parameters for wide coverage of scenarios
        self.query_embeddings = torch.randn(10, 128)  # Example embeddings
        self.passage_embeddings = torch.randn(10, 128)
        self.temperature = 0.1

    def test_loss_computation(self):
        # Initialize with a temperature value
        module = CosineSimilarity(temperature=self.temperature)

        # Compute loss
        loss = module(self.query_embeddings, self.passage_embeddings)

        # Check if loss is a single scalar value and not nan or inf
        self.assertTrue(torch.isfinite(loss))

    def test_temperature_effect(self):
        # High temperature
        high_temp_module = CosineSimilarity(temperature=100.0)
        high_temp_loss = high_temp_module(self.query_embeddings, self.passage_embeddings)

        # Low temperature
        low_temp_module = CosineSimilarity(temperature=0.01)
        low_temp_loss = low_temp_module(self.query_embeddings, self.passage_embeddings)

        # Expect the loss to be higher for the lower temperature due to sharper softmax
        self.assertTrue(low_temp_loss > high_temp_loss)

    def test_get_temperature(self):
        # Assuming dynamic_temperature or a related feature was meant to be implemented
        module = CosineSimilarity(temperature=self.temperature)
        retrieved_temp = module.get_temperature()

        # Simply check if the temperature retrieval is consistent with initialization
        self.assertEqual(retrieved_temp, self.temperature)

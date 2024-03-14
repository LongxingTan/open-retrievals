"""
trainable temperature parameter: Text and Code Embeddings by Contrastive Pre-Training
"""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class CosineSimilarity(nn.Module):
    def __init__(self, temperature: float = 0.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_embeddings: torch.Tensor, passage_embeddings: torch.Tensor):
        sim_pos_vector = torch.cosine_similarity(query_embeddings, passage_embeddings, dim=-1)
        sim_pos_vector = sim_pos_vector / self.temperature
        sim_neg_matrix = torch.cosine_similarity(
            query_embeddings.unsqueeze(1),
            passage_embeddings.unsqueeze(0),
            dim=-1,
        )
        sim_neg_matrix = sim_neg_matrix / self.temperature
        sim_diff_matrix = sim_pos_vector.unsqueeze(1) - sim_neg_matrix
        loss = -torch.log(torch.sigmoid(sim_diff_matrix)).mean()
        return loss

    def get_temperature(self):
        if not self.dynamic_temperature:
            return self.temperature
        return torch.clamp(self.temperature, min=1e-3)


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

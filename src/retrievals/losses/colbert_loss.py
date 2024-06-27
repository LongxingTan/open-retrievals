from typing import Callable, Optional, Union

import torch
import torch.nn as nn


class ColbertLoss(nn.Module):
    def __init__(
        self,
        criterion: Union[nn.Module, Callable] = nn.CrossEntropyLoss(reduction='mean'),
        temperature: float = 0.02,
        use_inbatch_negative: bool = True,
    ):
        super(ColbertLoss, self).__init__()
        self.criterion = criterion
        self.temperature = temperature
        self.use_inbatch_negative = use_inbatch_negative

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
    ):
        scores = self.similarity(query_embeddings, positive_embeddings)

        if negative_embeddings is not None:
            negative_scores = self.similarity(query_embeddings, negative_embeddings)
            scores = torch.cat([scores, negative_scores], dim=-1)

        if self.temperature is not None:
            scores = scores / self.temperature

        labels = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
        loss = self.criterion(scores, labels)

        return loss

    def similarity(self, query_embeddings, document_embeddings):
        if query_embeddings.size(0) != document_embeddings.size(0) and self.use_inbatch_negative:
            late_interactions = torch.einsum(
                "bsh,cdh->bcsd",
                query_embeddings,
                document_embeddings,
            )
        else:
            document_embeddings = document_embeddings.view(
                query_embeddings.size(0), -1, document_embeddings.size(1), document_embeddings.size(2)
            )

            late_interactions = torch.einsum(
                "bsh,bdth->bdst",
                query_embeddings,
                document_embeddings,
            )
        late_interactions = late_interactions.max(-1).values.sum(-1)
        return late_interactions

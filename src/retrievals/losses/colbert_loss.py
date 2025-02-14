from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from .base import Base


class ColbertLoss(Base):
    def __init__(
        self,
        criterion: Union[nn.Module, Callable] = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean'),
        temperature: float = 0.02,
        use_inbatch_negative: bool = False,
        negatives_cross_device: bool = False,
    ):
        super(ColbertLoss, self).__init__(negatives_cross_device)
        self.criterion = criterion
        self.temperature = temperature
        self.use_inbatch_negative = use_inbatch_negative

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):

        if negative_embeddings is None and self.use_inbatch_negative is False:
            raise ValueError(
                "No negative samples for ColBERT, either provide negative_embeddings or use_inbatch_negative=True"
            )
        if self.negatives_cross_device:
            query_embeddings = self._dist_gather_tensor(query_embeddings)
            positive_embeddings = self._dist_gather_tensor(positive_embeddings)
            negative_embeddings = self._dist_gather_tensor(negative_embeddings)

        scores = self.similarity(query_embeddings, positive_embeddings)

        if negative_embeddings is not None:
            negative_scores = self.similarity(query_embeddings, negative_embeddings, mask=mask)
            scores = torch.cat([scores, negative_scores], dim=-1)

        if self.temperature is not None:
            scores = scores / self.temperature

        if self.use_inbatch_negative:
            labels = torch.arange(scores.size(0), dtype=torch.long, device=scores.device)
        else:
            labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)

        loss = self.criterion(scores, labels)
        return loss

    def similarity(self, query_embeddings, document_embeddings, mask: Optional[torch.Tensor] = None):
        if self.use_inbatch_negative:  # query_embeddings.size(0) != document_embeddings.size(0) and
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

        # if mask is not None:
        #     query_sequence_length = mask[:, 1:].sum(-1, keepdim=False)
        #     if late_interactions.dim() == 2:  # if the train_group_size > 2, the late_interactions shape: batch * neg
        #         query_sequence_length = query_sequence_length.unsqueeze(1)
        #
        #     late_interactions = late_interactions / query_sequence_length
        # else:
        #     late_interactions = late_interactions / query_embeddings.size(1)
        return late_interactions

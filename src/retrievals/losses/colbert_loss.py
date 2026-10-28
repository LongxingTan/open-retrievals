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
        document_mask: Optional[torch.Tensor] = None,
        negative_mask: Optional[torch.Tensor] = None,
    ):

        if negative_embeddings is None and self.use_inbatch_negative is False:
            raise ValueError(
                "No negative samples for ColBERT, either provide negative_embeddings or use_inbatch_negative=True"
            )
        if self.negatives_cross_device:
            query_embeddings = self._dist_gather_tensor(query_embeddings)
            positive_embeddings = self._dist_gather_tensor(positive_embeddings)
            negative_embeddings = self._dist_gather_tensor(negative_embeddings)
            mask = self._dist_gather_tensor(mask)
            document_mask = self._dist_gather_tensor(document_mask)
            negative_mask = self._dist_gather_tensor(negative_mask)

        scores = self.similarity(query_embeddings, positive_embeddings, mask, document_mask)

        if negative_embeddings is not None:
            negative_scores = self.similarity(query_embeddings, negative_embeddings, mask, negative_mask)
            scores = torch.cat([scores, negative_scores], dim=-1)

        if self.temperature is not None:
            scores = scores / self.temperature

        if self.use_inbatch_negative:
            labels = torch.arange(scores.size(0), dtype=torch.long, device=scores.device)
        else:
            labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)

        loss = self.criterion(scores, labels)
        return loss

    def similarity(self, query_embeddings, document_embeddings, mask=None, document_mask=None):
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
        if document_mask is not None:
            if self.use_inbatch_negative:
                document_mask = document_mask[None, :, None, :]
            else:
                document_mask = document_mask.reshape(query_embeddings.size(0), -1, 1, document_mask.size(-1))
            late_interactions = late_interactions.masked_fill(~document_mask.bool(), float("-inf"))
        token_scores = late_interactions.max(-1).values
        token_scores = token_scores.masked_fill(~torch.isfinite(token_scores), 0)
        if mask is not None:
            # Legacy ColBERT callers include CLS; their embeddings omit it.
            if mask.size(-1) == token_scores.size(-1) + 1:
                mask = mask[:, 1:]
            token_scores = token_scores * mask[:, None, :].to(token_scores.dtype)
        return token_scores.sum(-1)

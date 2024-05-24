import torch
from torch import nn


class MoCoLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(MoCoLoss, self).__init__()
        self.temperature = temperature

    def forward(self, q, k):
        batch_size = q.shape[0]
        q = nn.functional.normalize(q, dim=1)  # Normalize query embeddings
        k = nn.functional.normalize(k, dim=1)  # Normalize key embeddings

        # Positive logits: Nx1
        logits_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # Negative logits: NxK
        logits_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([logits_pos, logits_neg], dim=1)

        logits /= self.temperature

        labels = torch.zeros(batch_size, dtype=torch.long).cuda()
        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss

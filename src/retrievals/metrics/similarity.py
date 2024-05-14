import logging

import torch

logger = logging.getLogger(__name__)


def pytorch_cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    return cos_sim(a, b)


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


#
#
# def dot_score(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     """
#     Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
#
#     :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
#     """
#     if not isinstance(a, torch.Tensor):
#         a = torch.tensor(a)
#
#     if not isinstance(b, torch.Tensor):
#         b = torch.tensor(b)
#
#     if len(a.shape) == 1:
#         a = a.unsqueeze(0)
#
#     if len(b.shape) == 1:
#         b = b.unsqueeze(0)
#
#     return torch.mm(a, b.transpose(0, 1))
#
#
# def pairwise_dot_score(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     """
#     Computes the pairwise dot-product dot_prod(a[i], b[i])
#
#     :return: Vector with res[i] = dot_prod(a[i], b[i])
#     """
#     if not isinstance(a, torch.Tensor):
#         a = torch.tensor(a)
#
#     if not isinstance(b, torch.Tensor):
#         b = torch.tensor(b)
#
#     return (a * b).sum(dim=-1)
#
#
# def pairwise_cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     """
#     Computes the pairwise cossim cos_sim(a[i], b[i])
#
#     :return: Vector with res[i] = cos_sim(a[i], b[i])
#     """
#     if not isinstance(a, torch.Tensor):
#         a = torch.tensor(a)
#
#     if not isinstance(b, torch.Tensor):
#         b = torch.tensor(b)
#
#     return pairwise_dot_score(normalize_embeddings(a), normalize_embeddings(b))
#
#
# def pairwise_angle_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     """
#     Computes the absolute normalized angle distance;
#     see AnglELoss or https://arxiv.org/abs/2309.12871v1
#     for more information.
#
#     :return: Vector with res[i] = angle_sim(a[i], b[i])
#     """
#
#     if not isinstance(x, torch.Tensor):
#         x = torch.tensor(x)
#
#     if not isinstance(y, torch.Tensor):
#         y = torch.tensor(y)
#
#     # modified from https://github.com/SeanLee97/AnglE/blob/main/angle_emb/angle.py
#     # chunk both tensors to obtain complex components
#     a, b = torch.chunk(x, 2, dim=1)
#     c, d = torch.chunk(y, 2, dim=1)
#
#     z = torch.sum(c**2 + d**2, dim=1, keepdim=True)
#     re = (a * c + b * d) / z
#     im = (b * c - a * d) / z
#
#     dz = torch.sum(a**2 + b**2, dim=1, keepdim=True) ** 0.5
#     dw = torch.sum(c**2 + d**2, dim=1, keepdim=True) ** 0.5
#     re /= dz / dw
#     im /= dz / dw
#
#     norm_angle = torch.sum(torch.concat((re, im), dim=1), dim=1)
#     return torch.abs(norm_angle)
#
#
# def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
#     """
#     Normalizes the embeddings matrix, so that each sentence embedding has unit length
#     """
#     return torch.nn.functional.normalize(embeddings, p=2, dim=1)

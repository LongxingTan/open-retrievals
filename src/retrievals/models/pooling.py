import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoPooling(nn.Module):
    """Pooling class to transfer input to 1D embedding"""

    def __init__(self, pooling_method: str, **kwargs) -> None:
        super().__init__()
        if pooling_method in ["mean", 'avg', 'average']:
            self.pooling = MeanPooling()
        elif pooling_method in ["cls", 'first']:
            self.pooling = ClsTokenPooling()
        elif pooling_method == "weighted":
            self.pooling = WeightedLayerPooling()
        elif pooling_method in ['eos', "last"]:
            self.pooling = LastTokenPooling()
        else:
            raise ValueError(
                f"pooling_method {pooling_method} is not a valid method: 'cls', 'mean', 'weighted', 'last'"
            )

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.pooling(last_hidden_state, attention_mask)


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        last hidden state: First element of model_output contains all token embeddings
        """
        # last_hidden_state = model_output[0]
        attention_mask = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(last_hidden_state * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class ClsTokenPooling(nn.Module):
    """CLS Pooling"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Take embeddings of first token per sample
        # left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        # if left_padding:
        #     return last_hidden_states[:, -1]

        cls_token = last_hidden_state[:, 0]
        return cls_token


class LastTokenPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            emb = last_hidden_state[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            emb = last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device),
                sequence_lengths,
            ]
        return emb


class AttentionPooling(nn.Module):
    def __init__(self):
        super(AttentionPooling, self).__init__()

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return last_hidden_state[:, 0, :]


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers: int, layer_start: int = 4, layer_weights=None):
        super().__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = (
            layer_weights
            if layer_weights is not None
            else nn.Parameter(torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float))
        )

    def forward(self, all_hidden_states: torch.Tensor) -> torch.Tensor:
        all_layer_embedding = all_hidden_states[self.layer_start :, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average


class GeMText(nn.Module):
    """GeM Pooling for NLP
    Generalized mean: https://arxiv.org/abs/1711.02512
    """

    def __init__(self, dim, p: int = 3, eps: float = 1e-6):
        super(GeMText, self).__init__()
        self.dim = dim
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(x.shape)
        x = (x.clamp(min=self.eps) * attention_mask_expanded).pow(self.p).sum(self.dim)
        ret = x / attention_mask_expanded.sum(self.dim).clip(min=self.eps)
        ret = ret.pow(1 / self.p)
        return ret


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps: float = 1e-6, p_trainable: bool = True):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class AdaptiveGeM(nn.Module):
    def __init__(self, size=(1, 1), p: int = 3, eps: float = 1e-6):
        super().__init__()
        self.size = size
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), self.size).pow(1.0 / self.p)

    def __repr__(self):
        return f"AdaptiveGeM(size={self.size}, p={self.p}, eps={self.eps})"


class TopKPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return


class SumPooling(nn.Module):
    def forward(self, x, x_mask=None):
        if x_mask is None or x_mask.data.sum() == 0:
            return torch.sum(x, 1)
        else:
            x_mask = x_mask.unsqueeze(-1).expand(x.size())
            x.data.masked_fill_(x_mask.data, 0.0)
            return torch.sum(x, 1)


class FMPooling(nn.Module):
    def __init__(self):
        super(FMPooling, self).__init__()
        self.sum_pooling = SumPooling()

    def forward(self, x, x_mask=None):
        summed_emb = self.sum_pooling(x, x_mask)
        summed_emb_square = summed_emb**2

        squared_emb = x**2
        squared_sum_emb = self.sum_pooling(squared_emb, x_mask)

        y_second_order = 0.5 * (summed_emb_square - squared_sum_emb)
        return y_second_order

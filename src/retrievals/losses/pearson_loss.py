import torch
import torch.nn as nn


class PearsonLoss(nn.Module):
    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        xm = x.sub(mean_x)
        ym = y.sub(mean_y)
        r_num = xm.dot(ym)
        r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
        r_den = torch.clamp(r_den, min=1e-6)
        r_val = r_num / r_den
        r_val = torch.clamp(r_val, min=-1.0, max=1.0)
        return 1 - r_val

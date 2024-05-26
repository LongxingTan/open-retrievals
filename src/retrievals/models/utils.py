from typing import Dict, Literal

import torch


def get_device_name() -> Literal["mps", "cuda", "cpu"]:
    """
    Returns the name of the device where this module is running on.
    It's a simple implementation that doesn't cover cases when more powerful GPUs are available and
    not a primary device ('cuda:0') or MPS device is available, but not configured properly:
    https://pytorch.org/docs/master/notes/mps.html

    :return: Device name, like 'cuda' or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def batch_to_device(batch: Dict, target_device: str) -> Dict[str, torch.Tensor]:
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
        else:
            batch[key] = torch.tensor(batch[key], dtype=torch.long).to(target_device)
    return batch

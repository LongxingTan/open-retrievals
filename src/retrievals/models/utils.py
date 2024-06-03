import re
from typing import Dict, List, Literal, Optional

import torch
from torch import nn
from transformers import PreTrainedModel

DEFAULT_LLM_PATTERNS = [r'.*llama.*', r'.*qwen.*', r'.*baichuan.*', r'.*mistral.*', r'.*intern.*']


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


def check_casual_lm(model_name_or_path: str, llm_regex_patterns: List[str] = None) -> bool:
    if llm_regex_patterns is not None:
        llm_regex_patterns += DEFAULT_LLM_PATTERNS
    else:
        llm_regex_patterns = DEFAULT_LLM_PATTERNS
    model_name_or_path = model_name_or_path.lower()
    for pattern in llm_regex_patterns:
        if re.match(pattern, model_name_or_path):
            return True
    return False


def find_all_linear_names(model: PreTrainedModel, linear_type: Optional[object] = None) -> List[str]:
    """
    Find all linear layer names

    :param model: PreTrainedModel
    :param linear_type: Optional[object] = None, linear type, such as nn.Linear and bnb.nn.Linear4bit.

    :return: List[str], linear layer names
    """
    if linear_type is None:
        linear_type = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_type):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def resize_token_embeddings(
    model,
    new_num_tokens: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
) -> nn.Embedding:
    return model.resize_token_embeddings(new_num_tokens=new_num_tokens, pad_to_multiple_of=pad_to_multiple_of)

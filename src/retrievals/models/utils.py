import re
from copy import deepcopy
from typing import Dict, List, Optional

import torch
from torch import nn
from transformers import PreTrainedModel, is_torch_npu_available

DEFAULT_LLM_PATTERNS = [
    r'.*llama.*',
    r'.*mistral.*',
    r'.*qwen.*',
    r'.*baichuan.*',
    r'.*intern.*',
    r'.*Phi.*',
    r'.*gemma.*',
]


def get_device_name():
    """
    Returns the name of the device where this module is running on.
    It's a simple implementation that doesn't cover cases when more powerful GPUs are available and
    not a primary device ('cuda:0') or MPS device is available, but not configured properly:
    https://pytorch.org/docs/master/notes/mps.html

    :return: Device name, like 'cuda' or 'cpu'
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    elif is_torch_npu_available():
        return torch.device('npu')
    else:
        return torch.device("cpu")


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


def check_causal_lm(model_name_or_path: str, llm_regex_patterns: List[str] = None) -> bool:
    """check if it's a decoder-only causal model"""
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
    :param linear_type: Optional[object] = None, linear type, such as nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt

    :return: List[str], linear layer names
    """
    if linear_type is None:
        linear_type = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_type):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def resize_token_embeddings(
    model,
    new_num_tokens: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
) -> nn.Embedding:
    """when you modify the tokenizer, such as adding new tokens or changing the vocabulary size"""
    return model.resize_token_embeddings(new_num_tokens=new_num_tokens, pad_to_multiple_of=pad_to_multiple_of)


def save_swa_weights(model: nn.Module, model_path_list: List[str], save_file: str, device: str):
    """Get the swa weights from a list of model weights"""

    def average_state_dicts(state_dicts: List[dict]) -> dict:
        """Average the state dictionaries."""
        averaged_state = {}
        num_states = len(state_dicts)
        for key in state_dicts[0]:
            averaged_state[key] = sum(state_dict[key] for state_dict in state_dicts) / num_states
        return averaged_state

    state_list = [torch.load(path, map_location=device) for path in model_path_list]
    averaged_state = average_state_dicts(state_list)
    msg = model.load_state_dict(averaged_state, strict=False)
    print(f"State dict load message: {msg}")

    model.half()
    torch.save(model.state_dict(), save_file)
    print(f"Checkpoint saved at: {save_file}")


def freeze_layers(model, n_layers: int = 6):
    """Freeze layers before the last n_layers."""
    trainable_layers = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_layers += 1

    for index, (name, param) in enumerate(iterable=model.named_parameters()):
        if index < (trainable_layers - n_layers):
            param.requires_grad = False

    return model


class DocumentSplitter(object):
    """
    Rerank the long document
    - https://github.com/netease-youdao/BCEmbedding/blob/master/BCEmbedding/models/utils.py
    """

    def __init__(self, chunk_size: int, chunk_overlap: int = 0, max_chunks_per_doc: int = 32):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks_per_doc = max_chunks_per_doc

    def create_documents(self, query, documents, tokenizer):
        res_merge_inputs = []
        res_merge_inputs_pids = []

        query_inputs = tokenizer.encode_plus(query, truncation=False, padding=False)
        sep_id = tokenizer.sep_token_id
        doc_max_length = self.chunk_size - len(query_inputs['input_ids']) - 2

        for pid, document in enumerate(documents):
            document_inputs = tokenizer.encode_plus(document, truncation=False, padding=False, add_special_tokens=False)
            doc_inputs_length = len(document_inputs['input_ids'])

            if doc_inputs_length <= doc_max_length:
                qc_merge_inputs = self._merge_inputs(query_inputs, document_inputs, sep_id)
                res_merge_inputs.append(qc_merge_inputs)
                res_merge_inputs_pids.append(pid)
            else:
                start_id = 0
                while start_id < doc_inputs_length:
                    end_id = start_id + doc_max_length
                    sub_document_inputs = {k: v[start_id:end_id] for k, v in document_inputs.items()}
                    start_id = end_id - self.chunk_overlap if end_id < doc_inputs_length else end_id

                    qp_merge_inputs = self._merge_inputs(query_inputs, sub_document_inputs, sep_id)
                    res_merge_inputs.append(qp_merge_inputs)
                    res_merge_inputs_pids.append(pid)
        return res_merge_inputs, res_merge_inputs_pids

    def _merge_inputs(self, chunk1_raw, chunk2, sep_id: int):
        chunk1 = deepcopy(chunk1_raw)

        chunk1['input_ids'].append(sep_id)
        chunk1['input_ids'].extend(chunk2['input_ids'])
        chunk1['input_ids'].append(sep_id)

        chunk1['attention_mask'].append(chunk2['attention_mask'][0])
        chunk1['attention_mask'].extend(chunk2['attention_mask'])
        chunk1['attention_mask'].append(chunk2['attention_mask'][0])

        if 'token_type_ids' in chunk1:
            token_type_ids = [1 for _ in range(len(chunk2['token_type_ids']) + 2)]
            chunk1['token_type_ids'].extend(token_type_ids)
        return chunk1

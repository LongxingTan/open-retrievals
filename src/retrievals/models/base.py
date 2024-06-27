import logging
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class Base(ABC, torch.nn.Module):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(model, str):
            assert ValueError("Please use AutoModelForEmbedding.from_pretrained(model_name_or_path)")
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Pytorch forward method."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, *args, **kwargs):
        """Encode documents."""
        pass

    def _encode_from_loader(
        self,
        loader: DataLoader,
        convert_to_numpy: bool = True,
        device: str = None,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = None,
        **kwargs,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        """Encode for sentence embedding"""
        device = device or self.device
        self.model.eval()
        self.model.to(device)

        all_embeddings = []

        for idx, inputs in enumerate(tqdm(loader, desc="Encoding", disable=not show_progress_bar)):
            with torch.autocast(device_type=device) if self.use_fp16 else nullcontext():
                with torch.no_grad():
                    inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
                    embeddings = self.forward_from_loader(inputs_on_device)
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    all_embeddings.append(embeddings)
        if convert_to_numpy:
            all_embeddings = np.concatenate([emb.cpu().numpy() for emb in all_embeddings], axis=0)
        else:
            all_embeddings = torch.concat(all_embeddings)
        return all_embeddings

    def preprocess(self, batch_sentence_pair, query_max_length, document_max_length):
        query_list = [item[0] for item in batch_sentence_pair]
        document_list = [item[1] for item in batch_sentence_pair]

        query_batch_tokens = self.tokenizer(
            query_list, padding='max_length', truncation=True, max_length=query_max_length, return_tensors='pt'
        )
        query_batch_tokens_on_device = {k: v.to(self.device) for k, v in query_batch_tokens.items()}
        document_batch_tokens = self.tokenizer(
            document_list, padding='max_length', truncation=True, max_length=document_max_length, return_tensors='pt'
        )
        document_batch_tokens_on_device = {k: v.to(self.device) for k, v in document_batch_tokens.items()}

        return {
            "query_input_ids": query_batch_tokens_on_device['input_ids'],
            "query_attention_mask": query_batch_tokens_on_device['attention_mask'],
            "doc_input_ids": document_batch_tokens_on_device['input_ids'],
            "doc_attention_mask": document_batch_tokens_on_device['attention_mask'],
        }

    def save_pretrained(self, path: str, safe_serialization: bool = True):
        """
        Saves all model and tokenizer to path
        """
        logger.info("Save model to {}".format(path))
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(path, state_dict=state_dict, safe_serialization=safe_serialization)
        self.tokenizer.save_pretrained(path)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def push_to_hub(self, hub_model_id: str, private: bool = True, **kwargs):
        """push model to hub

        :param hub_model_id: str, hub model id.
        :param private: bool, whether push to private repo. Default True.
        :param kwargs: other kwargs for `push_to_hub` method.
        """
        self.tokenizer.push_to_hub(hub_model_id, private=private, **kwargs)
        self.backbone.push_to_hub(hub_model_id, private=private, **kwargs)

    def _dist_gather_tensor(self, tensor: Optional[torch.Tensor]):
        if tensor is None:
            return None
        tensor = tensor.contiguous()

        all_tensors = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, tensor)

        all_tensors[self.process_rank] = tensor
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

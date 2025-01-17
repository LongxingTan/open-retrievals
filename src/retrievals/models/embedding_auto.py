"""Text embedding model"""

import copy
import logging
import os
import time
from collections.abc import Iterable
from contextlib import nullcontext
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizer,
)

from .base import Base
from .pooling import AutoPooling
from .utils import batch_to_device, check_causal_lm, get_device_name

logger = logging.getLogger(__name__)


class AutoModelForEmbedding(Base):
    """
    Loads or creates an Embedding model that can be used to map sentences / text.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        pooling_method: str = 'cls',
        max_length: Optional[int] = None,
        loss_fn: Optional[Callable] = None,
        query_instruction: Optional[str] = None,
        document_instruction: Optional[str] = None,
        use_fp16: bool = False,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Loads or creates an Embedding model that can be used to map sentences / text.
        """
        super().__init__(model, tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        self.pooling_method = pooling_method
        self.pooling = AutoPooling(pooling_method) if pooling_method else None
        self.loss_fn = loss_fn
        self.max_length = max_length or self._determine_max_length()
        self.query_instruction = query_instruction or '{}'
        self.document_instruction = document_instruction or '{}'
        self.use_fp16 = use_fp16
        self.device = device or get_device_name()
        try:
            self.model.to(self.device)
        except ValueError:
            # `4-bit` or `8-bit` bitsandbytes models have already been set to the correct devices
            pass

    def forward(
        self,
        inputs,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ):
        if isinstance(inputs, (dict, BatchEncoding)):
            embeddings = self.forward_from_tensor(inputs['input_ids'], inputs['attention_mask'])
        elif isinstance(inputs, str) or (isinstance(inputs, Iterable) and isinstance(inputs[0], str)):
            embeddings = self.forward_from_text(inputs)
        else:
            raise ValueError("Invalid input type.")

        if labels is None or self.loss_fn is None:
            return {"sentence_embedding": embeddings} if return_dict else embeddings
        else:
            loss_output = self.loss_fn(embeddings, labels)
            return {"loss": loss_output["loss"], "sentence_embedding": loss_output["sentence_embedding"]}

    def forward_from_tensor(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, without_pooling: bool = False):
        model_output = self.model(input_ids, attention_mask=attention_mask, return_dict=True)
        if self.pooling is not None and not without_pooling:
            last_hidden_state = model_output.get('last_hidden_state', model_output[0])
            embeddings = self.pooling(last_hidden_state, attention_mask=attention_mask)
            return embeddings
        return model_output

    def forward_from_text(self, texts):
        batch_dict = self.tokenizer(
            texts,
            max_length=self.max_length,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        batch_dict["input_ids"] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in batch_dict["input_ids"]]
        batch_dict = self.tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors="pt")
        batch_dict.pop("token_type_ids")
        return self.forward_from_tensor(**batch_dict)

    def encode(
        self,
        inputs: Union[DataLoader, Dict, List, str, np.ndarray],
        is_query: bool = False,
        batch_size: int = 16,
        show_progress_bar: bool = None,
        output_value: Literal["sentence_embedding", "token_embeddings", None] = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
    ):
        if isinstance(inputs, (DataLoader, BatchEncoding, Dict)):
            return self._encode_from_loader(
                loader=inputs,
                show_progress_bar=show_progress_bar,
                output_value=output_value,
                convert_to_numpy=convert_to_numpy,
                convert_to_tensor=convert_to_tensor,
                device=device,
                normalize_embeddings=normalize_embeddings,
            )
        elif isinstance(inputs, str) or isinstance(inputs[0], str):
            return self._encode_from_text(
                sentences=inputs,
                is_query=is_query,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                output_value=output_value,
                convert_to_numpy=convert_to_numpy,
                convert_to_tensor=convert_to_tensor,
                device=device,
                normalize_embeddings=normalize_embeddings,
            )
        else:
            raise ValueError(f'Invalid input type: {type(inputs)}')

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
        self.model.eval()
        if device and device != self.device:
            # for llm, avoid frequently to device
            self.model.to(device)
        device = device or self.device

        all_embeddings = []

        for idx, inputs in enumerate(tqdm(loader, desc="Encoding", disable=not show_progress_bar)):
            with torch.autocast(device_type=device) if self.use_fp16 else nullcontext():
                with torch.no_grad():
                    inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
                    embeddings = self.forward_from_tensor(
                        inputs_on_device['input_ids'], attention_mask=inputs_on_device['attention_mask']
                    )
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    # to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()
                    all_embeddings.append(embeddings)
        if convert_to_numpy:
            all_embeddings = np.concatenate([emb.numpy() for emb in all_embeddings], axis=0)
        else:
            all_embeddings = torch.concat(all_embeddings, dim=0)
        return all_embeddings

    def _encode_from_text(
        self,
        sentences: Union[str, List[str], Tuple[str], np.ndarray],
        is_query: bool = False,
        batch_size: int = 16,
        show_progress_bar: bool = None,
        output_value: Literal["sentence_embedding", "token_embeddings", None] = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        """
        Computes sentence embeddings from sentence-transformers library.

        :param sentences: the sentences to embed.
        :param is_query: if the text is query or document
        :param batch_size: the batch size used for the computation.
        :param show_progress_bar: Whether to output a progress bar when encode sentences.
        :param output_value: The type of embeddings to return: "sentence_embedding" to get sentence embeddings,
            "token_embeddings" to get wordpiece token embeddings, and `None`, to get all output values. Defaults
            to "sentence_embedding".
        :param convert_to_numpy: Whether the output should be a list of numpy vectors. If False, it is a list of tensors
        :param convert_to_tensor: Whether the output should be one large tensor. Overwrites `convert_to_numpy`.
        :param device: Which `torch.device` to use for the computation.
        :param normalize_embeddings: Whether to normalize returned vectors to have length 1. In that case,
            the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return: By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned.
            If convert_to_numpy, a numpy matrix is returned.
        """

        self.model.eval()
        if device and device != self.device:
            self.model.to(device)
        device = device or self.device

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sentence) for sentence in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        if is_query and self.query_instruction:
            logger.info("Encoding query")
            sentences_sorted = [self.query_instruction.format(sentence) for sentence in sentences_sorted]
        if not is_query and self.document_instruction:
            logger.info('Encoding document')
            sentences_sorted = [self.document_instruction.format(sentence) for sentence in sentences_sorted]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.forward(features, return_dict=True)

                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features["attention_mask"]):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id = last_mask_id - 1

                        embeddings.append(token_emb[0 : last_mask_id + 1])
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features["sentence_embedding"])):
                        row = {name: out_features[name][sent_idx] for name in out_features}
                        embeddings.append(row)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            if len(all_embeddings):
                all_embeddings = torch.stack(all_embeddings)
            else:
                all_embeddings = torch.Tensor()
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        return self._encode_from_text(queries, is_query=True, **kwargs)

    def build_index(
        self,
        inputs: Union[DataLoader, Dict, List, str],
        index_path: Optional[str] = None,
        max_document_length: int = 512,
        split_documents: bool = False,
        batch_size: int = 16,
        show_progress_bar: bool = None,
        use_gpu: bool = False,
        save_id: bool = False,
    ):
        import faiss

        logger.info("Start to build index")
        start_time = time.time()

        embeddings = self.encode(
            inputs, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=show_progress_bar
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        index = faiss.IndexFlatL2(len(embeddings[0]))
        if save_id:
            # id_index = faiss.IndexFlatL2(len(embeddings[0]))
            index = faiss.IndexIDMap2(index)
            ids = inputs['ids']

        if use_gpu and self.device == 'cuda':
            logger.info('Build index use faiss-gpu')
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            index = faiss.index_cpu_to_all_gpus(index, co=co)

        if save_id:
            index.add_with_ids(embeddings, ids)
        else:
            index.add(embeddings)

        if index_path:
            logger.info(f'Save faiss index to: {index_path}')
            if os.path.isdir(index_path):
                index_path = index_path + 'faiss.index'
            if not os.path.exists(os.path.dirname(index_path)):
                os.makedirs(os.path.dirname(index_path))

            if use_gpu and self.device == 'cuda':
                index = faiss.index_gpu_to_cpu(index)
            faiss.write_index(index, index_path)

        logger.info(f'Build index successfully, saved in {index_path}, elapsed: {time.time() - start_time:.2}s')
        return index

    @classmethod
    def as_retriever(cls, retrieval_args, **kwargs):
        from .retrieval_auto import AutoModelForRetrieval

        embedding_model = cls(**kwargs)
        return AutoModelForRetrieval(embedding_model, **retrieval_args)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Optional[str] = None,
        pooling_method: Optional[str] = "cls",
        pretrained: bool = True,
        config_path: Optional[str] = None,
        trust_remote_code: bool = True,
        use_fp16: bool = False,
        use_lora: bool = False,
        use_qlora: bool = False,
        lora_path: Optional[str] = None,
        lora_config: Optional["LoraConfig"] = None,  # noqa: F821
        quantization_config=None,
        device: Optional[str] = None,
        query_instruction: Optional[str] = None,
        document_instruction: Optional[str] = None,
        mrl_dim: int = -1,
        pretrained_linear_name: str = 'linear.pt',
        max_length: Optional[int] = None,
        **kwargs,
    ):
        config = AutoConfig.from_pretrained(
            config_path or model_name_or_path, output_hidden_states=True, trust_remote_code=trust_remote_code
        )

        if check_causal_lm(model_name_or_path) and pooling_method != 'last':
            logger.warning('You are using a LLM model, while pooling_method is not last, is that right?')

        if pretrained:
            model = AutoModel.from_pretrained(
                model_name_or_path,
                config=config,
                trust_remote_code=trust_remote_code,
                quantization_config=quantization_config,
                **kwargs,
            )
        else:
            model = AutoModel.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

        if device is None:
            device = get_device_name()

        if use_fp16 and device != 'cpu' and quantization_config is None and not hasattr(config, 'quantization_config'):
            logger.info('Set model to fp16 in inference, if you want fp16 during training, training_args fp16=True')
            model.half()

        if mrl_dim > 0:
            vector_linear = torch.nn.Linear(in_features=model.config.hidden_size, out_features=mrl_dim)
            vector_linear_dict = {
                k.replace("linear.", ""): v
                for k, v in torch.load(os.path.join(model_name_or_path, pretrained_linear_name)).items()
            }
            vector_linear.load_state_dict(vector_linear_dict)

        if (use_lora or use_qlora) and lora_path is None:
            logger.info('Set fine-tuning to LoRA')
            model = cls.setup_lora(model, lora_config, use_qlora)

        if lora_path is not None:
            model = cls.load_lora_weights(model, lora_path)

        return cls(
            model=model,
            tokenizer=tokenizer,
            pooling_method=pooling_method,
            device=device,
            query_instruction=query_instruction,
            document_instruction=document_instruction,
            use_fp16=use_fp16,
            max_length=max_length,
            **kwargs,
        )


class PairwiseModel(nn.Module):
    """Pairwise Model wrapper
    - bi_encoder
        - shared_weights or not
    - poly_encoder

    support: query + pos pair, or query + pos + neg triplet
    """

    def __init__(
        self,
        model: AutoModelForEmbedding,
        loss_fn: Optional[Callable] = None,
        shared_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = model
        self.model.loss_fn = loss_fn
        self.tokenizer = model.tokenizer  # for model save
        self.loss_fn = loss_fn
        self.shared_weights = shared_weights
        if not shared_weights:
            self.document_model = copy.deepcopy(self.model)

    def forward(
        self,
        inputs: Union[Dict[str, torch.Tensor], list],
        inputs_pair: Optional[Dict[str, torch.Tensor]] = None,
        return_dict: bool = True,
        **kwargs,
    ):

        if isinstance(inputs, (list, tuple, dict)) and 2 <= len(inputs) <= 3 or inputs_pair is not None:
            if inputs_pair:
                input1 = inputs
                input2 = inputs_pair
            elif isinstance(inputs, dict):
                inputs = list(inputs.values())
            if isinstance(inputs, (list, tuple)):
                input1 = inputs[0]
                input2 = inputs[1]
                if len(inputs) == 3:
                    input3 = inputs[2]

            ids1, mask1 = input1['input_ids'], input1['attention_mask']
            ids2, mask2 = input2['input_ids'], input2['attention_mask']

            if self.shared_weights:  # bi-encoder, pooling in each
                pooled_output1 = self.model.forward_from_tensor(ids1, attention_mask=mask1)
                pooled_output2 = self.model.forward_from_tensor(ids2, attention_mask=mask2)
                if len(inputs) == 3:
                    pooled_output3 = self.model.forward_from_tensor(
                        input3['input_ids'], attention_mask=input3['attention_mask']
                    )
                    if self.loss_fn is None:
                        return pooled_output1, pooled_output2, pooled_output3

                    outputs = dict()
                    loss = self.loss_fn(pooled_output1, pooled_output2, pooled_output3)
                    outputs["loss"] = loss
                    return outputs

            else:
                pooled_output1 = self.model.forward_from_tensor(ids1, attention_mask=mask1)
                pooled_output2 = self.document_model(ids2, mask2)

            if self.loss_fn is None:
                return pooled_output1, pooled_output2
            else:
                outputs = dict()
                loss = self.loss_fn(pooled_output1, pooled_output2)
                outputs["loss"] = loss
                return outputs

        else:
            # if the example data pair/triplet is already concat into one group. The Sentence-transformer style
            ids = inputs['input_ids']
            mask = inputs['attention_mask']
            transformer_out = self.model.forward_from_tensor(input_ids=ids, attention_mask=mask, without_pooling=True)
            pooled_output = self.pooling(transformer_out[0], mask)
            # pooled_output1 = pooled_output[: len(ids1), :]
            # pooled_output2 = pooled_output[len(ids1):, :]
            return pooled_output


class ListwiseModel(AutoModelForEmbedding):
    """
    Listwise model by segment_id
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        loss_fn: Optional[Callable] = None,
        segment_pooling: str = 'mean',
        num_segments: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(model=model, tokenizer=tokenizer, loss_fn=loss_fn, **kwargs)
        self.segment_pooling = self._get_segment_pooling(segment_pooling)
        self.num_segments = num_segments

    def forward(
        self,
        inputs,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True)

        pooled = self.pooling(outputs.last_hidden_state, inputs['attention_mask'])
        segment_pooled = self.segment_pooling(pooled, inputs['segment_ids'], self.num_segments)

        projected = self.projection(segment_pooled)
        loss = self.loss_fn(projected) if self.loss_fn else None
        return loss

    def _get_segment_pooling(self, method: str) -> Callable:
        if method == 'mean':
            return self._unsorted_segment_mean
        elif method == 'sorted_mean':
            return self._sorted_segment_mean
        raise ValueError(f"Unknown segment pooling method: {method}")

    def _unsorted_segment_mean(self, data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
        """Compute mean embeddings for unsorted segments"""
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(-1))
        result = data.new_zeros((num_segments, data.size(-1)))
        count = data.new_zeros((num_segments, data.size(-1)))

        result.scatter_add_(0, segment_ids, data)
        count.scatter_add_(0, segment_ids, torch.ones_like(data))

        return result / count.clamp(min=1)

    def _sorted_segment_mean(self, data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
        """
        Compute the mean of each segment in data based on sorted segment_ids.

        Args:
            data (torch.Tensor): Input data tensor of shape (batch_size, num_embedding).
            segment_ids (torch.Tensor): Sorted segment IDs tensor of shape (batch_size,).
            num_segments (int): Number of unique segments.

        Returns:
            torch.Tensor: Tensor of shape (num_segments, num_embedding) containing the mean of each segment.
        """
        result = torch.zeros((num_segments, data.size(1)), dtype=data.dtype, device=data.device)
        count = torch.zeros((num_segments,), dtype=torch.int32, device=data.device)

        start_idx = 0
        for i in range(num_segments):
            """Find the range of indices corresponding to the current segment"""
            while start_idx < segment_ids.size(0) and segment_ids[start_idx] == i:
                start_idx += 1

            if start_idx > 0 and segment_ids[start_idx - 1] == i:
                segment_slice = slice(start_idx - (start_idx - segment_ids[start_idx:].tolist().count(i)), start_idx)
                result[i] = data[segment_slice].sum(dim=0)
                count[i] = segment_slice.stop - segment_slice.start

        result /= count.clamp(min=1).unsqueeze(-1)
        return result

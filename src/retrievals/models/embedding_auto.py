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
from .utils import (
    batch_to_device,
    check_causal_lm,
    find_all_linear_names,
    get_device_name,
)

logger = logging.getLogger(__name__)


class AutoModelForEmbedding(Base):
    """
    Loads or creates an Embedding model that can be used to map sentences / text.

    :param model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path,
        it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model
        from the Hugging Face Hub with that name.
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
        super().__init__()
        if isinstance(model, str):
            assert ValueError("Please use AutoModelForEmbedding.from_pretrained(model_name_or_path)")
        self.model = model
        self.tokenizer = tokenizer
        self.pooling_method = pooling_method
        self.pooling = AutoPooling(pooling_method) if pooling_method else None
        self.loss_fn = loss_fn

        if max_length is None:
            if (
                hasattr(self.model, "config")
                and hasattr(self.model.config, "max_position_embeddings")
                and hasattr(self.tokenizer, "model_max_length")
            ):
                max_length = min(self.model.config.max_position_embeddings, self.tokenizer.model_max_length)
        else:
            logger.info('max_length will only work if the encode or forward function input text directly')

        self.max_length = max_length

        self.query_instruction = query_instruction if query_instruction else ''
        self.document_instruction = document_instruction if document_instruction else ''
        self.use_fp16 = use_fp16
        self.device = device or get_device_name()
        try:
            self.model.to(self.device)
        except ValueError:
            # `4-bit` or `8-bit` bitsandbytes models have already been set to the correct devices
            pass

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

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
            raise ValueError

        if labels is None or self.loss_fn is None:
            if return_dict:
                return {"sentence_embedding": embeddings}
            return embeddings
        else:
            outputs = dict()
            loss_output = self.loss_fn(embeddings, labels)
            outputs["loss"] = loss_output["loss"]
            outputs["sentence_embedding"] = loss_output["sentence_embedding"]
            return outputs

    def forward_from_tensor(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, without_pooling: bool = False):
        model_output = self.model(input_ids, attention_mask=attention_mask, return_dict=True)
        if self.pooling is not None and not without_pooling:
            if 'last_hidden_state' in model_output:
                last_hidden_state = model_output['last_hidden_state']
            elif 'hidden_states' not in model_output:
                last_hidden_state = model_output[0]
            else:
                hidden_states = model_output['hidden_states']
                last_hidden_state = hidden_states[-1]
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
            raise ValueError(f'Input type: {type(inputs)}')

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
            sentences_sorted = [self.query_instruction + sentence for sentence in sentences_sorted]
        if not is_query and self.document_instruction:
            logger.info('Encoding document')
            sentences_sorted = [self.document_instruction + sentence for sentence in sentences_sorted]

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

    def set_train_type(self, train_type: Literal['pointwise', 'pairwise', 'listwise'], **kwargs):
        train_type = train_type.lower().replace('-', '')
        logger.info(f'Set train type to {train_type}')
        model_class = {'pointwise': self, 'pairwise': PairwiseModel, 'listwise': ListwiseModel}
        model_class = model_class.get(train_type)

        return model_class(
            model=self.model,
            tokenizer=self.tokenizer,
            pooling_method=self.pooling_method,
            query_instruction=self.query_instruction,
            document_instruction=self.document_instruction,
            device=self.device,
            **kwargs,
        )

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
        custom_config_dict: Optional[Dict] = None,
        use_fp16: bool = False,
        use_lora: bool = False,
        use_qlora: bool = False,
        lora_path: Optional[str] = None,
        lora_config=None,
        quantization_config=None,
        device: Optional[str] = None,
        query_instruction: Optional[str] = None,
        document_instruction: Optional[str] = None,
        mrl_dim: int = -1,
        pretrained_linear_name: str = 'linear.pt',
        max_length: Optional[int] = None,
        **kwargs,
    ):
        if not model_name_or_path or not isinstance(model_name_or_path, str):
            assert ValueError(f'Please input valid model_name_or_path, instead of {model_name_or_path}')

        if config_path:
            config = AutoConfig.from_pretrained(
                config_path, output_hidden_states=True, trust_remote_code=trust_remote_code
            )
        else:
            config = AutoConfig.from_pretrained(
                model_name_or_path, output_hidden_states=True, trust_remote_code=trust_remote_code
            )

        if custom_config_dict:
            if not config:
                config = AutoConfig.from_pretrained(
                    model_name_or_path, output_hidden_states=True, trust_remote_code=trust_remote_code
                )
            config.update(custom_config_dict)

        # if quantization_config is None and hasattr(config, 'quantization_config'):
        #     quantization_config = config.quantization_config

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
            from peft import (
                LoraConfig,
                TaskType,
                get_peft_model,
                prepare_model_for_kbit_training,
            )

            if lora_config is None:
                lora_r = 64
                lora_alpha = 128
                lora_dropout = 0.05
                target_modules = find_all_linear_names(model)
                logger.info(f'Set Lora target module to {target_modules}, r to {lora_r}, lora_alpha to {lora_alpha}')
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                    bias='none',
                    task_type='FEATURE_EXTRACTION',
                )
            if use_qlora:
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        if lora_path is not None:
            logger.info(f'Load pretrained with LoRA adapter {lora_path}')
            from peft import LoraConfig, PeftModel

            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()

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


class PairwiseModel(AutoModelForEmbedding):
    """Pairwise Model wrapper
    - bi_encoder
        - shared_weights or not
    - poly_encoder

    support: query + pos pair, or query + pos + neg triplet
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        pooling_method: str = 'cls',
        loss_fn: Optional[Callable] = None,
        shared_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            pooling_method=pooling_method,
            loss_fn=loss_fn,
            **kwargs,
        )

        self.shared_weights = shared_weights
        if not shared_weights:
            self.document_model = copy.deepcopy(self.model)

    def forward(
        self,
        inputs: Union[Dict[str, torch.Tensor], list],
        inputs_pair: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
                pooled_output1 = super().forward_from_tensor(ids1, attention_mask=mask1)
                pooled_output2 = super().forward_from_tensor(ids2, attention_mask=mask2)
                if len(inputs) == 3:
                    pooled_output3 = super().forward_from_tensor(
                        input3['input_ids'], attention_mask=input3['attention_mask']
                    )
                    if self.loss_fn is None:
                        return pooled_output1, pooled_output2, pooled_output3

                    outputs = dict()
                    loss = self.loss_fn(pooled_output1, pooled_output2, pooled_output3)
                    outputs["loss"] = loss
                    return outputs

            else:
                pooled_output1 = super().forward_from_tensor(ids1, attention_mask=mask1)
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
            transformer_out = super().forward_from_tensor(input_ids=ids, attention_mask=mask, without_pooling=True)
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
        pooling_method: str = 'cls',
        loss_fn: Optional[Callable] = None,
        listwise_pooling: bool = False,
        num_segments: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            pooling_method=pooling_method,
            loss_fn=loss_fn,
            **kwargs,
        )
        self.pooling_method = pooling_method
        self.listwise_pooling = listwise_pooling
        self.num_segments = num_segments

    def forward(
        self,
        inputs,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        encoding = super().forward(inputs)

        res = dict()
        if self.pooling_method == 'unsorted_segment_mean':
            encoding = self._unsorted_segment_mean(
                encoding, segment_ids=inputs['segment_ids'], num_segments=self.num_segments
            )
            res['pred'] = self.fc(encoding[:, 1:])
        else:
            encodings = []
            for i in range(self.num_segments):
                mask_ = (inputs['segment_ids'] == i + 1).int()
                encoding_ = self.pooling(encoding, mask_)
                encoding_ = self.fc(encoding_)
                encodings.append(encoding_)
            res['pred'] = torch.stack(encodings, 1)

        res['pred'] = res['pred'].squeeze(-1)
        return res

    def _unsorted_segment_mean(self, data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
        result_shape = (num_segments, data.size(1))
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))  # (batch, num_embedding)
        result = data.new_full(result_shape, 0)  # init empty result tensor
        count = data.new_full(result_shape, 0)
        result.scatter_add_(0, segment_ids, data)  # fill the result from data to organized segment result
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

import logging
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import faiss
import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import trange
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    GenerationConfig,
)

from src.retrievals.models.pooling import AutoPooling

logger = logging.getLogger(__name__)


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
            print(batch[key])
            batch[key] = torch.tensor(batch[key], dtype=torch.long).to(target_device)
    return batch


class AutoModelForEmbedding(nn.Module):
    """
    Loads or creates a Embedding model that can be used to map sentences / text.

    :param model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path,
        it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model
        from the Hugging Face Hub with that name.
    """

    encode_kwargs: Dict[str, Any] = dict()
    show_progress: bool = False

    def __init__(
        self,
        model_name_or_path: str,
        pretrained: bool = True,
        config_path: Optional[str] = None,
        pooling_method: str = "cls",
        normalize_embeddings: bool = False,
        max_length: Optional[int] = None,
        loss_fn: Optional[Callable] = None,
        query_instruction: Optional[str] = None,
        passage_instruction: Optional[str] = None,
        generation_args: Dict = None,
        use_fp16: bool = False,
        use_lora: bool = False,
        lora_config=None,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, return_tensors=False, trust_remote_code=trust_remote_code
        )

        if pretrained:
            self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        else:
            if config_path is None:
                self.config = AutoConfig.from_pretrained(
                    model_name_or_path, output_hidden_states=True, trust_remote_code=trust_remote_code
                )
                self.model = AutoModel.from_config(self.config)
        self.loss_fn = loss_fn

        if max_length is None:
            if (
                hasattr(self.model, "config")
                and hasattr(self.model.config, "max_position_embeddings")
                and hasattr(self.tokenizer, "model_max_length")
            ):
                max_length = min(self.model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_length = max_length

        if use_fp16:
            self.model.half()
        if use_lora:
            # peft config and wrapping
            from peft import LoraConfig, TaskType, get_peft_model

            if not lora_config:
                raise ValueError("If use_lora is true, please provide a valid lora_config")
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        if device is None:
            self.device = get_device_name()
        else:
            self.device = device

        self.normalize_embeddings = normalize_embeddings
        self.pooling = AutoPooling(pooling_method)

        # self.fc = nn.Linear(768, 1)
        # self._init_weights(self.fc)

        self.query_instruction = query_instruction
        self.passage_instruction = passage_instruction
        if generation_args is not None:
            generation_config = self.model.generation_config.to_dict()
            generation_config.update(generation_args)
            generation_config.update({"pad_token_id": self.tokenizer.pad_token_id})
            self.model.generation_config = GenerationConfig(**generation_config)

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
            embeddings = self.forward_from_loader(inputs)
        elif isinstance(inputs, str) or (isinstance(inputs, list) and isinstance(inputs[0], str)):
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

    def forward_from_loader(self, inputs):
        model_output = self.model(inputs['input_ids'], inputs['attention_mask'])
        if self.pooling is not None:
            embeddings = self.pooling(model_output[0], inputs["attention_mask"])

            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings

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
        return self.forward_from_loader(batch_dict)

    def encode(
        self,
        inputs,
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
    ):
        if isinstance(inputs, (BatchEncoding, Dict)):
            return self.encode_from_loader(
                loader=inputs,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                output_value=output_value,
                convert_to_numpy=convert_to_numpy,
                convert_to_tensor=convert_to_tensor,
                device=device,
                normalize_embeddings=normalize_embeddings,
            )
        elif isinstance(inputs, (str, List, Tuple)):
            return self.encode_from_text(
                sentences=inputs,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                output_value=output_value,
                convert_to_numpy=convert_to_numpy,
                convert_to_tensor=convert_to_tensor,
                device=device,
                normalize_embeddings=normalize_embeddings,
            )
        else:
            raise ValueError

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model."""
        embeddings = self.encode(texts, show_progress_bar=self.show_progress, **self.encode_kwargs)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model."""
        return self.embed_documents([text])[0]

    def encode_from_loader(
        self,
        loader,
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
    ) -> Union[List[torch.Tensor], ndarray, torch.Tensor]:
        self.model.eval()
        self.model.to(device or self.device)
        all_embeddings = []

        with torch.no_grad():
            for idx, inputs in enumerate(loader):
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                embed = self.forward_from_loader(inputs)
                embed = embed.detach().cpu()
                if convert_to_numpy:
                    embed = embed.numpy()
                all_embeddings.append(embed.numpy())
        if convert_to_numpy:
            all_embeddings = np.concatenate(all_embeddings)
        else:
            all_embeddings = torch.concat(all_embeddings)
        return all_embeddings

    def encode_from_text(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
    ) -> Union[List[torch.Tensor], ndarray, torch.Tensor]:
        """
        Computes sentence embeddings from sentence-transformers library.

        :param sentences: the sentences to embed.
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
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self.device

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sentence) for sentence in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation="longest_first",
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
                            last_mask_id -= 1

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

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
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

    def build_index(self, inputs: BatchEncoding, batch_size: int = 64, use_gpu: bool = True):
        embeddings = self.encode(inputs, batch_size=batch_size)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        index = faiss.IndexFlatL2(len(embeddings[0]))
        if use_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            index = faiss.index_cpu_to_all_gpus(index, co=co)
        index.add(embeddings)
        return index

    def add_to_index(self):
        return

    def search(self):
        return

    def similarity(self, queries: Union[str, List[str]], keys: Union[str, List[str], ndarray]):
        return

    def save(self):
        pass

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        embedder = AutoModel.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(embedder, tokenizer, **kwargs)

    def save_pretrained(self, output_path: str):
        self.tokenizer.save_pretrained(output_path)
        self.model.config.save_pretrained(output_path)

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings


class PairwiseModel(AutoModelForEmbedding):
    """Pairwise Model wrapper
    - bi_encoder
    - cross_encoder
    - poly_encoder
    support: query + pos pair, or query + pos + neg triplet
    """

    def __init__(
        self,
        model_name_or_path: str,
        pooling_method: str = "cls",
        normalize_embeddings: bool = False,
        query_instruction: Optional[str] = None,
        use_fp16: bool = False,
        cross_encoder: bool = False,
        poly_encoder: bool = False,
        temperature: float = 0,
        dynamic_temperature: bool = False,
        loss_fn: Union[nn.Module, Callable] = None,
    ) -> None:
        super().__init__(
            model_name_or_path=model_name_or_path,
            pooling_method=pooling_method,
            normalize_embeddings=normalize_embeddings,
            query_instruction=query_instruction,
            use_fp16=use_fp16,
            loss_fn=None,
        )
        if loss_fn is not None:
            logger.warning("loss_fn in Pairwise embed model, which will be ignored")

        self.cross_encoder = cross_encoder
        self.temperature = temperature
        self.dynamic_temperature = dynamic_temperature

    def forward(
        self,
        inputs: List[torch.Tensor],
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        input1 = inputs[0]
        input2 = inputs[1]
        if len(inputs) == 3:
            input3 = inputs[2]

        if self.cross_encoder:
            ids1, mask1 = input1
            ids2, mask2 = input2
            ids = torch.cat([ids1, ids2], dim=0)
            mask = torch.cat([mask1, mask2], dim=0)

            transformer_out = super().forward({"input_ids": ids, "attention_mask": mask})
            pooled_output = self.pooling(transformer_out[0], mask)
            pooled_output1 = pooled_output[: len(ids1), :]
            pooled_output2 = pooled_output[len(ids1) :, :]
            return pooled_output1, pooled_output2
        else:
            # bi-encoder, pooling in each
            pooled_output1 = super().forward(input1)
            pooled_output2 = super().forward(input2)
            if len(inputs) == 3:
                pooled_output3 = super().forward(input3)
                return pooled_output1, pooled_output2, pooled_output3
            return pooled_output1, pooled_output2


def unsorted_segment_mean(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # init empty result tensor
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class ListwiseModel(AutoModelForEmbedding):
    """
    segment_id
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(
        self,
        inputs,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return

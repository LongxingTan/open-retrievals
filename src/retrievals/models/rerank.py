"""Text reranking model"""

import json
import logging
import os
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from ..data.collator import LLMRerankCollator
from ..losses.colbert_loss import ColbertLoss
from .base import BaseRanker
from .pooling import AutoPooling
from .utils import batch_to_device, check_causal_lm, get_device_name

logger = logging.getLogger(__name__)


class AutoModelForRanking(BaseRanker):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        train_group_size: int = 1,
        pooling_method: Optional[str] = None,
        loss_fn: Union[nn.Module, Callable] = None,
        loss_type: Literal['classification', 'regression'] = 'classification',
        max_length: Optional[int] = None,
        causal_lm: bool = False,
        task_prompt: Optional[str] = None,
        query_instruction: Optional[str] = None,
        document_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, tokenizer)
        if isinstance(model, str):
            assert ValueError("Please use AutoModelForRanking.from_pretrained(model_name_or_path)")

        self.model: Optional[nn.Module] = model
        self.tokenizer = tokenizer
        self.train_group_size = train_group_size
        self.pooling = AutoPooling(self.pooling_method) if pooling_method else None
        self.loss_fn = loss_fn
        self.loss_type = loss_type
        self.causal_lm = causal_lm
        self.task_prompt = task_prompt
        self.query_instruction = query_instruction if query_instruction else "{}"
        self.document_instruction = document_instruction if document_instruction else "{}"
        self.max_length = max_length or self._determine_max_length()
        self.temperature = temperature
        self.device = device or get_device_name()
        self.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.causal_lm:
            return self._forward_causal_lm(input_ids, attention_mask, labels, return_dict, **kwargs)
        return self._forward_cross_encoder(input_ids, attention_mask, labels, return_dict, **kwargs)

    def _forward_causal_lm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        return_dict: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for causal language models."""
        outputs = self.model(input_ids, attention_mask, output_hidden_states=True)
        result = {"logits": outputs.logits}
        if labels is not None:
            result["loss"] = self._compute_loss(outputs.logits, labels)
        return result if return_dict else outputs.logits

    def _forward_cross_encoder(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        return_dict: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for cross-encoder models."""
        features = self.encode(input_ids, attention_mask)
        logits = features.logits

        if self.temperature is not None:
            logits = logits / self.temperature
        if not self.training:
            return logits

        scores = logits.view(-1, self.train_group_size)
        if labels is not None:
            loss = self._compute_loss(scores, labels)
            return {'logits': logits, 'loss': loss}
        return {'logits': logits} if return_dict else logits

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Union[torch.Tensor, SequenceClassifierOutput]:
        """Encode input IDs and attention masks into embeddings."""
        model_output: SequenceClassifierOutput = self.model(input_ids, attention_mask, output_hidden_states=True)

        if not self.pooling:
            return model_output

        last_hidden_state = model_output.get('last_hidden_state', model_output[0])
        embeddings = self.pooling(last_hidden_state, attention_mask)
        return embeddings

    def _compute_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the loss based on the scores and labels."""
        if not self.loss_fn:
            self.loss_fn = nn.MSELoss() if self.loss_type == 'regression' else nn.BCEWithLogitsLoss(reduction='mean')
        return self.loss_fn(scores.squeeze(), labels.squeeze().float())

    def set_model_type(self, model_type: Literal['cross-encoder', 'colbert'], **kwargs):
        logger.info(f'Set model type to: {model_type}')
        model_type = model_type.lower().replace('-', '').replace('_', '')
        model_class = {'crossencoder': self, 'colbert': ColBERT, 'llm': LLMRanker}
        model_class = model_class.get(model_type)
        return model_class(
            model=self.model,
            tokenizer=self.tokenizer,
            loss_fn=self.loss_fn,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        pooling_method: Optional[str] = None,
        num_labels: int = 1,
        loss_fn: Union[nn.Module, Callable] = None,
        loss_type: Literal['classification', 'regression'] = 'classification',
        causal_lm: bool = False,
        generative_llm_reranking: bool = False,
        trust_remote_code: bool = True,
        use_fp16: bool = False,
        use_lora: bool = False,
        use_qlora: bool = False,
        lora_config=None,
        lora_path: Optional[str] = None,
        quantization_config=None,
        task_prompt: Optional[str] = None,
        query_instruction: Optional[str] = None,
        document_instruction: Optional[str] = None,
        device: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        config = AutoConfig.from_pretrained(
            model_name_or_path, output_hidden_states=True, trust_remote_code=trust_remote_code
        )
        if getattr(config, "num_labels", None) is not None:
            if num_labels != config.num_labels:
                logger.info(f"Set num_labels to {config.num_labels} according to config, ignore {num_labels}")
                num_labels = config.num_labels

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, return_tensors=False, trust_remote_code=trust_remote_code
        )

        logger.info('Set model to AutoModelForSequenceClassification, representation reranking')
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        if causal_lm or check_causal_lm(model_name_or_path):
            tokenizer.padding_side = "right"
            tokenizer.add_eos_token = True

        if device is None:
            device = get_device_name()

        if use_fp16 and device != 'cpu' and quantization_config is None and not hasattr(config, 'quantization_config'):
            logger.info('Set model to fp16 in inference, if you want fp16 during training, training_args fp16=True')
            model.half()

        if use_lora or use_qlora:
            logger.info('Set fine-tuning to LoRA')
            model = cls.setup_lora(model, lora_config, use_qlora)

        if lora_path is not None:
            model = cls.load_lora_weights(model, lora_path)

        reranker = cls(
            model=model,
            tokenizer=tokenizer,
            pooling_method=pooling_method,
            device=device,
            loss_fn=loss_fn,
            loss_type=loss_type,
            temperature=temperature,
            causal_lm=causal_lm,
            task_prompt=task_prompt,
            query_instruction=query_instruction,
            document_instruction=document_instruction,
        )
        return reranker

    @torch.no_grad()
    @torch.amp.autocast('cuda')
    def predict(
        self,
        sentences: Union[Tuple[str], List[str]],
        batch_size: int = 16,
        max_length: int = 512,
        show_progress_bar: bool = None,
        normalize: bool = False,
    ):
        self.model.eval()

        length_sorted_idx = np.argsort([-self._text_length(p) for p in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        all_scores: List[List[float]] = []
        for batch_start in tqdm(
            range(0, len(sentences_sorted), batch_size), desc='Scoring', disable=not show_progress_bar
        ):
            batch_sentences = sentences_sorted[batch_start : batch_start + batch_size]
            batch_on_device = self.preprocess_pair(batch_sentences, max_length=max_length, pairs=False)
            scores = self.model(**batch_on_device, return_dict=True).logits[:, 1]

            if normalize:
                scores = torch.sigmoid(scores)
            all_scores.extend(scores.cpu().float().tolist())

        all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]
        return all_scores


class ColBERT(BaseRanker):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        linear_layer: Optional[nn.Module] = None,
        max_length: Optional[int] = None,
        loss_fn: Union[nn.Module, Callable] = None,
        temperature: float = 1.0,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.linear = linear_layer

        self.loss_fn = loss_fn
        self.temperature = temperature
        self.max_length = max_length
        self.device = device or get_device_name()
        self.to(self.device)

    def forward(
        self,
        query_input_ids: Optional[torch.Tensor],
        query_attention_mask: Optional[torch.Tensor],
        pos_input_ids: Optional[torch.Tensor],
        pos_attention_mask: Optional[torch.Tensor],
        neg_input_ids: Optional[torch.Tensor] = None,
        neg_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        query_embedding = self._encode(query_input_ids, attention_mask=query_attention_mask, normalize_embeddings=True)
        positive_embedding = self._encode(pos_input_ids, attention_mask=pos_attention_mask, normalize_embeddings=True)

        if self.training:
            negative_embedding = None
            if neg_input_ids is not None:
                negative_embedding = self._encode(
                    neg_input_ids, attention_mask=neg_attention_mask, normalize_embeddings=True
                )
            loss = self._compute_loss(query_embedding, positive_embedding, negative_embedding, query_attention_mask)
            return {'loss': loss}
        return self.score(query_embedding, positive_embedding, query_attention_mask)

    def _encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, normalize_embeddings: bool = True
    ) -> torch.Tensor:
        """Encode input IDs and attention masks into embeddings."""
        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_state = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs.hidden_states[1]
        embeddings = self.linear(hidden_state[:, 1:])
        embeddings = embeddings * attention_mask[:, 1:].unsqueeze(-1).float()
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def _compute_loss(
        self,
        query_embedding: torch.Tensor,
        positive_embedding: torch.Tensor,
        negative_embedding: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the loss for ColBERT."""
        if not self.loss_fn:
            self.loss_fn = ColbertLoss(temperature=self.temperature, use_inbatch_negative=False)
        return self.loss_fn(query_embedding, positive_embedding, negative_embedding, attention_mask)

    def preprocess_pair(
        self,
        batch_sentence_pair: List[Tuple[str, str]],
        query_max_length: int,
        document_max_length: int,
        padding='max_length',
        **kwargs,
    ):
        query_list = [item[0] for item in batch_sentence_pair]
        document_list = [item[1] for item in batch_sentence_pair]

        query_batch_tokens = self.tokenizer(
            query_list, padding=padding, truncation=True, max_length=query_max_length, return_tensors='pt'
        )
        query_batch_tokens_on_device = {k: v.to(self.device) for k, v in query_batch_tokens.items()}
        document_batch_tokens = self.tokenizer(
            document_list, padding=padding, truncation=True, max_length=document_max_length, return_tensors='pt'
        )
        document_batch_tokens_on_device = {k: v.to(self.device) for k, v in document_batch_tokens.items()}

        return {
            "query_input_ids": query_batch_tokens_on_device['input_ids'],
            "query_attention_mask": query_batch_tokens_on_device['attention_mask'],
            "doc_input_ids": document_batch_tokens_on_device['input_ids'],
            "doc_attention_mask": document_batch_tokens_on_device['attention_mask'],
        }

    def encode(
        self,
        sentences: Union[str, List[str], Tuple[str], np.ndarray],
        batch_size: int = 16,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = True,
    ):

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
                embeddings = self._encode(**features, normalize_embeddings=normalize_embeddings)
                embeddings = embeddings.detach()

                if convert_to_numpy:
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            if len(all_embeddings) > 0:
                all_embeddings = torch.stack(all_embeddings)
            else:
                all_embeddings = torch.Tensor()
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    @torch.no_grad()
    def compute_score(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 16,
        max_length: int = 256,
        normalize: bool = False,
        show_progress_bar: bool = None,
        **kwargs,
    ):
        self.model.eval()
        if isinstance(sentence_pairs[0], str) and len(sentence_pairs) == 2:
            sentence_pairs = [sentence_pairs]

        scores_list: List[float] = []
        for i in tqdm(range(0, len(sentence_pairs), batch_size), desc="Scoring", disable=not show_progress_bar):
            batch_on_device = self.preprocess_pair(
                sentence_pairs[i : i + batch_size], query_max_length=max_length, document_max_length=max_length
            )
            query_embedding = self._encode(
                batch_on_device['query_input_ids'], batch_on_device['query_attention_mask'], normalize_embeddings=True
            )
            doc_embedding = self._encode(
                batch_on_device['doc_input_ids'], batch_on_device['doc_attention_mask'], normalize_embeddings=True
            )
            scores = self.score(
                query_embedding, doc_embedding, query_attention_mask=batch_on_device['query_attention_mask']
            )
            if normalize:
                scores = torch.sigmoid(scores)
            scores_list.extend(scores.cpu().numpy().tolist())

        if len(scores_list) == 1:
            return scores_list[0]
        return scores_list

    def score(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        query_attention_mask: Optional[torch.Tensor] = None,
    ):
        # pair embedding -> score, mask the pad token while calculate similarity score
        late_interactions = torch.einsum(
            "bsh,bth->bst",
            query_embeddings,
            document_embeddings,
        )
        # sum(1) will sum up in sequence_length dim, later divide the sum of non-pad token
        late_interactions = late_interactions.max(2).values.sum(1)
        if query_attention_mask is not None:
            late_interactions = late_interactions / query_attention_mask[:, 1:].sum(-1, keepdim=False)
        else:  # Or length of token sequence
            late_interactions = late_interactions / query_embeddings.size(1)
        return late_interactions

    def save_pretrained(self, save_directory: Union[str, os.PathLike], safe_serialization: bool = True):
        logger.info("Save model to {}".format(save_directory))
        state_dict_fn = lambda state_dict: type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        state_dict = self.model.state_dict()
        self.model.save_pretrained(
            save_directory, state_dict=state_dict_fn(state_dict), safe_serialization=safe_serialization
        )
        torch.save(state_dict_fn(self.linear.state_dict()), os.path.join(save_directory, 'colbert_linear.pt'))
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        colbert_dim: int = 1024,
        pretrained_linear_name: str = 'colbert_linear.pt',
        loss_fn: Union[nn.Module, Callable] = ColbertLoss(),
        trust_remote_code: bool = True,
        use_fp16: bool = False,
        device: Optional[str] = None,
        **kwargs,
    ):
        if not os.path.exists(model_name_or_path):
            """To load linear layer weight, manually download the model so the model_name_or_path will always be path"""
            from huggingface_hub import snapshot_download

            cache_dir = os.getenv('HF_HUB_CACHE')
            model_name_or_path = snapshot_download(
                repo_id=model_name_or_path,
                cache_dir=cache_dir,
                ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5'],
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, **kwargs)

        linear_layer = nn.Linear(model.config.hidden_size, colbert_dim)
        if os.path.exists(path=os.path.join(model_name_or_path, pretrained_linear_name)):
            logger.info(
                f'Loading colbert_linear pretrained weight from {model_name_or_path}, colbert_dim={colbert_dim}'
            )
            colbert_state_dict = torch.load(
                os.path.join(model_name_or_path, pretrained_linear_name), map_location='cpu'
            )
            linear_layer.load_state_dict(colbert_state_dict)
        else:
            logger.info(f'Xavier uniform random colbert linear layer,  colbert_dim={colbert_dim}')
            torch.nn.init.xavier_uniform_(tensor=linear_layer.weight)

        if os.path.exists(path=os.path.join(model_name_or_path, "metadata.json")):
            with open(file=os.path.join(model_name_or_path, "metadata.json"), mode="r") as f:
                metadata = json.load(fp=f)
            max_length = metadata["max_length"]
            print(max_length)

        if use_fp16:
            logger.info('Set model to fp16, please note that if you want fp16 during training, set training_args fp16')
            model.half()
            linear_layer.half()

        ranker = cls(
            model=model,
            tokenizer=tokenizer,
            linear_layer=linear_layer,
            device=device,
            loss_fn=loss_fn,
        )
        return ranker


class LLMRanker(AutoModelForRanking):
    """LLM Generative Reranker"""

    def __init__(
        self,
        task_prompt: Optional[str] = None,
        target_token: str = 'Yes',
        sep_token: str = '\n',
        query_instruction: Optional[str] = 'A: {}',
        document_instruction: Optional[str] = 'B: {}',
        **kwargs,
    ):
        super().__init__(**kwargs)
        if task_prompt is None:
            task_prompt = (
                """Given a query A and a passage B, determine whether the passage contains an answer to the query """
                """by providing a prediction of either 'Yes' or 'No'."""
            )
        self.task_prompt = task_prompt
        self.sep_token = sep_token
        self.target_token_loc = self.tokenizer(target_token, add_special_tokens=False)['input_ids'][0]
        self.query_instruction = query_instruction
        self.document_instruction = document_instruction

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ):
        model_output = self.model(input_ids, attention_mask, output_hidden_states=True)

        outputs_dict = dict()
        outputs_dict['logits'] = model_output.logits
        if self.training:
            loss = self.loss_fn(model_output.logits, labels)
            outputs_dict['loss'] = loss
        return outputs_dict

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        pooling_method: Optional[str] = None,
        num_labels: int = 1,
        loss_fn: Union[nn.Module, Callable] = None,
        loss_type: Literal['classification', 'regression'] = 'classification',
        causal_lm: bool = False,
        generative_llm_reranking: bool = False,
        trust_remote_code: bool = True,
        use_fp16: bool = False,
        use_lora: bool = False,
        use_qlora: bool = False,
        lora_config=None,
        lora_path: Optional[str] = None,
        quantization_config=None,
        task_prompt: Optional[str] = None,
        query_instruction: Optional[str] = 'A: {}',
        document_instruction: Optional[str] = 'B: {}',
        device: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):

        config = AutoConfig.from_pretrained(
            model_name_or_path, output_hidden_states=True, trust_remote_code=trust_remote_code
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, return_tensors=False, trust_remote_code=trust_remote_code
        )
        tokenizer.padding_side = "right"
        tokenizer.add_eos_token = True

        logger.info("Loading the model as AutoModelForCausalLM for LLM generative reranking.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        if device is None:
            device = get_device_name()

        if use_fp16 and device != 'cpu' and not hasattr(config, 'quantization_config'):
            logger.info('Set model to fp16')
            model.half()

        if use_lora or use_qlora:
            logger.info('Set fine-tuning to LoRA')
            model = cls.setup_lora(model, lora_config, use_qlora)

        if lora_path is not None:
            model = cls.load_lora_weights(model, lora_path)

        reranker = cls(
            model=model,
            tokenizer=tokenizer,
            pooling_method=pooling_method,
            loss_fn=loss_fn,
            loss_type=loss_type,
            temperature=temperature,
            causal_lm=causal_lm,
            task_prompt=task_prompt,
            query_instruction=query_instruction,
            document_instruction=document_instruction,
            device=device,
            **kwargs,
        )

        if reranker.task_prompt is None:
            reranker.task_prompt = (
                """Given a query A and a passage B, determine whether the passage contains an answer to the query """
                """by providing a prediction of either 'Yes' or 'No'."""
            )

        reranker.target_token_loc = reranker.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
        reranker.sep_token = '\n'

        return reranker

    def preprocess_pair(self, batch_sentence_pair: List[List[str]], max_length: int, **kwargs):
        collator = LLMRerankCollator(
            tokenizer=self.tokenizer,
            prompt=self.task_prompt,
            sep_token=self.sep_token,
            max_length=max_length,
        )
        batch_inputs = collator(batch_sentence_pair)

        batch_inputs = {
            'input_ids': batch_inputs['input_ids'].to(self.device),
            'attention_mask': batch_inputs['attention_mask'].to(self.device),
        }
        return batch_inputs

    @torch.no_grad()
    def compute_score(
        self,
        sentence_pairs: Union[List[Tuple[str]], Tuple[str]],
        batch_size: int = 16,
        max_length: int = 256,
        normalize: bool = False,
        show_progress_bar: bool = None,
        **kwargs,
    ):
        self.model.eval()

        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        length_sorted_idx = np.argsort([-self._text_length(q) - self._text_length(p) for q, p in sentence_pairs])
        sentences_sorted = [sentence_pairs[idx] for idx in length_sorted_idx]

        if self.query_instruction or self.document_instruction:
            sentences_sorted = [
                (self.query_instruction.format(pair[0]), self.document_instruction.format(pair[1]))
                for pair in sentences_sorted
            ]

        all_scores: List[float] = []
        for batch_start in tqdm(
            range(0, len(sentences_sorted), batch_size), desc='Scoring', disable=not show_progress_bar
        ):
            batch_sentences = sentences_sorted[batch_start : batch_start + batch_size]
            batch_on_device = self.preprocess_pair(batch_sentences, max_length=max_length)
            outputs = self.model(**batch_on_device)
            scores = self.score(outputs['logits'])

            if normalize:
                scores = torch.sigmoid(scores)
            all_scores.extend(scores.cpu().float().tolist())

        all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]

        if len(all_scores) == 1:
            return all_scores[0]

        return all_scores

    def score(self, logits: torch.Tensor):
        scores = logits[:, -1, self.target_token_loc]  # for left_padding
        return scores

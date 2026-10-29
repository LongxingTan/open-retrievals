"""Triton Python backend copied into exported model repositories."""

import json
from pathlib import Path

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoModel, AutoTokenizer


class TritonPythonModel:
    def initialize(self, args):
        device = "cpu"
        if args["model_instance_kind"] == "GPU":
            device = f"cuda:{args['model_instance_device_id']}"
        checkpoint = Path(__file__).resolve().parent / "checkpoint"
        self.device = device
        self.metadata = json.loads((checkpoint / "embedding_config.json").read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=False).to(device)
        self.model.eval()

    def embed(self, texts):
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.metadata["max_length"], return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            hidden = self.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"]
            pooling = self.metadata["pooling_method"]
            if pooling in {"cls", "first"}:
                embeddings = hidden[:, 0]
            elif pooling in {"mean", "avg", "average"}:
                weights = mask.unsqueeze(-1).to(hidden.dtype)
                embeddings = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
            elif pooling in {"last", "eos"}:
                positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
                indices = positions.masked_fill(~mask.bool(), -1).max(1).values
                embeddings = hidden[torch.arange(len(hidden), device=hidden.device), indices]
            else:
                raise ValueError(f"Unsupported pooling: {pooling}")
            if self.metadata.get("normalize_embeddings"):
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.float().cpu().numpy().astype(np.float32)

    def execute(self, requests):
        responses = [None] * len(requests)
        texts = []
        slices = []
        for index, request in enumerate(requests):
            try:
                values = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
                if values.ndim != 2 or values.shape[1] != 1 or not len(values):
                    raise ValueError("TEXT must have shape [batch, 1]")
                start = len(texts)
                texts.extend(
                    value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values[:, 0]
                )
                slices.append((index, start, len(texts)))
            except Exception as error:
                responses[index] = pb_utils.InferenceResponse(error=pb_utils.TritonError(str(error)))
        if texts:
            try:
                embeddings = self.embed(texts)
                for index, start, end in slices:
                    tensor = pb_utils.Tensor("EMBEDDING", embeddings[start:end])
                    responses[index] = pb_utils.InferenceResponse(output_tensors=[tensor])
            except Exception as error:
                for index, _, _ in slices:
                    responses[index] = pb_utils.InferenceResponse(error=pb_utils.TritonError(str(error)))
        return responses

"""Triton Python backend copied into exported model repositories."""

from pathlib import Path

import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from retrievals.models import AutoModelForEmbedding


class TritonPythonModel:
    def initialize(self, args):
        device = "cpu"
        if args["model_instance_kind"] == "GPU":
            device = f"cuda:{args['model_instance_device_id']}"
        checkpoint = Path(__file__).resolve().parent / "checkpoint"
        self.model = AutoModelForEmbedding.from_pretrained(str(checkpoint), device=device, trust_remote_code=False)
        self.model.eval()

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                values = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
                texts = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
                if values.ndim != 1 or not texts:
                    raise ValueError("TEXT must be a nonempty one-dimensional string array")
                tokens = self.model.tokenizer(
                    texts, padding=True, truncation=True, max_length=self.model.max_length, return_tensors="pt"
                ).to(self.model.device)
                with torch.inference_mode():
                    embeddings = self.model.forward_from_tensor(tokens["input_ids"], tokens["attention_mask"])
                tensor = pb_utils.Tensor("EMBEDDING", embeddings.float().cpu().numpy().astype(np.float32))
                responses.append(pb_utils.InferenceResponse(output_tensors=[tensor]))
            except Exception as error:
                responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(str(error))))
        return responses

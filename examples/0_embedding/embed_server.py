import argparse
import concurrent.futures
from abc import ABC, abstractmethod
from typing import List

from fastapi import FastAPI, HTTPException
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from pydantic import BaseModel
from transformers import AutoTokenizer

LOCAL_EMBED_THREADS = 1
LOCAL_EMBED_MAX_LENGTH = 512
LOCAL_EMBED_BATCH_SIZE = 1
LOCAL_EMBED_MODEL_PATH = ""


class EmbeddingBackend(ABC):
    # https://github.com/netease-youdao/QAnything
    embed_version = "local_v0.0.1"

    def __init__(self, use_cpu: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_EMBED_MODEL_PATH)
        self.workers = LOCAL_EMBED_THREADS

        sess_options = SessionOptions()
        sess_options.intra_op_num_threads = 0
        sess_options.inter_op_num_threads = 0
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        if use_cpu:
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self._session = InferenceSession(LOCAL_EMBED_MODEL_PATH, sess_options=sess_options, providers=providers)

    def get_embedding(self, sentences: List[str], max_length: int):
        inputs_onnx = self.tokenizer(
            sentences, padding=True, truncation=True, max_length=max_length, return_tensors='np'
        )
        inputs_onnx = {k: v for k, v in inputs_onnx.items()}

        outputs_onnx = self._session.run(output_names=['output'], input_feed=inputs_onnx)
        embeddings = outputs_onnx[0][:, 0]
        return embeddings

    def get_len_safe_embeddings(self, texts: List[str]):
        all_embeddings = []
        batch_size = LOCAL_EMBED_BATCH_SIZE

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                future = executor.submit(self.get_embedding, batch, LOCAL_EMBED_MAX_LENGTH)
                futures.append(future)

            for future in futures:
                embeddings = future.result()
                all_embeddings += embeddings
        return all_embeddings

    @property
    def get_model_version(self):
        return self.embed_version

    def inference(self, inputs):
        outputs_onnx = None
        try_num = 2
        while outputs_onnx is None and try_num > 0:
            try:
                io_binding = self._session.io_binding()
                for k, v in inputs.items():
                    io_binding.bind_cpu_input(k, v)
                io_binding.synchronize_inputs()
                io_binding.bind_output('output')

                self._session.run_with_iobinding(io_binding)

                io_binding.synchronize_outputs()
                outputs_onnx = io_binding.copy_outputs_to_cpu()
                io_binding.clear_binding_inputs()
                io_binding.clear_binding_outputs()
            except ValueError:
                outputs_onnx = None
            try_num -= 1

        return outputs_onnx

    def predict(self, queries, return_tokens_num=False):
        embeddings = []
        return embeddings


parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', action="store_true", help='use gpu or not')
parser.add_argument('--workers', type=int, default=1, help='workers')
args = parser.parse_args()
print("args:", args)

app = FastAPI()


class EmbeddingRequest(BaseModel):
    texts: List[str]


onnx_backend = EmbeddingBackend(use_cpu=not args.use_gpu)


@app.post("/embedding")
async def embedding(request: EmbeddingRequest):
    texts = request.texts
    result_data = onnx_backend.predict(texts)
    return result_data


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9001, workers=args.workers)

# Serve a trained embedding model

Train through the YAML workflow first. Its final `outputs/<run>/model`
directory contains Hugging Face weights, tokenizer files and
`embedding_config.json`. Use that directory, not a resumable Trainer checkpoint.
Exports copy files into a **new** directory without changing the training artifact.

## vLLM

```bash
rag-cli export --model outputs/run/model --backend vllm --output serving/embedding
vllm serve serving/embedding --runner pooling --convert embed --served-model-name embedding --max-model-len 512
curl http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embedding","input":["cat","cat dog"]}'
```

Install vLLM in a separate serving environment using its hardware-specific
installation instructions. Set `--max-model-len` to the checkpoint's
`embedding_config.json:max_length` (512 above is an example).
The export writes Sentence Transformers `modules.json`, pooling configuration,
normalization module metadata and maximum sequence length. No Open-Retrievals
installation is needed inside vLLM.

vLLM must support the checkpoint's backbone architecture. This exporter does
not convert unsupported architectures into supported ones. Check the supported
models list for your installed version. Current vLLM resolves pooling and
normalization from Sentence Transformers metadata; older releases may differ.
For current versions an explicit override is, for example:
`--pooler-config '{"pooling_type":"MEAN","use_activation":false}'`.
Do not use that example unless it matches the saved model.

## Triton Python backend

```bash
rag-cli export --model outputs/run/model --backend triton --output serving/model_repository
tritonserver --model-repository=/absolute/path/to/serving/model_repository
curl http://localhost:8000/v2/models/embedding/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"TEXT","shape":[2],"datatype":"BYTES","data":["cat","cat dog"]}],"outputs":[{"name":"EMBEDDING"}]}'
```

Run the server in an NVIDIA Triton Python-backend image. Install this project's
wheel (and its torch/transformers dependencies) in the Python environment used by
the backend stub. Match that environment's Python version to the Triton image;
a host installation is not sufficient for a container. Mount the exported
repository at the path passed to `--model-repository`.

The export contains `embedding/config.pbtxt`, `embedding/1/model.py` and a
self-contained `embedding/1/checkpoint/`. CPU is the portable default.
For GPU serving, change `KIND_CPU` to `KIND_GPU` in `config.pbtxt` and provide
a CUDA-compatible torch installation. Each model instance loads its own weights.

Input TEXT is a nonempty vector of UTF-8 strings. Output EMBEDDING is a
float32 matrix [number of texts, hidden dimension]. Each request is tokenized
as one batch and truncated to the saved maximum length. Server-side dynamic
batching is disabled (`max_batch_size: 0`); choose client batch sizes to fit
memory. Errors are reported per request.

## Scope and parity

Both exports support dense text embeddings with CLS, mean or last-token pooling,
including their aliases, and preserve the saved normalization flag. Adapter-only
checkpoints and quantized checkpoints are rejected: merge adapters into an
unquantized backbone before export. Weighted pooling, ColBERT, rerankers, VLMs,
and custom projection heads are not supported by this export contract.

Send the same fully formatted text used for local inference, including query or
document instructions. Servers do not automatically apply role-specific prompt
templates. Tokenizer special tokens, padding, input length and normalization must
match when comparing results. Triton truncates long input; for vLLM, truncate
client-side with the saved tokenizer or use your version's documented truncation
option. CLS exports for vLLM require right padding.

Offline tests exercise the generated Triton adapter with real tiny BERT weights
and a simulated Triton protocol for all three pooling modes and normalization
settings. They do not certify a live vLLM/Triton deployment. Before production,
compare local and server embeddings on representative text with a tolerance
appropriate to the serving dtype; also check nearest-neighbor agreement.

## Research references (2026-10-27)

- [vLLM pooling models](https://docs.vllm.ai/en/latest/models/pooling_models/):
  pooling runner, architecture conversion and Sentence Transformers metadata precedence.
- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/):
  check backbone support for the installed release.
- [NVIDIA Hugging Face deployment tutorial](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Quick_Deploy/HuggingFaceTransformers/README.html):
  Python backend deployment of Hugging Face models.
- [Triton Python backend](https://github.com/triton-inference-server/python_backend):
  repository layout, initialize/execute contract, runtime dependencies and errors.

The Python backend was chosen to reuse the exact embedding forward path without
requiring an ONNX/TensorRT graph conversion. It does not provide TensorRT
optimization or vLLM scheduling.

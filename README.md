<div align="center">

<h1 align="center">
<img src="./docs/source/_static/logo.svg" width="420" align=center/>
</h1>

**Open-Retrievals**: The Unified Training Framework for Embeddings, Retrieval, and RAG

 [![LICENSE][license-image]][license-url]
 [![PyPI Version][pypi-image]][pypi-url]
 [![Build Status][build-image]][build-url]
 [![Lint Status][lint-image]][lint-url]
 [![Docs Status][docs-image]][docs-url]
 [![Code Coverage][coverage-image]][coverage-url]
 [![Contributing][contributing-image]][contributing-url]

 **[Documentation](https://open-retrievals.readthedocs.io/en/latest/)** | **[中文](https://github.com/LongxingTan/open-retrievals/blob/master/README_zh-CN.md)** | **[日本語](https://github.com/LongxingTan/open-retrievals/blob/master/README_ja-JP.md)**

</div>

Open-Retrievals provides models, losses, and pipelines for text embeddings,
retrieval, reranking, and RAG. YAML training uses a small assembly function and
explicit task trainers built on Hugging Face Transformers.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use Python 3.10 or newer:

```bash
uv sync --locked
uv run rag-cli version
make test
```

Dependencies are defined in `pyproject.toml`; `uv.lock` pins reproducible versions.
Use `uv add PACKAGE` for runtime libraries and `uv add --dev PACKAGE` for development tools.
Run `uv lock` after editing dependencies and commit both files. Optional features:
`uv sync --extra lora`, `uv sync --extra eval`; documentation: `uv sync --group docs`.
For a runtime-only environment, use `uv sync --no-dev`.

Pip installation is also supported:


From the repository root, install the current code and training dependencies:

```bash
pip install -e .
# Optional LoRA support
pip install -e '.[lora]'
```

## Train from YAML

```bash
rag-cli train examples/configs/dense_retrieval.yaml
rag-cli train examples/configs/cross_encoder.yaml
rag-cli train examples/configs/colbert.yaml
```

These examples use repository fixture data. Replace the paths with your training
data. Model identifiers download pretrained weights; local directories work offline.

```yaml
task_type: dense_retrieval
output_dir: ./outputs/dense
model:
  model_name_or_path: bert-base-uncased
  pooling_method: mean
  normalize_embeddings: true
data:
  train_file: tests/fixtures/retrieval_train.jsonl
  query_max_length: 32
  doc_max_length: 128
  negative_samples: 2
training:
  num_epochs: 1
  batch_size: 2
  learning_rate: 2e-5
loss:
  loss_type: infonce
  temperature: 0.05
```

```bash
rag-cli train examples/configs/dense_retrieval.yaml training.learning_rate=1e-5
```

The supported YAML workflows are:

| Task | Data | Loss |
| --- | --- | --- |
| Dense retrieval | Query, positive(s), optional negatives | InfoNCE or triplet |
| Cross-encoder classification | Query, document, label | BCE with logits |
| ColBERT | Query, positive(s), optional negatives | Masked late interaction |

Encoder and decoder dense models share the same training path. Choose the pooling
method explicitly, such as `mean` for an encoder or `last` for a decoder. LoRA is
optional. Older Python model, loss, and trainer APIs have no compatibility guarantee.

## Install and train

```bash
pip install open-retrievals
rag-cli init dense --output train.yaml
rag-cli validate train.yaml
rag-cli train train.yaml training.learning_rate=1e-5
rag-cli evaluate outputs/dense/model/config.yaml
```

Starter templates and sample data ship with the package. Replace the sample data
for real training. Dataset and output paths are relative to the YAML file.
Pretrained weights require a first download. Use `cross_encoder` or `colbert`
in place of `dense` for those tasks. Install `open-retrievals[lora]` for adapters.

Models retain their inference API:

```python
from retrievals import AutoModelForEmbedding

model = AutoModelForEmbedding.from_pretrained("outputs/dense/model")
embeddings = model.encode(["What is retrieval?", "Finding relevant documents."])
```

For reranking, load `AutoModelForRanking` and call `compute_score()` on
query/document pairs. ColBERT uses `ColBERT.from_pretrained()` with the exported
projection weights and matching `colbert_dim`. LoRA exports are adapters; reload them through the saved YAML
configuration with `build_trainer()` or `evaluate()`.

## Evaluate, save, and resume

Set `data.valid_file` and `training.eval_strategy: epoch` to evaluate during
training. Evaluation reports task loss; corpus-level MRR/NDCG/Recall evaluation
remains separate from the YAML training loop.

Exports are saved in `output_dir/model/`, including tokenizer and resolved config.
Trainer checkpoints in `output_dir/checkpoint-*` include optimizer and scheduler
state for resuming.

```bash
rag-cli evaluate outputs/dense/model/config.yaml \
  --model_path outputs/dense/model --eval_file data/valid.jsonl
rag-cli train examples/configs/dense_retrieval.yaml \
  training.resume_from_checkpoint=outputs/dense/checkpoint-100
```

## Distributed execution and scope

```bash
torchrun --nproc_per_node=2 --module retrievals.train \
  examples/configs/dense_retrieval.yaml
```

The YAML path uses Transformers Trainer/DDP with negatives local to each rank.
The offline tests do not establish multi-GPU performance. Ray, verl,
LLaMA-Factory execution, quantized YAML training, and YAML distillation are not
implemented. Unsupported backend selections fail explicitly. The existing
YAML distillation is not implemented.

## Design and examples

- [Architecture](docs/architecture.md): component responsibilities and persistence.
- [YAML guide](YAML_CONFIG_README.md): configuration, column mappings, and LoRA.
- [Configuration examples](examples/configs/README.md).
- [Embedding examples](examples/0_embedding).
- [Retrieval examples](examples/1_retrieval).
- [Reranking examples](examples/2_reranking).
- [ColBERT examples](examples/3_colbert).
- [RAG examples](examples/4_rag).

## Development

```bash
make style
make test
```

For deterministic offline workflow tests using tiny locally constructed models:

```bash
HF_HUB_OFFLINE=1 python -m unittest tests.test_trainer.test_training_workflows
```

Some legacy tests require downloaded models or additional optional dependencies.

## License

[Apache License 2.0](LICENSE).

[license-image]: https://img.shields.io/badge/License-Apache%202.0-blue.svg
[license-url]: https://opensource.org/licenses/Apache-2.0
[pypi-image]: https://badge.fury.io/py/open-retrievals.svg
[pypi-url]: https://pypi.org/project/open-retrievals
[pepy-image]: https://pepy.tech/badge/retrievals/month
[pepy-url]: https://pepy.tech/project/retrievals
[build-image]: https://github.com/LongxingTan/open-retrievals/actions/workflows/test.yml/badge.svg?branch=master
[build-url]: https://github.com/LongxingTan/open-retrievals/actions/workflows/test.yml?query=branch%3Amaster
[lint-image]: https://github.com/LongxingTan/open-retrievals/actions/workflows/lint.yml/badge.svg?branch=master
[lint-url]: https://github.com/LongxingTan/open-retrievals/actions/workflows/lint.yml?query=branch%3Amaster
[docs-image]: https://readthedocs.org/projects/open-retrievals/badge/?version=latest
[docs-url]: https://open-retrievals.readthedocs.io/en/master/
[coverage-image]: https://codecov.io/gh/longxingtan/open-retrievals/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/gh/longxingtan/open-retrievals?branch=master
[contributing-image]: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]: https://github.com/longxingtan/open-retrievals/blob/master/CONTRIBUTING.md

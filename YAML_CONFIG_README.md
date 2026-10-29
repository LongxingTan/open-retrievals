# YAML training

The supported YAML tasks are dense retrieval, pointwise cross-encoder reranking,
and ColBERT. They share a Transformers training engine. YAML configuration is the supported
public training interface. Older Python model, loss, and trainer APIs have no
compatibility guarantee.

```bash
pip install -e .
rag-cli train examples/configs/dense_retrieval.yaml
rag-cli train examples/configs/dense_retrieval.yaml training.learning_rate=1e-5
```

The examples use small fixture datasets. Replace their paths for real training.
Pretrained model identifiers require an initial download; local model directories
work offline.

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
  mixed_precision: "no"
loss:
  loss_type: infonce
  temperature: 0.05
```

The canonical training fields are `num_epochs`, `batch_size`, and
`eval_batch_size`. Unknown fields and older aliases are rejected.
Overrides use canonical dotted keys. Dataset and output paths are relative to the YAML file.

## Data

Dense retrieval and ColBERT:

```json
{"query": "cat", "positive": ["pet"], "negative": ["car", "road"]}
```

Positive-only dense batches use in-batch negatives. Set `negative_samples: 0`
to omit explicit negatives. Every row in a batch must agree on whether negatives
are present. ColBERT without explicit negatives requires `loss.use_inbatch_negative: true`.

Cross-encoder classification:

```json
{"query": "cat", "document": "pet", "label": 1}
```

Map source columns using `query_col`, `positive_col`, `negative_col`, `doc_col`,
and `label_col`. Labels for classification are in [0, 1]. Local file extensions
select JSON/JSONL, CSV, or Parquet loading. For Hub datasets, set `dataset_name`,
optional `dataset_config`, and `dataset_split` instead of `train_file`. Validation
requires an explicit `valid_file`; `valid_split` does not create a holdout.
CSV positive/negative columns should contain individual text strings; JSON/JSONL
and Parquet can also store lists of texts. Query/document instructions for YAML
training must be included in the input text.

## Evaluation, checkpoints, and export

Set `data.valid_file` and `training.eval_strategy: epoch` to evaluate during
training. Evaluation reports task loss, not corpus retrieval metrics.

The export lives in `output_dir/model/`. Its config points at the exported model:

```bash
rag-cli evaluate outputs/dense/model/config.yaml \
  data.valid_file=/absolute/path/to/data/valid.jsonl
rag-cli train examples/configs/dense_retrieval.yaml \
  training.resume_from_checkpoint=outputs/dense/checkpoint-100
```

Evaluation does not load `data.train_file`. Resume from `checkpoint-*`, not the
inference export. LoRA adapters retain a dependency on their original base model.

```python
from retrievals.train import build_trainer, evaluate, train

trainer = train("examples/configs/dense_retrieval.yaml")
# evaluate("outputs/dense/model/config.yaml")  # requires data.valid_file
# Reload an export (including LoRA adapters) without reading training data:
# trainer = build_trainer("outputs/dense/model/config.yaml", training=False)
```

## Optional execution

Use `model.use_lora: true` and configure `lora_r`, `lora_alpha`, `lora_dropout`,
and comma-separated `lora_target_modules`. Install `.[lora]` for adapters.
For decoder embeddings choose `pooling_method: last` and the appropriate tokenizer.

```bash
torchrun --nproc_per_node=2 --module retrievals.train \
  examples/configs/dense_retrieval.yaml
```

Multi-GPU execution uses Trainer/DDP and local-rank negatives. It has not been
GPU-benchmarked by the offline tests. Ray, verl, LLaMA-Factory backends, quantized
YAML training, punctuation masking, and YAML distillation are rejected explicitly.
Distillation requires a YAML workflow before it becomes a supported training task.

# Implemented training features

| YAML workflow | Inputs | Objective |
| --- | --- | --- |
| Dense retrieval | Queries, positives, optional negatives | InfoNCE or triplet |
| Cross-encoder | Labeled query/document pairs | Binary cross entropy with logits |
| ColBERT | Queries, positives, optional negatives | Late interaction with token masks |

All three use Transformers Trainer. YAML loading and Python dictionaries share the
same schema and validation. File-based JSON/JSONL, CSV, and Parquet datasets and
Hugging Face datasets are normalized before collation. Explicit source column
mappings are supported.

The offline tests cover encoder and decoder dense backbones, LoRA adapter export,
training updates, evaluation loss, export/reload, and dense checkpoint resume.
Mixed precision and multi-process execution are delegated to Transformers; GPU
performance is not established by those tests.

Exports contain model/adapter weights, tokenizer, and resolved configuration.
ColBERT projection weights and dense pooling/normalization settings are preserved.
Evaluation does not require loading a training dataset.

Existing Python models, losses, and retrieval utilities remain available without
compatibility guarantees. Legacy trainers and training pipelines have been removed;
use the YAML workflows instead. Not every model/loss combination has a YAML workflow.
Ray, verl, LLaMA-Factory execution, quantized training, and YAML distillation are
not implemented and are rejected where selected.

See [architecture](architecture.md) and [the YAML guide](../YAML_CONFIG_README.md).

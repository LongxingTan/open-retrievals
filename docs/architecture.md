# Training architecture

The YAML entry point uses Transformers Trainer for optimization, mixed precision,
checkpoint state, and distributed execution. Training is assembled in
`src/retrievals/train.py`; there is no backend registry or separate execution engine.

## Execution

```text
YAML or Python config
  -> resolve_config: schema, validation
  -> build_trainer:
       training arguments
       normalized dataset + task collator
       backbone + task head
       task trainer + loss
  -> train / evaluate
  -> export model + resolved config
```

`DenseTrainer` encodes queries, positives, and optional negatives, then calls a
contrastive loss. `CrossEncoderTrainer` scores labeled query/document pairs and
calls BCE-with-logits. `ColBERTTrainer` obtains token embeddings and calls the
late-interaction loss with query and document masks. Loss computation also works
in evaluation mode. No trainer chooses behavior based on dictionary iteration order.

YAML configuration is the supported public training interface. Older Python
model, loss, and trainer APIs have no compatibility guarantee. Legacy trainer facades and entry points have been removed.

## Configuration and data

YAML configurations use canonical schema fields and dotted CLI overrides.
Unknown fields are rejected. Dataset and output paths resolve relative to the
configuration file. There is no Hydra defaults-list composition.

Custom column names are translated at the dataset boundary. Dense and ColBERT
rows contain a query, one or more positives, and optional negatives. Positive and
negative selection is deterministic for evaluation. Reranking uses labeled pairs.
JSON/JSONL, CSV, Parquet, and Hugging Face datasets are supported.

## Model loading and execution

`load_model` loads the tokenizer and backbone on CPU. Trainer owns device
placement. Encoder and decoder dense models use the same encoding path with an
explicit pooling choice. LoRA is optional and loaded only when requested.

Launch multiple processes explicitly with `torchrun --module retrievals.train`.
Negatives are local to each rank in these workflows. Cross-device negative sharing,
Ray, verl, LLaMA-Factory execution, quantized training, and YAML distillation are
not implemented. Unsupported backend choices fail before model loading.

## Evaluation and persistence

`build_trainer(config, training=False)` never reads the training dataset. The
current evaluation reports held-out task loss; it is not corpus-level MRR, NDCG,
or Recall. Existing retrieval metrics and pipelines remain separate.

Trainer writes resumable `checkpoint-*` directories with model, optimizer,
scheduler, and RNG state. Final export goes to `output_dir/model/`, including the
tokenizer and a resolved `config.yaml`. ColBERT exports its projection separately;
dense embeddings also export pooling and normalization metadata. LoRA exports an
adapter and records its base model reference. Reloading the exported config uses
that adapter without wrapping it a second time.

## Verification

`tests/test_trainer/test_training_workflows.py` builds tiny local BERT and decoder
models. It checks real training, evaluation, export/reload, checkpoint resume,
column mapping, positive-only batches, unknown-key rejection, and padding masks without network
access. Extend these tests whenever a new YAML task or loss is added.

# Training examples

Start with `dense_retrieval.yaml`, `cross_encoder.yaml`, or `colbert.yaml`.
These use repository fixture data and Transformers Trainer. `smoke_test.yaml`
limits training to one step. See [the YAML guide](../../YAML_CONFIG_README.md)
for data formats, evaluation, LoRA, exports, and resume.

`distillation.yaml`, `production.yaml`, and `fast_training.yaml` are earlier
experimental configurations. They are not supported recipes; configurations
requesting unimplemented execution features fail validation.

For fully offline verification with tiny local models:

```bash
HF_HUB_OFFLINE=1 python -m unittest tests.test_trainer.test_training_workflows
```

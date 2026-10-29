"""Portable serving exports from saved dense embedding checkpoints."""

import argparse
import json
import re
import shutil
from pathlib import Path

POOLING = {
    "cls": "cls",
    "first": "cls",
    "mean": "mean",
    "avg": "mean",
    "average": "mean",
    "last": "lasttoken",
    "eos": "lasttoken",
}


def export_model(model_path, output_dir, backend="vllm", model_name="embedding"):
    """Copy a complete checkpoint into a new serving artifact; never overwrite it."""
    source = Path(model_path).resolve()
    destination = Path(output_dir).resolve()
    if backend not in {"vllm", "triton"}:
        raise ValueError("backend must be vllm or triton")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", model_name):
        raise ValueError("model_name must contain only letters, digits, underscores or hyphens")
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError("output_dir must be a new directory outside the source checkpoint")
    if (source / "adapter_config.json").exists():
        raise ValueError("Merge LoRA/QLoRA adapters into a full-precision base model before exporting")
    metadata_file = source / "embedding_config.json"
    if not metadata_file.is_file():
        raise ValueError("Expected a saved dense embedding model with embedding_config.json")
    metadata = json.loads(metadata_file.read_text())
    pooling = POOLING.get(metadata.get("pooling_method"))
    if pooling is None:
        raise ValueError("Serving export supports only CLS, mean and last-token dense pooling")
    config = json.loads((source / "config.json").read_text())
    if config.get("quantization_config"):
        raise ValueError("Export an unquantized merged checkpoint; quantize in the serving runtime")
    if not any(
        (source / name).exists()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    ):
        raise ValueError("Checkpoint has no complete Hugging Face model weights")
    if backend == "vllm" and pooling == "cls":
        tokenizer_config = json.loads((source / "tokenizer_config.json").read_text())
        if tokenizer_config.get("padding_side", "right") == "left":
            raise ValueError("CLS serving requires right padding for consistent first-token pooling")
    target = destination if backend == "vllm" else destination / model_name / "1" / "checkpoint"
    shutil.copytree(source, target)
    if backend == "vllm":
        dimension = config.get("hidden_size", config.get("d_model", config.get("n_embd")))
        modules = [
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        ]
        pool_config = {"word_embedding_dimension": dimension}
        for mode in ("cls", "mean", "max", "mean_sqrt_len", "weightedmean", "lasttoken"):
            pool_config[f"pooling_mode_{mode}_tokens" if mode not in ("lasttoken",) else "pooling_mode_lasttoken"] = (
                mode == pooling
            )
        # Sentence Transformers uses these exact names for CLS and weighted mean.
        pool_config["pooling_mode_cls_token"] = pool_config.pop("pooling_mode_cls_tokens")
        (target / "1_Pooling").mkdir(exist_ok=True)
        (target / "1_Pooling" / "config.json").write_text(json.dumps(pool_config, indent=2))
        if metadata.get("normalize_embeddings", False):
            modules.append(
                {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}
            )
            (target / "2_Normalize").mkdir(exist_ok=True)
        (target / "modules.json").write_text(json.dumps(modules, indent=2))
        (target / "sentence_bert_config.json").write_text(
            json.dumps({"max_seq_length": metadata["max_length"], "do_lower_case": False}, indent=2)
        )
    else:
        model_dir = destination / model_name
        shutil.copyfile(Path(__file__).with_name("triton_backend.py"), model_dir / "1" / "model.py")
        (model_dir / "config.pbtxt").write_text(
            f'name: "{model_name}"\nbackend: "python"\nmax_batch_size: 64\n'
            'input [{ name: "TEXT" data_type: TYPE_STRING dims: [ 1 ] }]\n'
            'output [{ name: "EMBEDDING" data_type: TYPE_FP32 dims: [ -1 ] }]\n'
            'dynamic_batching { max_queue_delay_microseconds: 1000 }\n'
            'instance_group [{ kind: KIND_CPU count: 1 }]\n'
        )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Saved dense embedding checkpoint")
    parser.add_argument("--output", required=True, help="New export directory")
    parser.add_argument("--backend", choices=["vllm", "triton"], required=True)
    parser.add_argument("--model-name", default="embedding")
    args = parser.parse_args()
    print(export_model(args.model, args.output, args.backend, args.model_name))


if __name__ == "__main__":
    main()

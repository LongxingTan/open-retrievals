"""Offline serving artifact and Triton numerical parity tests."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from retrievals.cli import main
from retrievals.export import export_model
from retrievals.models import AutoModelForEmbedding


class ServingExportTests(unittest.TestCase):
    def test_exports_and_triton_parity(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocab = root / "vocab.txt"
            vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "cat", "dog"]))
            tokenizer = BertTokenizerFast(vocab_file=str(vocab))
            for pooling in ("cls", "mean", "last"):
                for normalize in (False, True):
                    with self.subTest(pooling=pooling, normalize=normalize):
                        name = f"{pooling}-{normalize}"
                        model = AutoModelForEmbedding(
                            BertModel(
                                BertConfig(
                                    vocab_size=7,
                                    hidden_size=8,
                                    num_hidden_layers=1,
                                    num_attention_heads=2,
                                    intermediate_size=16,
                                )
                            ),
                            tokenizer,
                            pooling_method=pooling,
                            normalize_embeddings=normalize,
                            max_length=16,
                            device="cpu",
                        ).eval()
                        source = root / name
                        model.save_pretrained(source)
                        vllm = root / f"{name}-vllm"
                        main(["export", "--model", str(source), "--output", str(vllm), "--backend", "vllm"])
                        modules = json.loads((vllm / "modules.json").read_text())
                        self.assertEqual(len(modules), 3 if normalize else 2)
                        pool = json.loads((vllm / "1_Pooling/config.json").read_text())
                        key = {"cls": "cls_token", "mean": "mean_tokens", "last": "lasttoken"}[pooling]
                        self.assertTrue(pool[f"pooling_mode_{key}"])
                        destination = export_model(source, root / f"{name}-triton", "triton")
                        backend_file = destination / "embedding/1/model.py"
                        fake = types.SimpleNamespace(
                            get_input_tensor_by_name=lambda request, _: types.SimpleNamespace(as_numpy=lambda: request),
                            Tensor=lambda name, value: value,
                            InferenceResponse=lambda **kwargs: kwargs,
                            TritonError=str,
                        )
                        with patch.dict(sys.modules, {"triton_python_backend_utils": fake}):
                            spec = importlib.util.spec_from_file_location("exported_backend", backend_file)
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            backend = module.TritonPythonModel()
                            backend.initialize({"model_instance_kind": "CPU"})
                            texts = ["cat", "cat dog"]
                            tokens = tokenizer(texts, padding=True, return_tensors="pt")
                            with torch.inference_mode():
                                expected = model.forward_from_tensor(tokens["input_ids"], tokens["attention_mask"])
                            responses = backend.execute(
                                [np.array([[x.encode()] for x in texts], dtype=object), np.array([], dtype=object)]
                            )
                            np.testing.assert_allclose(
                                responses[0]["output_tensors"][0], expected.numpy(), rtol=1e-5, atol=1e-6
                            )
                            self.assertIn("error", responses[1])

    def test_rejects_unsafe_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            for metadata in ({}, {"pooling_method": "weighted"}):
                (source / "embedding_config.json").write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):
                    export_model(source, Path(directory) / "output")
            (source / "adapter_config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Merge"):
                export_model(source, Path(directory) / "output")
            with self.assertRaises(ValueError):
                export_model(source, source / "nested")
            with self.assertRaises(ValueError):
                export_model(source, Path(directory) / "output", model_name="../bad")

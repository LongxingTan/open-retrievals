"""Exercise generated serving batching without a Triton installation."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class TritonBatchingTest(unittest.TestCase):
    def test_coalesces_requests_and_isolates_invalid_shapes(self):
        backend = SimpleNamespace(
            get_input_tensor_by_name=lambda request, name: SimpleNamespace(as_numpy=lambda: request),
            Tensor=lambda name, value: value,
            InferenceResponse=lambda **kwargs: kwargs,
            TritonError=str,
        )
        path = Path(__file__).parents[2] / "src/retrievals/triton_backend.py"
        spec = importlib.util.spec_from_file_location("standalone_triton_backend", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"triton_python_backend_utils": backend}):
            spec.loader.exec_module(module)
        model = module.TritonPythonModel()
        calls = []

        def embed(texts):
            calls.append(texts)
            return np.arange(len(texts) * 2, dtype=np.float32).reshape(-1, 2)

        model.embed = embed
        responses = model.execute(
            [
                np.array([[b"one"], [b"two"]]),
                np.array([b"invalid"]),
                np.array([[b"three"]]),
            ]
        )
        self.assertEqual(calls, [["one", "two", "three"]])
        self.assertEqual(responses[0]["output_tensors"][0].shape, (2, 2))
        self.assertIn("error", responses[1])
        np.testing.assert_array_equal(responses[2]["output_tensors"][0], [[4, 5]])

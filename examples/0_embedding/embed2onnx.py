"""
transfer embedding model to onnx by optimum: https://github.com/huggingface/optimum
"""

import argparse

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

from retrievals import AutoModelForEmbedding


def convert_to_onnx(model_name_or_path, output_path, opset=11):
    # model = AutoModelForEmbedding.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    onnx_model = ORTModelForFeatureExtraction.from_pretrained(model_name_or_path)
    onnx_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model successfully converted to ONNX and saved at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert a embedding model to ONNX format.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the embedding model to convert.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the ONNX model.")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version to use for conversion.")

    args = parser.parse_args()

    convert_to_onnx(args.model_name, args.output_path, args.opset)


if __name__ == "__main__":
    main()

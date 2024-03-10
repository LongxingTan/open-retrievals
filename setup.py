import os
import re
import shutil
from pathlib import Path

from setuptools import Command, find_packages, setup

version_file = 'src/retrievals/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


extras = {}

_deps = [
    "Pillow>=10.0.1,<=15.0",
    "accelerate>=0.21.0",
    "peft",
    "av==9.2.0",  # Latest version of PyAV (10.0.0) has issues with audio stream.
    "beautifulsoup4",
    "codecarbon==1.2.0",
    "cookiecutter==1.7.3",
    "dataclasses",
    "datasets!=2.5.0",
    "decord==0.6.0",
    "deepspeed>=0.9.3",
    "diffusers",
    "dill<0.3.5",
    "evaluate>=0.2.0",
    "faiss-cpu",
    "fastapi",
    "filelock",
    "flax>=0.4.1,<=0.7.0",
    "fsspec<2023.10.0",
    "ftfy",
    "fugashi>=1.0",
    "GitPython<3.1.19",
    "hf-doc-builder>=0.3.0",
    "huggingface-hub>=0.19.3,<1.0",
    "importlib_metadata",
    "ipadic>=1.0.0,<2.0",
    "isort>=5.5.4",
    "jax>=0.4.1,<=0.4.13",
    "jaxlib>=0.4.1,<=0.4.13",
    "kenlm",
    "transformers",
    "nltk",
    "natten>=0.14.6,<0.15.0",
    "numpy>=1.17",
    "onnxconverter-common",
    "onnxruntime-tools>=1.4.2",
    "onnxruntime>=1.4.0",
    "optuna",
    "optax>=0.0.8,<=0.1.4",
    "packaging>=20.0",
    "parameterized",
    "phonemizer",
    "protobuf",
    "psutil",
    "pyyaml>=5.1",
    "pydantic",
    "pytest>=7.2.0,<8.0.0",
    "pytest-timeout",
    "pytest-xdist",
    "python>=3.8.0",
    "regex!=2019.12.17",
    "requests",
    "rhoknp>=1.1.0,<1.3.1",
    "rjieba",
    "rouge-score!=0.0.7,!=0.0.8,!=0.1,!=0.1.1",
    "ruff==0.1.5",
    "sacrebleu>=1.4.12,<2.0.0",
    "sacremoses",
    "safetensors>=0.4.1",
    "sagemaker>=2.31.0",
    "scikit-learn",
    "sentencepiece>=0.1.91,!=0.1.92",
    "sigopt",
    "starlette",
    "sudachipy>=0.6.6",
    "sudachidict_core>=20220729",
    "timeout-decorator",
    "timm",
    "tokenizers>=0.14,<0.19",
    "torch",
    "pyctcdecode>=0.4.0",
    "tqdm>=4.27",
    "unidic>=1.0.2",
    "unidic_lite>=1.0.7",
    "urllib3<2.0.0",
    "uvicorn",
]


# this is a lookup table with items like:
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ ]+)(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)}

install_requires = [
    deps["filelock"],  # filesystem locks, e.g., to prevent parallel downloads
    deps["huggingface-hub"],
    deps["numpy"],
    deps["packaging"],  # utilities from PyPA to e.g., compare versions
    deps["pyyaml"],  # used for the model cards metadata
    deps["regex"],  # for OpenAI GPT
    deps["requests"],  # for downloading models over HTTPS
    deps["tokenizers"],
    deps["safetensors"],
    deps["tqdm"],  # progress bars in model download and training scripts
]

setup(
    name="open-retrievals",
    version=get_version(),
    author="Longxing Tan",
    author_email="tanlongxing888@163.com",
    description="Text Embeddings for Retrieval and RAG based on transformers",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP retrieval RAG rerank text embedding contrastive",
    license="Apache 2.0 License",
    url="https://github.com/LongxingTan/open-retrievals",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    package_data={"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx"]},
    zip_safe=False,
    extras_require=extras,
    python_requires=">=3.8.0",
    install_requires=list(install_requires),
    classifiers=[
        # "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

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
    "Pillow>=10.0.1",
    "accelerate>=0.21.0",
    "peft",
    "dataclasses",
    "decord==0.6.0",
    "deepspeed>=0.9.3",
    "datasets",
    "importlib_metadata",
    "transformers",
    "numpy>=1.17",
    "protobuf",
    "pyyaml>=5.1",
    "python>=3.8.0",
    "requests",
    "sentencepiece>=0.1.91,!=0.1.92",
    "tokenizers>=0.14",
    "torch",
    "tqdm>=4.27",
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

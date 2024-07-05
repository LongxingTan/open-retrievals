from setuptools import Command, find_packages, setup

version_file = 'src/retrievals/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


base_packages = [
    "torch>=1.6.0",
    "transformers >= 4.0.0",
    "tokenizers>=0.14",
    "datasets>=1.1.3",
    "tqdm >= 4.66",
]


eval = ["beir >= 2.0.0", "mteb", "C-MTEB>=1.1.0"]


dev = [
    "mkdocs-material == 9.2.8",
    "mkdocs-awesome-pages-plugin == 2.9.2",
    "mkdocs-jupyter == 0.24.7",
    "faiss-gpu",
    "peft",
    "accelerate>=0.20.0",
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
    python_requires=">=3.7.0",
    install_requires=base_packages,
    extras_require={"eval": base_packages + eval, "dev": base_packages + eval + dev},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
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

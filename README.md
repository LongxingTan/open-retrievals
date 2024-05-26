[license-image]: https://img.shields.io/badge/License-Apache%202.0-blue.svg
[license-url]: https://opensource.org/licenses/Apache-2.0
[pypi-image]: https://badge.fury.io/py/open-retrievals.svg
[pypi-url]: https://pypi.org/project/open-retrievals
[pepy-image]: https://pepy.tech/badge/retrievals/month
[pepy-url]: https://pepy.tech/project/retrievals
[build-image]: https://github.com/LongxingTan/open-retrievals/actions/workflows/test.yml/badge.svg?branch=master
[build-url]: https://github.com/LongxingTan/open-retrievals/actions/workflows/test.yml?query=branch%3Amaster
[lint-image]: https://github.com/LongxingTan/open-retrievals/actions/workflows/lint.yml/badge.svg?branch=master
[lint-url]: https://github.com/LongxingTan/open-retrievals/actions/workflows/lint.yml?query=branch%3Amaster
[docs-image]: https://readthedocs.org/projects/open-retrievals/badge/?version=latest
[docs-url]: https://open-retrievals.readthedocs.io/en/latest/?version=latest
[coverage-image]: https://codecov.io/gh/longxingtan/open-retrievals/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/longxingtan/open-retrievals?branch=master

<h1 align="center">
<img src="./docs/source/_static/logo.svg" width="520" align=center/>
</h1><br>

[![LICENSE][license-image]][license-url]
[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Lint Status][lint-image]][lint-url]
[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]


**[Documentation](https://open-retrievals.readthedocs.io)** | **[中文](https://github.com/LongxingTan/open-retrievals/blob/master/README_zh-CN.md)** | **[日本語](https://github.com/LongxingTan/open-retrievals/blob/master/README_ja-JP.md)**

**Open-retrievals** simplifies text embeddings, retrievals, ranking, and RAG applications using PyTorch and Transformers. This user-friendly framework is designed for information retrieval and LLM-enhanced generation.
- Contrastive learning enhanced embeddings/ LLM embeddings
- Cross-encoder and ColBERT Rerank
- Fast RAG demo integrated with Langchain and LlamaIndex


## Installation

**Prerequisites**
```shell
pip install transformers
pip install faiss-cpu  # if necessary
pip install peft  # if necessary
```

**With pip**
```shell
pip install open-retrievals
```

[//]: # (**With conda**)

[//]: # (```shell)

[//]: # (conda install open-retrievals -c conda-forge)

[//]: # (```)


## Quick-start

**Text embedding from Pretrained weights**
```python
from retrievals import AutoModelForEmbedding

sentences = ["Hello NLP", "Open-retrievals is designed for retrieval, rerank and RAG"]
model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModelForEmbedding(model_name_or_path, pooling_method="mean")
sentence_embeddings = model.encode(sentences, normalize_embeddings=True, convert_to_tensor=True)
print(sentence_embeddings)
```

**Index building for dense retrieval search**
```python
from retrievals import AutoModelForEmbedding, AutoModelForRetrieval

sentences = ['A dog is chasing car.', 'A man is playing a guitar.']
model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
index_path = './database/faiss/faiss.index'
model = AutoModelForEmbedding(model_name_or_path)
model.build_index(sentences, index_path=index_path)

query_embed = model.encode("He plays guitar.")
matcher = AutoModelForRetrieval()
dists, indices = matcher.similarity_search(query_embed, index_path=index_path)
print(indices)
```

**Rerank using pretrained weights**
```python
from retrievals import RerankModel

model_name_or_path: str = "microsoft/mdeberta-v3-base"
rerank_model = RerankModel.from_pretrained(model_name_or_path)
rerank_model.eval()
rerank_model.to("cuda")
rerank_model.compute_score(["In 1974, I won the championship in Southeast Asia in my first kickboxing match," "In 1982, I defeated the heavy hitter Ryu Long."])
```

**RAG with LangChain integration**
```shell
pip install langchain
pip install langchain_community
pip install chromadb
```

```python
from retrievals.tools.langchain import LangchainEmbedding, LangchainReranker
from retrievals import RerankModel
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import Chroma as Vectorstore

persist_directory = './database/faiss/faiss.index'
embeddings = LangchainEmbedding(model_name_or_path="BAAI/bge-large-zh-v1.5")
vectordb = Vectorstore(
    persist_directory=persist_directory,
    embedding_function=embeddings,
)
retrieval_args = {"search_type" :"similarity", "score_threshold": 0.15, "k": 30}
retriever = vectordb.as_retriever(retrieval_args)

ranker = RerankModel.from_pretrained("maidalun1020/bce-reranker-base_v1")
reranker = LangchainReranker(model=ranker, top_n=7)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=retriever
)

query = 'what is open-retrievals?'
docs = compression_retriever.invoke(query)
```

[//]: # (**RAG with LLamaIndex**)

[//]: # ()
[//]: # (```shell)

[//]: # (pip install llamaindex)

[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # (```python)

[//]: # ()
[//]: # ()
[//]: # (```)

**Text embedding model fine-tuned by contrastive learning**
```python
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, TrainingArguments
from retrievals import AutoModelForEmbedding, RetrievalTrainer, PairCollator, TripletCollator
from retrievals.losses import ArcFaceAdaptiveMarginLoss, InfoNCE, SimCSE, TripletLoss

model_name_or_path: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
batch_size: int = 128
epochs: int = 3

train_dataset = load_dataset('shibing624/nli_zh', 'STS-B')['train']
train_dataset = train_dataset.rename_columns({'sentence1': 'query', 'sentence2': 'positive'})
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
model = AutoModelForEmbedding(model_name_or_path, pooling_method="cls")
optimizer = AdamW(model.parameters(), lr=5e-5)
num_train_steps=int(len(train_dataset) / batch_size * epochs)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_steps, num_training_steps=num_train_steps)

training_arguments = TrainingArguments(
    output_dir='./',
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    remove_unused_columns=False,
)
trainer = RetrievalTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    data_collator=PairCollator(tokenizer, max_length=512),
    loss_fn=InfoNCE(nn.CrossEntropyLoss(label_smoothing=0.05)),
)
trainer.optimizer = optimizer
trainer.scheduler = scheduler
trainer.train()
```

**Finetuning for rerank models**
```python
from torch.optim import AdamW
from transformers import AutoTokenizer, TrainingArguments, get_cosine_schedule_with_warmup
from retrievals import RerankCollator, RerankModel, RerankTrainer, RerankDataset

model_name_or_path: str = "microsoft/mdeberta-v3-base"
query_max_length: int = 512
learning_rate: float = 3e-5
batch_size: int = 64
epochs: int = 3

train_dataset = RerankDataset(args=data_args)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
model = RerankModel(model_name_or_path, pooling_method="mean")
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_train_steps = int(len(train_dataset) / batch_size * epochs)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_steps, num_training_steps=num_train_steps)

training_args = TrainingArguments(
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs,
    output_dir = './checkpoints',
)
trainer = RerankTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=RerankCollator(tokenizer, max_length=query_max_length),
)
trainer.optimizer = optimizer
trainer.scheduler = scheduler
trainer.train()
```

**Semantic search by cosine similarity/KNN**
```python
from retrievals import AutoModelForEmbedding, AutoModelForRetrieval

query_texts = ['A dog is chasing car.']
document_texts = ['A man is playing a guitar.', 'A bee is flying low']
model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModelForEmbedding(model_name_or_path)
query_embeddings = model.encode(query_texts, convert_to_tensor=True)
document_embeddings = model.encode(document_texts, convert_to_tensor=True)

matcher = AutoModelForRetrieval(method='cosine')
dists, indices = matcher.similarity_search(query_embeddings, document_embeddings, top_k=1)
```


## Reference & Acknowledge
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- [uniem](https://github.com/wangyuxinwhy/uniem)
- [BCEmbedding](https://github.com/netease-youdao/BCEmbedding)

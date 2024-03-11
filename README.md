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


**[Documentation](https://open-retrievals.readthedocs.io)** | **[Tutorials](https://open-retrievals.readthedocs.io/en/latest/tutorials.html)** | **[中文](https://github.com/LongxingTan/open-retrievals/blob/master/README_zh-CN.md)**

**Open-Retrievals** is an easy-to-use python framework getting SOTA text embeddings, oriented to information retrieval and LLM retrieval augmented generation, based on PyTorch and Transformers.
- Contrastive learning enhanced embeddings
- LLM embeddings


## Installation
**Prerequisites**
```shell
pip install transformers
pip install faiss
pip install peft
```

**With pip**
```shell
pip install open-retrievals
```

[//]: # (**With conda**)

[//]: # (```shell)

[//]: # (conda install open-retrievals -c conda-forge)

[//]: # (```)


## Usage


**Use Pretrained sentence embedding**
```python
from retrievals import AutoModelForEmbedding

sentences = ["Hello world", "How are you?"]
model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModelForEmbedding(model_name_or_path, pooling_method="cls")
sentence_embeddings = model.encode(sentences)
print(sentence_embeddings)
```


**Finetune transformers by contrastive learning**
```python
from dataclasses import dataclass, field
import transformers
from transformers import AutoTokenizer
from retrievals import AutoModelForEmbedding, AutoModelForMatch, RetrievalTrainer, PairCollator, TripletCollator
from retrievals.losses import ArcFaceAdaptiveMarginLoss, InfoNCE, SimCSE, TripletLoss
from retrievals.data import  RetrievalDataset, RerankDataset

tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, use_fast=False)
train_dataset = RetrievalDataset(args=data_args)

model = AutoModelForEmbedding(
    model_args.model_name_or_path,
    pooling_method="cls"
)
optimizer = get_optimizer(model, lr=5e-5, weight_decay=1e-3)

lr_scheduler = get_scheduler(optimizer, num_train_steps=int(len(train_dataset) / 2 * 1))

trainer = RetrievalTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=TripletCollator(tokenizer, max_length=data_args.query_max_len),
    loss_fn=TripletLoss(),
)
trainer.optimizer = optimizer
trainer.scheduler = lr_scheduler
trainer.train()
```

**Finetune LLM for embedding by Contrastive learning**
```python
# just change the model in AutoModelForEmbedding
from retrievals import AutoModelForEmbedding

model = AutoModelForEmbedding('llama', pooling_method='last', query_instruction='')
```

**Search by KNN**
```python
from retrievals import AutoModelForEmbedding, AutoModelForMatch

query_texts = []
passage_texts = []
model = AutoModelForEmbedding('')
query_embeddings = model.encode(query_texts, convert_to_tensor=True)
passage_embeddings = model.encode(passage_texts, convert_to_tensor=True)

matcher = AutoModelForMatch(method='cosine')
dists, indices = matcher.similarity_search(query_embeddings, passage_embeddings, top_k=1)

```

**Search by Faiss**
```python

```


**Rerank**
```python

```

[//]: # (**RAG with LangChain**)

[//]: # ()
[//]: # (- Prerequisites)

[//]: # (```shell)

[//]: # (pip install langchain)

[//]: # (```)

[//]: # ()
[//]: # (- Server)

[//]: # (```python)

[//]: # ()
[//]: # (```)

[//]: # (**RAG with LLamaIndex**)

[//]: # (```shell)

[//]: # (pip install llamaindex)

[//]: # (```)

[//]: # ()
[//]: # (```python)

[//]: # ()
[//]: # (```)


## Reference & Acknowledge
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- [uniem](https://github.com/wangyuxinwhy/uniem)
- [BCEmbedding](https://github.com/netease-youdao/BCEmbedding)

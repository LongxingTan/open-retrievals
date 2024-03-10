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

**With conda**
```shell
conda install open-retrievals -c conda-forge
```


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
from retrievals import AutoModelForEmbedding, AutoModelForMatch, RetrievalTrainer, PairCollator
from retrievals.losses import ArcFaceAdaptiveMarginLoss, InfoNCE, SimCSE, TripletLoss
from retrievals.data import  RetrievalDataset, RerankDataset


train_dataset = RetrievalDataset(topic_df, tokenizer, CFG.max_len, aug=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=CFG.num_workers,
    pin_memory=False,
    drop_last=True,
)

loss_fn = ArcFaceAdaptiveMarginLoss(
    criterion=cross_entropy,
    in_features=768,
    out_features=CFG.num_classes,
    scale=CFG.arcface_scale,
    margin=CFG.arcface_margin,
)
model = AutoModelForEmbedding(CFG.MODEL_NAME, pooling_method="cls", loss_fn=loss_fn)

optimizer = get_optimizer(model, lr=CFG.learning_rate)
scheduler = get_scheduler(
    optimizer=optimizer, cfg=CFG, total_steps=len(train_dataset)
)
trainer = CustomTrainer(model, device="cuda", apex=CFG.apex)
trainer.train(
    train_loader=train_loader,
    criterion=None,
    optimizer=optimizer,
    epochs=CFG.epochs,
    scheduler=scheduler,
    dynamic_margin=True,
)
torch.save(model.state_dict(), CFG.output_dir + f"model_{CFG.exp_id}.pth")
```

**Finetune LLM for embedding by Contrastive learning**
```python
from retrievals import AutoModelForEmbedding

model = AutoModelForEmbedding('llama', pooling_method='last', query_instruction='')
```

**Search by KNN**
```python
from retrievals import AutoModelForEmbedding

retrieval_model = AutoModelForEmbedding('')
retrieval_model.query(method='knn')

```

**Search by Faiss**
```python

```


**Rerank**
```python

```

**RAG with LangChain**

- Prerequisites
```shell
pip install langchain
```

- Server
```python

```

**RAG with LLamaIndex**
```shell
pip install llamaindex
```

```python

```


## Reference & Acknowledge
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- [uniem](https://github.com/wangyuxinwhy/uniem)
- [BCEmbedding](https://github.com/netease-youdao/BCEmbedding)

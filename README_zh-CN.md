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
<img src="./docs/source/_static/logo.svg" width="490" align=center/>
</h1><br>

[![LICENSE][license-image]][license-url]
[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Lint Status][lint-image]][lint-url]
[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]


**[文档](https://open-retrievals.readthedocs.io)** | **[Tutorials](https://open-retrievals.readthedocs.io/en/latest/tutorials.html)** | **[Release Notes](https://open-retrievals.readthedocs.io/en/latest/CHANGELOG.html)** | **[中文](https://github.com/LongxingTan/open-retrievals/blob/master/README_zh-CN.md)**

**Open-Retrievals** 帮助开发者在信息检索、大语言模型等领域便捷地应用与增强文本向量，基于Pytorch、Transformers框架。
- 对比学习增强性能
- 支持大语言模型文本向量


## 安装

**基础**
```shell
pip install transformers
pip install faiss
pip install peft
```

**安装**
```shell
pip install open-retrievals
```


## 快速使用

**预训练文本向量**
```python
from retrievals import AutoModelForEmbedding

sentences = ["Hello world", "How are you?"]
model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModelForEmbedding(model_name_or_path, pooling_method="mean", normalize_embeddings=True)
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
print(sentence_embeddings)
```

**基于余弦相似度和紧邻搜索**
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

**Faiss向量数据库检索**
```python
from retrievals import AutoModelForEmbedding, AutoModelForMatch

sentences = ['A woman is reading.', 'A man is playing a guitar.']
model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModelForEmbedding(model_name_or_path)
model.build_index(sentences)

matcher = AutoModelForMatch()
results = matcher.faiss_search("He plays guitar.")
```

**重排**
```python
from transformers import AutoTokenizer
from retrievals import RerankCollator, RerankModel, RerankTrainer, RerankDataset

train_dataset = RerankDataset(args=data_args)
tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, use_fast=False)

model = RerankModel(
    model_args.model_name_or_path,
    pooling_method="mean"
)
optimizer = get_optimizer(model, lr=5e-5, weight_decay=1e-3)

lr_scheduler = get_scheduler(optimizer, num_train_steps=int(len(train_dataset) / 2 * 1))

trainer = RerankTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=RerankCollator(tokenizer, max_length=data_args.query_max_len),
)
trainer.optimizer = optimizer
trainer.scheduler = lr_scheduler
trainer.train()
```


## 参考与致谢
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- [uniem](https://github.com/wangyuxinwhy/uniem)
- [BCEmbedding](https://github.com/netease-youdao/BCEmbedding)

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


**[中文wiki](https://github.com/LongxingTan/open-retrievals/wiki)** | **[英文文档](https://open-retrievals.readthedocs.io)** | **[Release Notes](https://open-retrievals.readthedocs.io/en/latest/CHANGELOG.html)**

**Open-Retrievals** 帮助开发者在信息检索、大语言模型等领域便捷地应用文本向量，快速搭建检索、排序、RAG等应用。
- 支持point-wise、pairwise、listwise训练
- 多种对比学习进行文本向量微调、rerank微调
- 支持大语言模型LLM文本向量微调
- 结合Langchain、LLamaIndex快速产出RAG demo


## 安装

**基础**
```shell
pip install transformers
pip install faiss  # 如有必要
pip install peft  # 如有必要
```

**pip安装**
```shell
pip install open-retrievals
```

**源码安装**
```shell
git clone https://github.com/LongxingTan/open-retrievals
cd open-retrievals
pip install -e .
```


## 快速入门

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-WBMisdWLeHUKlzJ2DrREXY_kSV8vjP3?usp=sharing)

**使用预训练权重的文本向量**
```python
from retrievals import AutoModelForEmbedding, AutoModelForRetrieval

# 文本示例
sentences = [
    "在1974年，第一次在东南亚打自由搏击就得了冠军",
    "1982年打赢了日本重炮手雷龙，接着连续三年打败所有日本空手道高手，赢得全日本自由搏击冠军",
    "中国古拳法唯一传人鬼王达，被喻为空手道的克星，绰号“魔鬼筋肉人”"
]

# 向量模型
model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModelForEmbedding(model_name_or_path)

embeddings = model.encode(sentences)
print(embeddings) # 384维度的文本向量
```

**使用Faiss向量数据库检索**
```python
from retrievals import AutoModelForEmbedding, AutoModelForRetrieval

index_path = './database/faiss/faiss.index'
sentences = ['A dog is chasing car.', 'A man is playing a guitar.']
model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModelForEmbedding(model_name_or_path)
model.build_index(sentences, index_path=index_path)

query_embed = model.encode("He plays guitar.")
matcher = AutoModelForRetrieval()
dists, indices = matcher.similarity_search(query_embed, index_path=index_path)
print(indices)
```

**重排**
```python
from retrievals import RerankModel

model_name_or_path: str = "BAAI/bge-reranker-base"
rerank_model = RerankModel.from_pretrained(model_name_or_path)
scores_list = rerank_model.compute_score(
    [["在1974年，第一次在东南亚打自由搏击就得了冠军", "1982年打赢了日本重炮手雷龙"],
     ["铁砂掌，源于泗水铁掌帮，三日练成，收费六百", "铁布衫，源于福建省以北70公里，五日练成，收费八百"]]
)
print(scores_list)
```

**Langchain RAG应用**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fJC-8er-a4NRkdJkwWr4On7lGt9rAO4P?usp=sharing)

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

query = '1974年，谁获得了东南亚自由搏击的冠军？'
docs = compression_retriever.invoke(query)
```

**微调文本向量模型**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17KXe2lnNRID-HiVvMtzQnONiO74oGs91?usp=sharing)

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
# model.set_train_type('pointwise')  # 'pointwise', 'pairwise', 'listwise'
optimizer = AdamW(model.parameters(), lr=5e-5)
num_train_steps=int(len(train_dataset) / batch_size * epochs)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_steps, num_training_steps=num_train_steps)

training_arguments = TrainingArguments(
    output_dir='./checkpoints',
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

**基于余弦相似度和近邻搜索**
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

**微调重排模型**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QvbUkZtG56SXomGYidwI4RQzwODQrWNm?usp=sharing)

```python
from transformers import AutoTokenizer, TrainingArguments, get_cosine_schedule_with_warmup, AdamW
from retrievals import RerankCollator, RerankModel, RerankTrainer, RerankDataset

model_name_or_path: str = "microsoft/deberta-v3-base"
max_length: int = 128
learning_rate: float = 3e-5
batch_size: int = 4
epochs: int = 3

train_dataset = RerankDataset('./t2rank.json', positive_key='pos', negative_key='neg')
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
model = RerankModel.from_pretrained(model_name_or_path, pooling_method="mean")
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_train_steps = int(len(train_dataset) / batch_size * epochs)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_steps, num_training_steps=num_train_steps)

training_args = TrainingArguments(
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs,
    output_dir = './checkpoints',
    remove_unused_columns=False,
)
trainer = RerankTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=RerankCollator(tokenizer, max_length=max_length),
)
trainer.optimizer = optimizer
trainer.scheduler = scheduler
trainer.train()
```


## 参考与致谢
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- [uniem](https://github.com/wangyuxinwhy/uniem)
- [BCEmbedding](https://github.com/netease-youdao/BCEmbedding)

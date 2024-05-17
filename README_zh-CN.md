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
- 基于Pytorch、Transformers框架
- 支持point-wise、pairwise、listwise训练
- 多种对比学习进行文本向量微调、rerank微调
- 支持大语言模型LLM文本向量微调
- 支持Langchain、LLamaIndex快速产出RAG demo


## 安装

**基础**
```shell
pip install transformers
pip install faiss  # 如有必要
pip install peft  # 如有必要
```

**安装**
```shell
pip install open-retrievals
```


## 快速入门

**使用预训练权重**

```python
from retrievals import AutoModelForEmbedding, AutoModelForRetrieval

# 文本示例
sentences = [
    "你好，世界",
    "你吃饭了吗?",
    "Open-retrievals 是一个用于检索生成的文本向量库"
]

# 向量模型
model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModelForEmbedding(model_name_or_path)

embeddings = model.encode(sentences)
print(embeddings) # 384维度的文本向量
```


## 使用

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
from retrievals import AutoModelForEmbedding, AutoModelForRetrieval

query_texts = []
document_texts = []
model = AutoModelForEmbedding('')
query_embeddings = model.encode(query_texts, convert_to_tensor=True)
document_embeddings = model.encode(document_texts, convert_to_tensor=True)

matcher = AutoModelForRetrieval(method='cosine')
dists, indices = matcher.similarity_search(query_embeddings, document_embeddings, top_k=1)
```

**Faiss向量数据库检索**
```python
from retrievals import AutoModelForEmbedding, AutoModelForRetrieval

sentences = ['A woman is reading.', 'A man is playing a guitar.']
model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModelForEmbedding(model_name_or_path)
model.build_index(sentences)

matcher = AutoModelForRetrieval()
results = matcher.faiss_search("He plays guitar.")
```

**重排**
```python
from torch.optim import AdamW
from transformers import AutoTokenizer, TrainingArguments, get_cosine_schedule_with_warmup
from retrievals import RerankCollator, RerankModel, RerankTrainer, RerankDataset

model_name_or_path: str = "microsoft/mdeberta-v3-base"
learning_rate: float = 3e-5
batch_size: int = 64
epochs: int = 3

train_dataset = RerankDataset(args=data_args)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

model = RerankModel(model_name_or_path, pooling_method="mean")
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_train_steps = int(len(train_dataset) / batch_size * epochs)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_train_steps)

training_args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    num_train_epochs=2,
    output_dir = './checkpoints',
)

trainer = RerankTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=RerankCollator(tokenizer, max_length=data_args.query_max_length),
)
trainer.optimizer = optimizer
trainer.scheduler = scheduler
trainer.train()
trainer.save_model('weights')
```

**LangChain RAG**
```python
from retrievals.tools.langchain import LangchainEmbedding, LangchainReranker
from retrievals import RerankModel
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import Chroma as Vectorstore


persist_directory = './database/faiss.index'
embeddings = LangchainEmbedding(model_name="BAAI/bge-large-zh-v1.5")
vectordb = Vectorstore(
    persist_directory=persist_directory,
    embedding_function=embeddings,
)
retrieval_args = {"search_type" :"similarity", "score_threshold": 0.15, "k": 30}
retriever = vectordb.as_retriever(retrieval_args)

rank = RerankModel("maidalun1020/bce-reranker-base_v1", use_fp16=True)
reranker = LangchainReranker(model=rank, top_n=7)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=retriever
)

query = 'what is open-retrievals?'
docs = compression_retriever.get_relevant_documents(query)
```


## 参考与致谢

- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- [uniem](https://github.com/wangyuxinwhy/uniem)
- [BCEmbedding](https://github.com/netease-youdao/BCEmbedding)

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


**[Documentation](https://open-retrievals.readthedocs.io)** | **[中文](https://github.com/LongxingTan/open-retrievals/blob/master/README_zh-CN.md)**

**Open-Retrievals** is an easy-to-use python framework getting SOTA text embeddings, oriented to information retrieval and LLM retrieval augmented generation, based on PyTorch and Transformers.
- Contrastive learning enhanced embeddings
- LLM embeddings
- fast RAG demo


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

**Use Pretrained weights**
```python
from retrievals import AutoModelForEmbedding

sentences = ["Hello world", "How are you doing?", "Open-retrievals is a text embedding libraries for RAG application"]
model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModelForEmbedding(model_name_or_path, pooling_method="mean", normalize_embeddings=True)
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
print(sentence_embeddings)
```


## Usage

**Build Index and Retrieval**
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


**Rerank**
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


**RAG with LangChain**

```shell
pip install langchain
```

- Server

```python
from retrievals.tools.langchain import LangchainEmbedding, LangchainReranker
from retrievals import RerankModel
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import Chroma as Vectorstore


class DenseRetrieval:
    def __init__(self, persist_directory):
        embeddings = LangchainEmbedding(model_name="BAAI/bge-large-zh-v1.5")
        vectordb = Vectorstore(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        retrieval_args = {"search_type" :"similarity", "score_threshold": 0.15, "k": 30}
        self.retriever = vectordb.as_retriever(retrieval_args)

        ranker = RerankModel("maidalun1020/bce-reranker-base_v1", use_fp16=True)
        self.reranker = LangchainReranker(model=ranker, top_n=7)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker, base_retriever=self.retriever
        )

    def query(
        self,
        question: str
    ):
        docs = self.compression_retriever.get_relevant_documents(question)
        return docs
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


**Finetune transformers weights by contrastive learning**
```python
from transformers import AutoTokenizer
from retrievals import AutoModelForEmbedding, AutoModelForRetrieval, RetrievalTrainer, PairCollator, TripletCollator
from retrievals.losses import ArcFaceAdaptiveMarginLoss, InfoNCE, SimCSE, TripletLoss
from retrievals.data import  RetrievalDataset, RerankDataset

model_name_or_path = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

train_dataset = RetrievalDataset(args=data_args)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

model = AutoModelForEmbedding(model_name_or_path, pooling_method="cls")
optimizer = get_optimizer(model, lr=5e-5, weight_decay=1e-3)

lr_scheduler = get_scheduler(optimizer, num_train_steps=int(len(train_dataset) / 2 * 1))

trainer = RetrievalTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=TripletCollator(tokenizer, max_length=data_args.query_max_length),
    loss_fn=TripletLoss(),
)
trainer.optimizer = optimizer
trainer.scheduler = lr_scheduler
trainer.train()
```

**Finetune LLM for embedding by Contrastive learning**
```python
from retrievals import AutoModelForEmbedding

model = AutoModelForEmbedding(
    "mistralai/Mistral-7B-v0.1",
    pooling_method='cls',
    query_instruction=f'Instruct: Retrieve semantically similar text\nQuery: '
)
```

**Search by Cosine similarity/KNN**
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

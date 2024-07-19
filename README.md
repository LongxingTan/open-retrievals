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
[contributing-image]: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]: https://github.com/longxingtan/open-retrievals/blob/master/CONTRIBUTING.md

<h1 align="center">
<img src="./docs/source/_static/logo.svg" width="420" align=center/>
</h1>

<div align="center">

  [![LICENSE][license-image]][license-url]
  [![PyPI Version][pypi-image]][pypi-url]
  [![Build Status][build-image]][build-url]
  [![Lint Status][lint-image]][lint-url]
  [![Docs Status][docs-image]][docs-url]
  [![Code Coverage][coverage-image]][coverage-url]
  [![Contributing][contributing-image]][contributing-url]

  **[Documentation](https://open-retrievals.readthedocs.io)** | **[中文](https://github.com/LongxingTan/open-retrievals/blob/master/README_zh-CN.md)** | **[日本語](https://github.com/LongxingTan/open-retrievals/blob/master/README_ja-JP.md)**

</div>

![structure](./docs/source/_static/structure.png)

**Open-retrievals** improve and unify text embedding, retrieval, reranking and RAG.
- Embedding fine-tuned through point-wise, pairwise, listwise, contrastive learning, and LLM.
- Reranking fine-tuned with Cross Encoder, ColBERT, and LLM.
- Easily build enhanced modular RAG, integrated with Transformers, Langchain, and LlamaIndex.

| Exp                           | Model                   | Original | Finetuned | Demo                                                                                                                                                                |
|-------------------------------|-------------------------|----------|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **embed** pairwise finetune   | bge-base-zh-v1.5        | 0.657    | **0.703** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17KXe2lnNRID-HiVvMtzQnONiO74oGs91?usp=sharing) |
| **embed** LLM finetune (LoRA) | Qwen2-1.5B-Instruct     | 0.546    | **0.695** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jj1kBQWFcuQ3a7P9ttnl1hgX7H8WA_Za?usp=sharing) |
| **rerank** cross encoder      | bge-reranker-base       | 0.666    | **0.706** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QvbUkZtG56SXomGYidwI4RQzwODQrWNm?usp=sharing) |
| **rerank** colbert            | chinese-roberta-wwm-ext | 0.643    | **0.687** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QVtqhQ080ZMltXoJyODMmvEQYI6oo5kO?usp=sharing) |
| **rerank** LLM (LoRA)         | Qwen2-1.5B-Instruct     | 0.531    | **0.699** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fzq1iV7-f8hNKFnjMmpVhVxadqPb9IXk?usp=sharing) |

* The metrics is MAP in 10% [t2-reranking data](https://huggingface.co/datasets/C-MTEB/T2Reranking). Original score of LLM and colbert original is Zero-shot


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

**With source code**
```shell
python -m pip install -U git+https://github.com/LongxingTan/open-retrievals.git
```


## Quick-start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-WBMisdWLeHUKlzJ2DrREXY_kSV8vjP3?usp=sharing)

**Embeddings from pretrained weights**
```python
from retrievals import AutoModelForEmbedding

sentences = [
    'query: how much protein should a female eat',
    'query: summit define',
    "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
]
model_name_or_path = 'intfloat/e5-base-v2'
model = AutoModelForEmbedding.from_pretrained(model_name_or_path, pooling_method="mean")
embeddings = model.encode(sentences, normalize_embeddings=True, convert_to_tensor=True)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
```

**Index building for dense retrieval search**
```python
from retrievals import AutoModelForEmbedding, AutoModelForRetrieval

sentences = ['A dog is chasing car.', 'A man is playing a guitar.']
model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
index_path = './database/faiss/faiss.index'
model = AutoModelForEmbedding.from_pretrained(model_name_or_path)
model.build_index(sentences, index_path=index_path)

query_embed = model.encode("He plays guitar.")
matcher = AutoModelForRetrieval()
dists, indices = matcher.search(query_embed, index_path=index_path)
print(indices)
```

**Rerank using pretrained weights**
```python
from retrievals import AutoModelForRanking

model_name_or_path: str = "BAAI/bge-reranker-base"
rerank_model = AutoModelForRanking.from_pretrained(model_name_or_path)
scores_list = rerank_model.compute_score(["In 1974, I won the championship in Southeast Asia in my first kickboxing match", "In 1982, I defeated the heavy hitter Ryu Long."])
print(scores_list)
```

**RAG with LangChain integration**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fJC-8er-a4NRkdJkwWr4On7lGt9rAO4P?usp=sharing)

```shell
pip install langchain
pip install langchain_community
pip install chromadb
```

```python
from retrievals.tools.langchain import LangchainEmbedding, LangchainReranker, LangchainLLM
from retrievals import AutoModelForRanking
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import Chroma as Vectorstore
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA

persist_directory = './database/faiss.index'
embed_model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
rerank_model_name_or_path = "BAAI/bge-reranker-base"
llm_model_name_or_path = "microsoft/Phi-3-mini-128k-instruct"

embeddings = LangchainEmbedding(model_name=embed_model_name_or_path)
vectordb = Vectorstore(
    persist_directory=persist_directory,
    embedding_function=embeddings,
)
retrieval_args = {"search_type" :"similarity", "score_threshold": 0.15, "k": 10}
retriever = vectordb.as_retriever(**retrieval_args)

ranker = AutoModelForRanking.from_pretrained(rerank_model_name_or_path)
reranker = LangchainReranker(model=ranker, top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=retriever
)

llm = LangchainLLM(model_name_or_path=llm_model_name_or_path)

RESPONSE_TEMPLATE = """[INST]
<>
You are a helpful AI assistant. Use the following pieces of context to answer the user's question.<>
Anything between the following `context` html blocks is retrieved from a knowledge base.

    {context}

REMEMBER:
- If you don't know the answer, just say that you don't know, don't try to make up an answer.
- Let's take a deep breath and think step-by-step.

Question: {question}[/INST]
Helpful Answer:
"""

PROMPT = PromptTemplate(template=RESPONSE_TEMPLATE, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type='stuff',
    retriever=compression_retriever,
    chain_type_kwargs={
        "verbose": True,
        "prompt": PROMPT,
    }
)

user_query = 'Introduce this'
response = qa_chain({"query": user_query})
print(response)
```


**Embedding fine-tuning**

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
train_dataset = train_dataset.rename_columns({'sentence1': 'query', 'sentence2': 'document'})
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
model = AutoModelForEmbedding.from_pretrained(model_name_or_path, pooling_method="cls")
# model = model.set_train_type('pointwise')  # 'pointwise', 'pairwise', 'listwise'
optimizer = AdamW(model.parameters(), lr=5e-5)
num_train_steps = int(len(train_dataset) / batch_size * epochs)
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
    data_collator=PairCollator(tokenizer, query_max_length=128, document_max_length=128),
    loss_fn=InfoNCE(nn.CrossEntropyLoss(label_smoothing=0.05)),
)
trainer.optimizer = optimizer
trainer.scheduler = scheduler
trainer.train()
```

**Reranking Fine-tuning**

```python
from transformers import AutoTokenizer, TrainingArguments, get_cosine_schedule_with_warmup, AdamW
from retrievals import RerankCollator, AutoModelForRanking, RerankTrainer, RerankTrainDataset

model_name_or_path: str = "microsoft/deberta-v3-base"
max_length: int = 128
learning_rate: float = 3e-5
batch_size: int = 4
epochs: int = 3

train_dataset = RerankTrainDataset('./t2rank.json', positive_key='pos', negative_key='neg')
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
model = AutoModelForRanking.from_pretrained(model_name_or_path)
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_train_steps = int(len(train_dataset) / batch_size * epochs)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_steps, num_training_steps=num_train_steps)

training_args = TrainingArguments(
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs,
    output_dir='./checkpoints',
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


## Reference & Acknowledge
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [Dense](https://github.com/luyug/Dense)
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- [uniem](https://github.com/wangyuxinwhy/uniem)
- [BCEmbedding](https://github.com/netease-youdao/BCEmbedding)

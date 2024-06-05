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
<img src="./docs/source/_static/logo.svg" width="420" align=center/>
</h1>

[![LICENSE][license-image]][license-url]
[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Lint Status][lint-image]][lint-url]
[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]


**[中文wiki](https://github.com/LongxingTan/open-retrievals/wiki)** | **[英文文档](https://open-retrievals.readthedocs.io)** | **[Release Notes](https://open-retrievals.readthedocs.io/en/latest/CHANGELOG.html)**

**Open-Retrievals** 帮助开发者在信息检索、大语言模型等领域便捷地应用文本向量，快速搭建检索、排序、RAG等应用。
- 文本向量, 检索，重排都可以简单的一句开始 `AutoModelForEmbedding`
- 多种对比学习与大语言模型进行文本向量微调、rerank微调
- 支持point-wise、pairwise、listwise训练
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
from retrievals import AutoModelForEmbedding

sentences = [
    "在1974年，第一次在东南亚打自由搏击就得了冠军",
    "1982年打赢了日本重炮手雷龙，接着连续三年打败所有日本空手道高手，赢得全日本自由搏击冠军",
    "中国古拳法唯一传人鬼王达，被喻为空手道的克星，绰号魔鬼筋肉人",
    "古人有云，有功夫，无懦夫"
]

model_name_or_path = 'intfloat/multilingual-e5-base'
model = AutoModelForEmbedding.from_pretrained(model_name_or_path)
embeddings = model.encode(sentences)  # 384维度的文本向量
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
```

**使用Faiss向量数据库检索**
```python
from retrievals import AutoModelForEmbedding, AutoRetrieval

index_path = './database/faiss/faiss.index'
sentences = ['A dog is chasing car.', 'A man is playing a guitar.']
model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModelForEmbedding.from_pretrained(model_name_or_path)
model.build_index(sentences, index_path=index_path)

query_embed = model.encode("He plays guitar.")
matcher = AutoRetrieval()
dists, indices = matcher.similarity_search(query_embed, index_path=index_path)
print(indices)
```

**重排**
```python
from retrievals import AutoRanking

model_name_or_path: str = "BAAI/bge-reranker-base"
rerank_model = AutoRanking.from_pretrained(model_name_or_path)
scores_list = rerank_model.compute_score(
    [["在1974年，第一次在东南亚打自由搏击就得了冠军", "1982年打赢了日本重炮手雷龙"],
     ["铁砂掌，源于泗水铁掌帮，三日练成，收费六百", "铁布衫，源于福建省以北70公里，五日练成，收费八百"]]
)
print(scores_list)
```

**搭配Langchain构建RAG应用**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fJC-8er-a4NRkdJkwWr4On7lGt9rAO4P?usp=sharing)

```shell
pip install langchain
pip install langchain_community
pip install chromadb
```

```python
from retrievals.tools.langchain import LangchainEmbedding, LangchainReranker, LangchainLLM
from retrievals import AutoRanking
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import Chroma as Vectorstore
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA

persist_directory = './database/faiss.index'
embed_model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
rerank_model_name_or_path = "BAAI/bge-reranker-base"
llm_model_name_or_path = "microsoft/Phi-3-mini-128k-instruct"

embeddings = LangchainEmbedding(model_name_or_path=embed_model_name_or_path)
vectordb = Vectorstore(
    persist_directory=persist_directory,
    embedding_function=embeddings,
)
retrieval_args = {"search_type" :"similarity", "score_threshold": 0.15, "k": 10}
retriever = vectordb.as_retriever(**retrieval_args)

ranker = AutoRanking.from_pretrained(rerank_model_name_or_path)
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

user_query = '1974年，谁获得了东南亚自由搏击的冠军？'
response = qa_chain({"query": user_query})
print(response)
```


**微调文本向量模型**

- Model performance fine-tuned in [T2Ranking](https://huggingface.co/datasets/THUIR/T2Ranking)

| cfg | Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> |
| :-- | :-: | :-: | :-: | :-: |
| TripletLoss | 672 | 47.7% |52.6% | 61.4% |


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
train_dataset = train_dataset.rename_columns({'sentence1': 'query', 'sentence2': 'document'})
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
model = AutoModelForEmbedding.from_pretrained(model_name_or_path, pooling_method="cls")
# model = model.set_train_type('pointwise')  # 'pointwise', 'pairwise', 'listwise'
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
    data_collator=PairCollator(tokenizer, query_max_length=128, document_max_length=128),
    loss_fn=InfoNCE(nn.CrossEntropyLoss(label_smoothing=0.05)),
)
trainer.optimizer = optimizer
trainer.scheduler = scheduler
trainer.train()
```

- 一键训练

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1w2dRoRThG6DnUW46swqEUuWySKS1AXCp?usp=sharing)

```shell
MODEL_NAME='BAAI/bge-small-zh-v1.5'

torchrun --nproc_per_node 1 \
  -m retrievals.pipelines.embed \
  --output_dir train \
  --overwrite_output_dir \
  --model_name_or_path $MODEL_NAME \
  --do_train \
  --train_data train.jsonl \
  --learning_rate 3e-5 \
  --fp16 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 32 \
  --dataloader_drop_last True \
  --query_max_length 64 \
  --document_max_length 512 \
  --train_group_size 2 \
  --logging_steps 100
```


**微调重排模型**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QvbUkZtG56SXomGYidwI4RQzwODQrWNm?usp=sharing)

```python
from transformers import AutoTokenizer, TrainingArguments, get_cosine_schedule_with_warmup, AdamW
from retrievals import RerankCollator, AutoRanking, RerankTrainer, RerankDataset

model_name_or_path: str = "microsoft/deberta-v3-base"
max_length: int = 128
learning_rate: float = 3e-5
batch_size: int = 4
epochs: int = 3

train_dataset = RerankDataset('./t2rank.json', positive_key='pos', negative_key='neg')
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
model = AutoRanking.from_pretrained(model_name_or_path, pooling_method="mean")
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
    data_collator=RerankCollator(tokenizer, query_max_length=max_length, document_max_length=max_length),
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

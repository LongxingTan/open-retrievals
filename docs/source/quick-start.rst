Quick start
======================

.. _quick-start:

We can easily use Open-retrievals to fine-tune the model easily for information retrieval and RAG application.

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1-WBMisdWLeHUKlzJ2DrREXY_kSV8vjP3?usp=sharing
    :alt: Open In Colab


1. Embedding
-----------------------------

We can use the pretrained embedding easily from transformers or sentence-transformers.

.. code-block:: python

    from retrievals import AutoModelForEmbedding

    sentences = [
        'query: how much protein should a female eat',
        'query: summit define',
        "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. ",
        "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level."
    ]
    model_name_or_path = 'intfloat/e5-base-v2'
    # sentence embedding mode
    model = AutoModelForEmbedding.from_pretrained(model_name_or_path, pooling_method="mean")
    # encode the sentence to embedding vector
    embeddings = model.encode(sentences, normalize_embeddings=True, convert_to_tensor=True)
    scores = (embeddings[:2] @ embeddings[2:].T) * 100
    print(scores.tolist())

.. code::

    [[89.92379760742188, 68.0742416381836], [68.93356323242188, 91.32250213623047]]


Embedding fine-tuned
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we want to further improve the retrieval performance, an optional method is to fine tune the embedding model weights. It will project the vector of query and answer to similar representation space.

.. code-block:: python

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


2. Indexing
-----------------------------

Save the document embedding offline.

.. code-block:: python

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


3. Rerank
-----------------------------

If we have multiple retrieval source or a better sequence, we can add the reranking to pipeline.

.. code-block:: python

    from retrievals import AutoModelForRanking

    model_name_or_path: str = "BAAI/bge-reranker-base"
    rerank_model = AutoModelForRanking.from_pretrained(model_name_or_path)
    scores_list = rerank_model.compute_score(["In 1974, I won the championship in Southeast Asia in my first kickboxing match", "In 1982, I defeated the heavy hitter Ryu Long."])
    print(scores_list)

.. code::

    -5.075255870819092


Rerank fine-tuned
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

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


4. RAG
-----------------------------

We can use open-retrievals easily to build RAG application, or integrated with LangChain and Llamaindex.

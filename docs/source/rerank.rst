Rerank
===============================

.. _rerank:

1. Use reranking from open-retrievals
-------------------------------------------

Cross encoder reranking

.. code-block:: python

    from retrievals import AutoModelForRanking

    sentences = [
        ["In 1974, I won the championship in Southeast Asia in my first kickboxing match", "In 1982, I defeated the heavy hitter Ryu Long."],
        ['A dog is chasing car.', 'A man is playing a guitar.'],
    ]
    model_name_or_path: str = "BAAI/bge-reranker-base"
    model = AutoModelForRanking.from_pretrained(model_name_or_path)
    scores_list = model.compute_score(sentences)
    print('Ranking score: ', scores_list)

.. code::

    Ranking score: [-5.075257778167725, -10.194067001342773]


ColBERT reranking

.. code-block:: python

    from retrievals import ColBERT

    sentences = [
        ["In 1974, I won the championship in Southeast Asia in my first kickboxing match", "In 1982, I defeated the heavy hitter Ryu Long."],
        ["In 1974, I won the championship in Southeast Asia in my first kickboxing match", "A man is playing a guitar."],
    ]
    model_name_or_path: str = 'BAAI/bge-m3'
    model = ColBERT.from_pretrained(
        model_name_or_path,
        colbert_dim=1024,
        use_fp16=True,
    )
    embeddings = model.encode(sentences[0], normalize_embeddings=True)
    print('Embedding shape: ', embeddings.shape)

    scores_list = model.compute_score(sentences)
    print('Ranking score: ', scores_list)

.. code::

    Embedding shape: (2, 21, 1024)
    Ranking score: [5.445939064025879, 3.0762712955474854]


2. Fine-tune cross-encoder reranking model
-----------------------------------------------

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1QvbUkZtG56SXomGYidwI4RQzwODQrWNm?usp=sharing
    :alt: Open In Colab


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
    model = AutoModelForRanking.from_pretrained(model_name_or_path, pooling_method="mean")
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


3. Fine-tune ColBERT reranking model
----------------------------------------

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1QVtqhQ080ZMltXoJyODMmvEQYI6oo5kO?usp=sharing
    :alt: Open In Colab


4. Fine-tune LLM reranker
-------------------------------------

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1fzq1iV7-f8hNKFnjMmpVhVxadqPb9IXk?usp=sharing
    :alt: Open In Colab


- Point-wise style prompt:

    "Passage: {text}\nPlease write a question based on this passage."

- Point-wise style prompt:

    "Passage: {text}\nQuery: {query}\nDoes the passage answer the query? Answer 'Yes' or 'No'"

- pairwise style prompt:

    """Given a query "{query}", which of the following two passages is more relevant to the query?

    Passage A: "{doc1}"

    Passage B: "{doc2}"

    Output Passage A or Passage B:"""

- listwise style prompt:

    I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."

- set-wise style prompt:

    Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
    + passages + '\n\nOutput only the passage label of the most relevant passage:'


Reference
-------------------

- https://github.com/ielab/llm-rankers/tree/main

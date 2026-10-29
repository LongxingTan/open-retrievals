Quick start
======================

.. _quick-start:

We can use Open-retrievals to easily fine-tune models of information retrieval and RAG.

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1-WBMisdWLeHUKlzJ2DrREXY_kSV8vjP3?usp=sharing
    :alt: Open In Colab


1. Embedding
-----------------------------

Embedding from the pretrained model

.. code-block:: python

    from retrievals import AutoModelForEmbedding

    sentences = [
        'query: how much protein should a female eat',
        'query: summit define',
        "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. ",
        "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level."
    ]
    model_name_or_path = 'intfloat/e5-base-v2'
    # sentence embedding model
    model = AutoModelForEmbedding.from_pretrained(model_name_or_path, pooling_method="mean")
    # encode the sentence to embedding vector
    embeddings = model.encode(sentences, normalize_embeddings=True, convert_to_tensor=True)
    scores = (embeddings[:2] @ embeddings[2:].T) * 100
    print(scores.tolist())

.. code::

    [[89.92379760742188, 68.0742416381836], [68.93356323242188, 91.32250213623047]]

.. note::

    - Use the right pooling_method while calling, add the query_instruction if it's not embedding into the data already.
    - Don't forget set remove_unused_columns=False in TrainingArguments


1.1 Fine-tune embedding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To further improve the retrieval performance, we can fine tune the embedding model. It can project the vector of query and answer to similar representation space to get better retrieval performance.

.. code-block:: python

    from retrievals.train import train

    trainer = train(
        "examples/configs/dense_retrieval.yaml",
        overrides=["training.learning_rate=3e-5"],
    )

Run from the repository root after installing ``pip install -e .``.
The example uses fixture data; replace ``data.train_file`` with your own JSON/JSONL,
CSV, Parquet, or Hugging Face dataset configuration. Dense rows contain ``query``,
``positive`` (text or a list of texts), and optional ``negative`` values.
``train()`` saves the model, tokenizer, and resolved configuration in
``output_dir/model/``.



It's also easy to use multiple gpus for transformers `Trainer`.

.. note::

    `transformer trainer docs <https://huggingface.co/docs/transformers/main_classes/trainer>`_

1.2 Serving embedding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


2. Indexing
-----------------------------

Save the document embedding offline using the vector database.

.. code-block:: python

    from retrievals import AutoModelForEmbedding, AutoModelForRetrieval

    sentences = ['A dog is chasing car.', 'A man is playing a guitar.']
    model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
    index_path = './database/faiss/faiss.index'
    model = AutoModelForEmbedding.from_pretrained(model_name_or_path, pooling_method='mean')
    from pathlib import Path

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    model.build_index(sentences, index_path=index_path)

    query_embed = model.encode("He plays guitar.")
    matcher = AutoModelForRetrieval()
    dists, indices = matcher.search(query_embed, index_path=index_path, top_k=2)
    print(indices)


3. Rerank
-----------------------------

If we have multiple retrieval source or a better sequence, we can add the reranking to pipeline.

.. code-block:: python

    from retrievals import AutoModelForRanking

    sentences = [
        ["In 1974, I won the championship in Southeast Asia in my first kickboxing match", "In 1982, I defeated the heavy hitter Ryu Long."],
        ['A dog is chasing car.', 'A man is playing a guitar.'],
    ]
    model_name_or_path: str = "BAAI/bge-reranker-base"
    rerank_model = AutoModelForRanking.from_pretrained(model_name_or_path)
    scores_list = rerank_model.compute_score(sentences)
    print(scores_list)

.. code::

    [-5.075257778167725, -10.194067001342773]


3.1 Fine-tune reranking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarly, fine tune the reranking model to get a better performance for the specific task.

.. code-block:: python

    from retrievals.train import train

    trainer = train("examples/configs/cross_encoder.yaml")

Cross-encoder YAML training uses labeled query/document pairs and BCE with logits.
Map custom columns with ``data.query_col``, ``data.doc_col``, and ``data.label_col``.
Set ``data.valid_file`` and ``training.eval_strategy: epoch`` for held-out loss.
Resume using ``training.resume_from_checkpoint`` pointing at a ``checkpoint-*``
directory; the final ``model/`` export is intended for inference.



4. RAG
-----------------------------

Use open-retrievals to build RAG easily, or integrated with LangChain or LlamaIndex.

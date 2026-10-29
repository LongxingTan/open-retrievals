Rerank
===============================

.. _rerank:

1. Use reranking from open-retrievals
-------------------------------------------

**Cross encoder reranking**

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


**ColBERT reranking**

.. code-block:: python

    from retrievals import ColBERT

    sentences = [
        ["In 1974, I won the championship in Southeast Asia in my first kickboxing match", "In 1982, I defeated the heavy hitter Ryu Long."],
        ["In 1974, I won the championship in Southeast Asia in my first kickboxing match", "A man is playing a guitar."],
    ]
    model_name_or_path: str = 'outputs/colbert/model'
    model = ColBERT.from_pretrained(
        model_name_or_path,
        colbert_dim=128,
        use_fp16=True,
    )
    scores_list = model.compute_score(sentences)
    print('Ranking score: ', scores_list)

Use a trained ColBERT export so its projection weights are restored. Loading a
plain backbone without projection weights initializes a new projection.



**LLM generative reranking**

- `AutoModelForRanking.from_pretrained(model_name_or_path, causal_lm=True)`
- Prompt: "Given a query with a relevant body, determine whether the document is pertinent to the query by providing a prediction of either 'Yes' or 'No'."


.. code-block:: python

    from retrievals import LLMRanker

    model_name = 'BAAI/bge-reranker-v2-gemma'
    model = LLMRanker.from_pretrained(
                model_name,
                causal_lm=True,
                use_fp16=True,
            )

    scores = model.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
    print('Ranking score: ', scores)


2. Fine-tune cross-encoder reranking model
-----------------------------------------------

prepare data
    `{(query1, document1, label1), (query2, document2, label2), ...}`


.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1QvbUkZtG56SXomGYidwI4RQzwODQrWNm?usp=sharing
    :alt: Open In Colab


.. code-block:: python

    from retrievals.train import train

    trainer = train("examples/configs/cross_encoder.yaml")

Provide rows such as ``{"query": "cat", "document": "pet", "label": 1}``.
Map columns with ``data.query_col``, ``data.doc_col``, and ``data.label_col``.
The YAML workflow supports binary classification with labels in [0, 1] and
``loss.loss_type: bce``. Regression and listwise objectives are not YAML workflows.



3. Fine-tune ColBERT reranking model
----------------------------------------

prepare data
    `{}`


.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1QVtqhQ080ZMltXoJyODMmvEQYI6oo5kO?usp=sharing
    :alt: Open In Colab

.. code-block:: python

    from retrievals.train import train

    trainer = train("examples/configs/colbert.yaml")

ColBERT uses query/positive/negative rows, ``loss.loss_type: colbert``, and masked
late interaction. Set projection size with ``model.colbert_dim`` or
``colbert.colbert_dim``. Positive-only batches require
``loss.use_inbatch_negative: true``. Punctuation masking is not implemented.
Exports include ``colbert_linear.pt`` alongside the backbone and tokenizer.



4. Fine-tune LLM Generative reranker
-------------------------------------

prepare generative reranking data
    `{}`

prepare representative reranking data
    `{}`


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


The following commands use the legacy pipeline interface. Generative LLM
reranking is separate from YAML cross-encoder classification.

**Cross encoder reranking**

.. code-block:: bash

    rag-cli init cross_encoder --output train.yaml
    rag-cli train train.yaml

Use YAML configuration for training. See the YAML training guide for supported tasks.

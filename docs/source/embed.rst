Embedding
======================================

.. _embed:

1. Use embedding from open-retrievals
---------------------------------------

we can use `AutoModelForEmbedding` to get the text embedding from pretrained transformer or LLM.

The Transformer model could get a representation vector from a sentence.


**Transformer encoder embedding model**

- Choose the right `pooling_method`, check in `huggingface <https://huggingface.co/models>`_

.. code-block:: python

    from retrievals import AutoModelForEmbedding

    model = AutoModelForEmbedding.from_pretrained('moka-ai/m3e-base', pooling_method='mean')
    sentences = [
        'The cat is sleeping on the couch.',
        'He decided to take a walk after dinner.',
        'I want to attend Kaggle'
    ]
    embeddings = model.encode(sentences)


**LLM decoder embedding model**

.. code-block:: python

    from retrievals import AutoModelForEmbedding

    model_name = 'intfloat/e5-mistral-7b-instruct'
    model = AutoModelForEmbedding.from_pretrained(
                model_name,
                pooling_method='last',
                use_fp16=True,
                query_instruction='Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {}',
                document_instruction='',
            )

Use ``model.encode(queries, is_query=True)`` to apply the query instruction.
Instruction templates must include ``{}`` for the input text.


2. Fine-tune
--------------------

Prepare data
~~~~~~~~~~~~~~~~~~~~

- Text label: point-wise fine-tuning

    `{(query, label), (document, label), ...}`

- Text pair: in-batch negative pairwise fine-tuning

    `{(query, positive, negative), {query, positive, negative}, ...}`

- Triplet pair: hard negative fine-tuning

    `{(query, positive, negative1, negative2, negative3), (query, positive, negative1, negative2, negative3), ...}`

- Text scored pair

    `{(query, positive, label), (query, negative, label), ...}`

- listwise


Pair wise
~~~~~~~~~~~~~

If the positive and negative examples have some noise in label, the directly point-wise cross-entropy maybe not the best. The pair wise just compare relatively, or the hinge loss with margin could be better.

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/17KXe2lnNRID-HiVvMtzQnONiO74oGs91?usp=sharing
    :alt: Open In Colab


.. code-block:: python

    from retrievals.train import train

    trainer = train("examples/configs/dense_retrieval.yaml")

The YAML workflow supports InfoNCE (including ``multiple_negatives_ranking`` aliases)
and triplet loss. Set ``data.negative_samples: 0`` for positive-only InfoNCE batches.
In-batch negatives are local to each process. Every row in a batch must agree on
whether explicit negatives are present.

For decoder models, set ``model.pooling_method: last``. Optional LoRA uses
``model.use_lora: true`` and requires ``pip install -e '.[lora]'``.
Include model-specific instructions directly in the training text.
The commands below use the separate legacy pipeline argument interface.



**Pairwise fine-tune embedding model**

.. code-block:: bash

    rag-cli init dense --output train.yaml
    rag-cli train train.yaml

Use YAML configuration for training. See the YAML training guide for supported tasks.

Retrieval
============================

.. _retrieval:

1. Pipeline
----------------------------

The retrieval method could solve the **search** or **extreme multiclass classification** problem.

generate data -> train -> eval

pretrained encoding -> build hard negative -> train -> eval -> indexing -> retrieval

pretrain -> fine tuning -> distill


2. Offline indexing
----------------------------


.. code-block:: python

    from retrievals import AutoModelForEmbedding, AutoModelForRetrieval

    model = AutoModelForEmbedding.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", pooling_method="mean"
    )
    documents = ["A dog is chasing a car.", "A man is playing a guitar."]
    document_embeddings = model.encode(
        documents, normalize_embeddings=True, convert_to_tensor=True
    )
    query_embeddings = model.encode(
        ["He plays guitar."], normalize_embeddings=True, convert_to_tensor=True
    )
    matcher = AutoModelForRetrieval(method="cosine")
    scores, indices = matcher.search(
        query_embeddings, document_embed=document_embeddings, top_k=2
    )

The cosine search path expects tensors and computes dot products; normalize both
sets of embeddings for cosine similarity. Returned indices refer to document order.
For persisted indexes, install ``faiss-cpu``, call ``model.build_index()`` with an
``index_path``, then pass NumPy query embeddings and that path to ``search()``.



3. Retrieval
----------------------------


Faiss retrieval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



BM25 retrieval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Elastic search retrieval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Ensemble retrieval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

we can use `RRF_fusion` to ensemble multiple retrievals to improve the retrieval performance.

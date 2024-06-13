Quick start
======================

.. _quick-start:

1. Embedding
-----------------------------

We can use the pretrained embedding easily from transformers or sentence-transformers.

.. code-block:: python

    from retrievals import AutoModelForEmbedding

    sentences = ["Hello NLP", "Open-retrievals is designed for retrieval, rerank and RAG"]
    model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
    model = AutoModelForEmbedding.from_pretrained(model_name_or_path, pooling_method="mean")
    sentence_embeddings = model.encode(sentences, normalize_embeddings=True, convert_to_tensor=True)
    print(sentence_embeddings)


Embedding fine-tuned
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we want to further improve the retrieval performance, an optional method is to fine tune the embedding model weights. It will project the vector of query and answer to similar representation space.


2. Indexing
-----------------------------

Save the document embedding offline.


3. Rerank
-----------------------------

If we have multiple retrieval source or a better sequence, we can add the reranking to pipeline.


Rerank fine-tuned
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


4. RAG
-----------------------------

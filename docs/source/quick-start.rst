Quick start
======================

.. _quick-start:

Use the pretrained weight as embedding
---------------------------------------------

You can use the pretrained embedding easily from transformers or sentence-transformers.

.. code-block:: python

    from retrievals import AutoModelForEmbedding

    sentences = ["Hello NLP", "Open-retrievals is designed for retrieval, rerank and RAG"]
    model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
    model = AutoModelForEmbedding.from_pretrained(model_name_or_path, pooling_method="mean")
    sentence_embeddings = model.encode(sentences, normalize_embeddings=True, convert_to_tensor=True)
    print(sentence_embeddings)


Fine tune the transformer pretrained weight by contrastive learning
----------------------------------------------------------------------


Query search by faiss
--------------------------


Rerank to enhance the performance
----------------------------------------


Langchain example for RAG
--------------------------------

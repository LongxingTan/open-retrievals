Open-Retrievals Documentation
======================================
.. raw:: html

   <a class="github-button" href="https://github.com/LongxingTan/open-retrievals" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star LongxingTan/open-retrievals on GitHub">GitHub</a>

Retrievals is an easy, flexible, scalable framework supporting state-of-the-art embeddings, retrieval and reranking for information retrieval or RAG.

* Embedding fine-tuned through point-wise, pairwise, listwise, contrastive learning and LLM.
* Reranking fine-tuned with Cross-Encoder, ColBERT and LLM.
* Easily build enhanced modular RAG, integrated with Transformers, Langchain and LlamaIndex.


Installation
------------------

Install the **prerequisites**

* transformers
* peft
* faiss-cpu


Now you are ready, proceed with

.. code-block:: shell

    # install with basic module
    pip install open-retrievals

    # install with support of evaluation
    pip install open-retrievals[eval]

Or install from source code

.. code-block:: shell

    python -m pip install -U git+https://github.com/LongxingTan/open-retrievals.git


Examples
------------------

Run a simple example

.. code-block:: python

    from retrievals import AutoModelForEmbedding

    sentences = ["Hello NLP", "Open-retrievals is designed for retrieval, rerank and RAG"]
    model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
    model = AutoModelForEmbedding.from_pretrained(model_name_or_path, pooling_method="mean")
    sentence_embeddings = model.encode(sentences, normalize_embeddings=True)
    print(sentence_embeddings)

Open-retrievals support to fine-tune the embedding model, reranking model, llm easily for custom usage.

* `Embedding pairwise fine-tuning <https://github.com/LongxingTan/open-retrievals/blob/master/examples/embedding_pairwise_finetune.py>`_
* `LLM embedding pairwise fine-tuning <https://github.com/LongxingTan/open-retrievals/blob/master/examples/embedding_llm_finetune.py>`_
* `ColBERT fine-tuning <https://github.com/LongxingTan/open-retrievals/blob/master/examples/rerank_colbert.py>`_
* `Cross-encoder reranking fine-tuning <https://github.com/LongxingTan/open-retrievals/blob/master/examples/rerank_cross_encoder.py>`_
* `LLM reranking fine-tuning <https://github.com/LongxingTan/open-retrievals/blob/master/examples/rerank_llm_finetune.py>`_


More datasets examples

* `T2 ranking dataset <https://github.com/LongxingTan/open-retrievals/tree/master/examples/t2_ranking>`_
* `scifact dataset <https://github.com/LongxingTan/open-retrievals/tree/master/examples/scifact>`_
* `msmacro dataset <https://github.com/LongxingTan/open-retrievals/tree/master/examples/msmacro>`_
* `wikipedia nq dataset <https://github.com/LongxingTan/open-retrievals/tree/master/examples/wikipedia-nq>`_
* `rag example <https://github.com/LongxingTan/open-retrievals/tree/master/examples/rag>`_


Contributing
---------------------

If you want to contribute to the project, please refer to our `contribution guidelines <https://github.com/LongxingTan/Time-series-prediction/blob/master/CONTRIBUTING.md>`_.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   quick-start
   embed.rst
   retrieval.rst
   rerank.rst
   rag.rst

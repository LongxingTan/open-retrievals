Open-Retrievals Documentation
======================================
.. raw:: html

   <a class="github-button" href="https://github.com/LongxingTan/open-retrievals" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star LongxingTan/open-retrievals on GitHub">GitHub</a>

Retrievals is an easy-to-use framework supporting state-of-the-art embeddings, retrieval and reranking for information retrieval or RAG, based on PyTorch and Transformers.

* Contrastive learning enhanced embeddings
* LLM embeddings


Installation
------------------

Install the **prerequisites**

* transformers
* peft
* faiss-cpu


Now you are ready, proceed with

.. code-block:: shell

    $ pip install open-retrievals

To run evaluation

.. code-block:: shell

    $ pip install open-retrievals[eval]


Examples
------------------

.. image:: https://github.com/LongxingTan/open-retrievals/blob/master/docs/source/_static/structure.png
   :alt: retrievals Image


* `T2 ranking dataset <https://github.com/LongxingTan/open-retrievals/tree/master/examples/t2_ranking>`_
* `scifact dataset <https://github.com/LongxingTan/open-retrievals/tree/master/examples/scifact>`_
* `msmacro dataset <https://github.com/LongxingTan/open-retrievals/tree/master/examples/msmacro>`_
* `wikipedia nq dataset <https://github.com/LongxingTan/open-retrievals/tree/master/examples/wikipedia-nq>`_
* `rag example <https://github.com/LongxingTan/open-retrievals/tree/master/examples/rag>`_
* `graph rag example <URL>`_


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   quick-start
   embed.rst
   retrieval.rst
   rerank.rst
   rag.rst

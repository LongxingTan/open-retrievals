.. Retrievals documentation master file, created by
   sphinx-quickstart on Mon Feb 19 14:43:55 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Open-Retrievals Documentation
======================================
.. raw:: html

   <a class="github-button" href="https://github.com/LongxingTan/open-retrievals" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star LongxingTan/open-retrievals on GitHub">GitHub</a>

Retrievals is an easy-to-use python framework supporting state-of-the-art embeddings, especially for information retrieval and rerank in NLP/LLM, based on PyTorch and Transformers.

* Contrastive learning enhanced embeddings
* LLM embeddings


Installation
------------------

Install the **Prerequisites**

* transformers
* faiss-cpu / faiss-gpu


Now you are ready, proceed with

.. code-block:: shell

    $ pip install open-retrievals

To run evaluation

.. code-block:: shell

    $ pip install open-retrievals[eval]


Examples
------------------


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quick-start
   embed.rst
   retrieval.rst
   rerank.rst
   rag.rst

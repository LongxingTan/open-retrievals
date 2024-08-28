Open-Retrievals Documentation
======================================
.. raw:: html

   <a class="github-button" href="https://github.com/LongxingTan/open-retrievals" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star LongxingTan/open-retrievals on GitHub">GitHub</a>

Retrievals is an easy, flexible, scalable framework supporting state-of-the-art embeddings, retrieval and reranking for information retrieval or RAG, based on PyTorch and Transformers.

* Embeddings fine-tuned by Contrastive learning
* Embeddings from LLM model


.. image:: https://github.com/LongxingTan/open-retrievals/blob/master/docs/source/_static/structure.png
   :alt: retrievals Image


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


Examples
------------------

* `T2 ranking dataset <https://github.com/LongxingTan/open-retrievals/tree/master/examples/t2_ranking>`_
* `scifact dataset <https://github.com/LongxingTan/open-retrievals/tree/master/examples/scifact>`_
* `msmacro dataset <https://github.com/LongxingTan/open-retrievals/tree/master/examples/msmacro>`_
* `wikipedia nq dataset <https://github.com/LongxingTan/open-retrievals/tree/master/examples/wikipedia-nq>`_
* `rag example <https://github.com/LongxingTan/open-retrievals/tree/master/examples/rag>`_
* `graph rag example <URL>`_

Run a simple example

.. code-block:: python

    import retrievals



Contributing
---------------------

If you wish to contribute to the project, please refer to our `contribution guidelines <https://github.com/LongxingTan/Time-series-prediction/blob/master/CONTRIBUTING.md>`_.

Open-Retrievals Documentation
======================================
.. raw:: html

   <a class="github-button" href="https://github.com/LongxingTan/open-retrievals" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star LongxingTan/open-retrievals on GitHub">GitHub</a>

Retrievals is an easy, flexible, scalable framework supporting state-of-the-art embeddings, retrieval and reranking for information retrieval or RAG.

* Embedding fine-tuned through point-wise, pairwise, listwise, contrastive learning and LLM.
* Reranking fine-tuned with Cross-Encoder, ColBERT and LLM.
* Easily build modular RAG, integrated with Transformers, Langchain and LlamaIndex.


Installation
------------------

Install the **prerequisites**

* transformers
* peft  # for lora fine-tuning if necessary
* faiss-cpu  # for faiss retrieval if necessary


Now you are ready, proceed with

.. code-block:: shell

    # install with basic module
    pip install open-retrievals

    # install with support of evaluation
    pip install 'open-retrievals[eval]'

Or install from source code

.. code-block:: shell

    git clone https://github.com/LongxingTan/open-retrievals.git
    cd open-retrievals
    pip install -e .
    # Optional LoRA support
    pip install -e '.[lora]'

Train the supported YAML tasks from the repository root:

.. code-block:: shell

    rag-cli train examples/configs/dense_retrieval.yaml
    rag-cli train examples/configs/cross_encoder.yaml
    rag-cli train examples/configs/colbert.yaml

These workflows use Transformers Trainer. Ray, verl, LLaMA-Factory execution,
quantized YAML training, and YAML distillation are not implemented. Other CLI
commands such as ``api``, ``chat``, ``export``, and ``webui`` are placeholders.


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

* `Embedding pairwise fine-tuning <https://github.com/LongxingTan/open-retrievals/blob/master/examples/0_embedding/train_pairwise.md>`_
* `LLM embedding pairwise fine-tuning <https://github.com/LongxingTan/open-retrievals/blob/master/examples/0_embedding/train_llm.md>`_
* `ColBERT fine-tuning <https://github.com/LongxingTan/open-retrievals/blob/master/examples/3_colbert/train_colbert.md>`_
* `Cross-encoder reranking fine-tuning <https://github.com/LongxingTan/open-retrievals/blob/master/examples/2_reranking/train_cross_encoder.md>`_
* `LLM reranking fine-tuning <https://github.com/LongxingTan/open-retrievals/blob/master/examples/2_reranking/train_llm.md>`_


Use cases
------------------

* `T2 ranking dataset <https://github.com/LongxingTan/open-retrievals/tree/master/examples/t2_ranking>`_
* `scifact dataset <https://github.com/LongxingTan/open-retrievals/tree/master/examples/scifact>`_
* `msmacro dataset <https://github.com/LongxingTan/open-retrievals/tree/master/examples/msmacro>`_
* `wikipedia nq dataset <https://github.com/LongxingTan/open-retrievals/tree/master/examples/wikipedia-nq>`_
* `rag example <https://github.com/LongxingTan/open-retrievals/tree/master/examples/4_rag>`_


Contributing
---------------------

If you want to contribute to the project, please refer to our `contribution guidelines <https://github.com/LongxingTan/open-retrievals/blob/master/CONTRIBUTING.md>`_.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   quick-start
   embed
   retrieval
   rerank
   rag

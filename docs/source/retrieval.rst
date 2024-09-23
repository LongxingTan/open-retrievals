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


.. code-block:: shell

    QUERY_ENCODE_DIR=nq-queries
    OUT_DIR=temp
    MODEL_DIR="BAAI/bge-base-zh-v1.5"
    QUERY=nq-test-queries.json
    mkdir $QUERY_ENCODE_DIR

    python -m retrievals.pipelines.embed \
        --model_name_or_path $MODEL_DIR \
        --output_dir $OUT_DIR \
        --do_encode \
        --fp16 \
        --per_device_eval_batch_size 256 \
        --data_name_or_path $QUERY \
        --is_query true


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

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
        '* Moka 此文本嵌入模型由 MokaAI 训练并开源，训练脚本使用 uniem',
        '* Massive 此文本嵌入模型通过**千万级**的中文句对数据集进行训练',
        '* Mixed 此文本嵌入模型支持中英双语的同质文本相似度计算，异质文本检索等功能，未来还会支持代码检索，ALL in one'
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
                query_instruction='Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ',
                document_instruction='',
            )

.. code::

    [[82.9375, 47.96875], [46.9375, 81.8125]]


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

    import torch.nn as nn
    from datasets import load_dataset
    from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, TrainingArguments
    from retrievals import AutoModelForEmbedding, RetrievalTrainer, PairCollator, TripletCollator
    from retrievals.losses import ArcFaceAdaptiveMarginLoss, InfoNCE, SimCSE, TripletLoss

    model_name_or_path: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    batch_size: int = 128
    epochs: int = 3

    train_dataset = load_dataset('shibing624/nli_zh', 'STS-B')['train']
    train_dataset = train_dataset.rename_columns({'sentence1': 'query', 'sentence2': 'positive'})
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = AutoModelForEmbedding.from_pretrained(model_name_or_path, pooling_method="mean")
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_train_steps=int(len(train_dataset) / batch_size * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_steps, num_training_steps=num_train_steps)

    training_arguments = TrainingArguments(
        output_dir='./checkpoints',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        remove_unused_columns=False,
    )
    trainer = RetrievalTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        data_collator=PairCollator(tokenizer, query_max_length=128, document_max_length=128),
        loss_fn=InfoNCE(nn.CrossEntropyLoss(label_smoothing=0.05)),
    )
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.train()


**Pairwise fine-tune embedding model**

.. code-block:: shell

    MODEL_NAME="BAAI/bge-base-zh-v1.5"
    TRAIN_DATA="/t2_ranking.jsonl"
    OUTPUT_DIR="/t2_output"

    torchrun --nproc_per_node 1 \
      -m retrievals.pipelines.embed \
      --output_dir $OUTPUT_DIR \
      --overwrite_output_dir \
      --model_name_or_path $MODEL_NAME \
      --do_train \
      --data_name_or_path $TRAIN_DATA \
      --positive_key positive \
      --negative_key negative \
      --learning_rate 3e-5 \
      --fp16 \
      --num_train_epochs 5 \
      --per_device_train_batch_size 32 \
      --dataloader_drop_last True \
      --query_max_length 64 \
      --document_max_length 512 \
      --train_group_size 4 \
      --logging_steps 100 \
      --temperature 0.02 \
      --use_inbatch_negative false


**Pairwise fine-tune LLM embedding**

.. code-block:: shell

    MODEL_NAME="intfloat/e5-mistral-7b-instruct"
    TRAIN_DATA="/t2_ranking.jsonl"
    OUTPUT_DIR="/t2_output"

    torchrun --nproc_per_node 1 \
      -m retrievals.pipelines.embed \
      --output_dir $OUTPUT_DIR \
      --overwrite_output_dir \
      --model_name_or_path $MODEL_NAME \
      --pooling_method last \
      --do_train \
      --data_name_or_path $TRAIN_DATA \
      --positive_key positive \
      --negative_key negative \
      --use_lora True \
      --query_instruction "Retrieve the possible answer for query.\nQuery: " \
      --document_instruction 'Document: ' \
      --learning_rate 2e-4 \
      --bf16 \
      --num_train_epochs 3 \
      --per_device_train_batch_size 4 \
      --gradient_accumulation_steps 16 \
      --dataloader_drop_last True \
      --query_max_length 64 \
      --document_max_length 256 \
      --train_group_size 4 \
      --logging_steps 100 \
      --temperature 0.02 \
      --use_inbatch_negative false \
      --save_total_limit 1


Point wise
~~~~~~~~~~~~~~~~~~

We can use point-wise train, similar to use `tfidf` in information retrieval.

**arcface**

- layer wise learning rate
- batch size is important
- dynamic arcface_margin, margin is important
- arc_weight init


List wise
~~~~~~~~~~~~~~~~~~


3. Training skills to enhance the performance
----------------------------------------------

multiple gpus

multiple precisions: int4, int8, float16, bfloat16


* Pretrain
* In batch negative
* Hard negative, multiple rounds negative
* Cross batch negative
* knowledge distill from cross encoder
* maxsim (multi vector)
* Matryoshka

tuning the important parameters:

* temperature


Hard negative mining
~~~~~~~~~~~~~~~~~~~~~~~~~

- offline hard mining or online hard mining

If we only have query and positive, we can use it to generate more negative samples to enhance the retrieval performance.

The data format of `input_file` to generate hard negative is `(query, positive)` or `(query, positive, negative)`
The format of `candidate_pool` of corpus is jsonl of `{text}`


.. code-block:: shell

    python -m retrievals.pipelines.build_hn \
        --model_name_or_path BAAI/bge-base-en-v1.5 \
        --input_file /t2_ranking.jsonl \
        --output_file /t2_ranking_hn.jsonl \
        --positive_key positive \
        --negative_key negative \
        --range_for_sampling 2-200 \
        --negative_number 15 \


Matryoshka Representation Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Contrastive loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


binary classification:

- similarity(query, positive) > similarity(query, negative)
- hinge loss: max(0, similarity(query, positive) - similarity(query, negative) + margin)
- logistic loss: logistic(similarity(query, positive) - similarity(query, negative))

multi-label classification:

- similarity(query, positive), similarity(query, negative1), similarity(query, negative2)


cosent loss

- similar to circle loss, but with cosine


Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


4. Embedding serving
----------------------------------------------

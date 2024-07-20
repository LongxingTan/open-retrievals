Embedding
==============================

.. _embed:


Pretrained
---------------------

we can use `AutoModelForEmbedding` to get the sentence embedding from pretrained transformer or large language model.


Fine-tune
------------------

point-wise

- `{(query, label), (document, label)}`


pairwise

- `{(query, positive, label), (query, negative, label)}`

- `{(query, positive, negative), {query, positive, negative}}`

- `{(query, positive, negative1, negative2, negative3...)}`

listwise

- `{(query+positive)}`


Loss function
~~~~~~~~~~~~~~~~~~~~~~

- binary classification:
    - similarity(query, positive) > similarity(query, negative)
    - hinge loss: max(0, similarity(query, positive) - similarity(query, negative) + margin)
    - logistic loss: logistic(similarity(query, positive) - similarity(query, negative))
- multi-label classification:
    - similarity(query, positive), similarity(query, negative1), similarity(query, negative2)


Pair wise
~~~~~~~~~~~~~

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
    model = AutoModelForEmbedding.from_pretrained(model_name_or_path, pooling_method="cls")
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



Point wise
~~~~~~~~~~~~~

If the positive and negative examples have some noise in label, the directly point-wise cross-entropy maybe not the best. The pair wise just compare relatively, or the hinge loss with margin could be better.

arcface

- layer wise learning rate
- batch size is important
- dynamic arcface_margin, margin is important
- arc_weight init


List wise
~~~~~~~~~~~~~~


Enhance the performance
--------------------------------------

* Pretrain
* In batch negative
* Hard negative, multiple rounds negative
* Cross batch negative
* knowledge distill from cross encoder
* maxsim (multi vector)
* Matryoshka

tuning the important parameters:

* temperature


Hard mining
~~~~~~~~~~~~~~~~~~~~~~
offline hard mining

online hard mining


Matryoshka Representation Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

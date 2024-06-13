Embedding
====================

.. _embed:


Fine-tuning text embedding
------------------------------

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
    # model = model.set_train_type('pointwise')  # 'pointwise', 'pairwise', 'listwise'
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




Point-wise
--------------

arcface
- 分层学习率
- batch size影响大
- arcface_margin动态调整, margin大小影响较大
- arc_weight初始化
- 含状态训练的损失函数不适合每个epoch训练时也过一遍评价指标

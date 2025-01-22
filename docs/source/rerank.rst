Rerank
===============================

.. _rerank:

1. Use reranking from open-retrievals
-------------------------------------------

**Cross encoder reranking**

.. code-block:: python

    from retrievals import AutoModelForRanking

    sentences = [
        ["In 1974, I won the championship in Southeast Asia in my first kickboxing match", "In 1982, I defeated the heavy hitter Ryu Long."],
        ['A dog is chasing car.', 'A man is playing a guitar.'],
    ]
    model_name_or_path: str = "BAAI/bge-reranker-base"
    model = AutoModelForRanking.from_pretrained(model_name_or_path)
    scores_list = model.compute_score(sentences)
    print('Ranking score: ', scores_list)

.. code::

    Ranking score: [-5.075257778167725, -10.194067001342773]


**ColBERT reranking**

.. code-block:: python

    from retrievals import ColBERT

    sentences = [
        ["In 1974, I won the championship in Southeast Asia in my first kickboxing match", "In 1982, I defeated the heavy hitter Ryu Long."],
        ["In 1974, I won the championship in Southeast Asia in my first kickboxing match", "A man is playing a guitar."],
    ]
    model_name_or_path: str = 'BAAI/bge-m3'
    model = ColBERT.from_pretrained(
        model_name_or_path,
        colbert_dim=1024,
        use_fp16=True,
    )
    embeddings = model.encode(sentences[0], normalize_embeddings=True)
    print('Embedding shape: ', embeddings.shape)

    scores_list = model.compute_score(sentences)
    print('Ranking score: ', scores_list)

.. code::

    Embedding shape: (2, 21, 1024)
    Ranking score: [5.445939064025879, 3.0762712955474854]


**LLM generative reranking**

- `AutoModelForRanking.from_pretrained(model_name_or_path, causal_lm=True)`
- Prompt: "Given a query with a relevant body, determine whether the document is pertinent to the query by providing a prediction of either 'Yes' or 'No'."


.. code-block:: python

    from retrievals import LLMRanker

    model_name = 'BAAI/bge-reranker-v2-gemma'
    model = LLMRanker.from_pretrained(
                model_name,
                causal_lm=True,
                use_fp16=True,
            )

    scores = model.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
    print('Ranking score: ', scores)


2. Fine-tune cross-encoder reranking model
-----------------------------------------------

prepare data
    `{(query1, document1, label1), (query2, document2, label2), ...}`


.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1QvbUkZtG56SXomGYidwI4RQzwODQrWNm?usp=sharing
    :alt: Open In Colab


.. code-block:: python

    from transformers import AutoTokenizer, TrainingArguments, get_cosine_schedule_with_warmup, AdamW
    from retrievals import RerankCollator, AutoModelForRanking, RerankTrainer, RerankTrainDataset

    model_name_or_path: str = "BAAI/bge-reranker-base"
    max_length: int = 128
    learning_rate: float = 3e-5
    batch_size: int = 4
    epochs: int = 3
    output_dir: str = "./checkpoints"

    train_dataset = RerankTrainDataset("C-MTEB/T2Reranking", positive_key="positive", negative_key="negative", dataset_split='dev')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = AutoModelForRanking.from_pretrained(model_name_or_path)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_train_steps = int(len(train_dataset) / batch_size * epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.05 * num_train_steps,
        num_training_steps=num_train_steps,
    )

    training_args = TrainingArguments(
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        output_dir=output_dir,
        remove_unused_columns=False,
        logging_steps=100,
        report_to="none",
    )
    trainer = RerankTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=RerankCollator(tokenizer, max_length=max_length),
    )
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.train()


3. Fine-tune ColBERT reranking model
----------------------------------------

prepare data
    `{}`


.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1QVtqhQ080ZMltXoJyODMmvEQYI6oo5kO?usp=sharing
    :alt: Open In Colab

.. code-block:: python

    import os
    import transformers
    from transformers import (
        AdamW,
        AutoTokenizer,
        TrainingArguments,
        get_cosine_schedule_with_warmup,
    )

    from retrievals import ColBERT, ColBertCollator, RerankTrainer, RetrievalTrainDataset
    from retrievals.losses import ColbertLoss

    transformers.logging.set_verbosity_error()
    os.environ["WANDB_DISABLED"] = "true"

    model_name_or_path: str = "BAAI/bge-m3"
    learning_rate: float = 5e-6
    batch_size: int = 1
    epochs: int = 3
    colbert_dim: int = 1024
    output_dir: str = './checkpoints'

    train_dataset = RetrievalTrainDataset(
        'C-MTEB/T2Reranking', positive_key='positive', negative_key='negative', dataset_split='dev'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    data_collator = ColBertCollator(
        tokenizer,
        query_max_length=64,
        document_max_length=256,
        positive_key='positive',
        negative_key='negative',
    )
    model = ColBERT.from_pretrained(
        model_name_or_path,
        colbert_dim=colbert_dim,
        loss_fn=ColbertLoss(use_inbatch_negative=False),
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_train_steps = int(len(train_dataset) / batch_size * epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=0.05 * num_train_steps, num_training_steps=num_train_steps
    )

    training_args = TrainingArguments(
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        output_dir=output_dir,
        remove_unused_columns=False,
        logging_steps=100,
    )
    trainer = RerankTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.train()


4. Fine-tune LLM Generative reranker
-------------------------------------

prepare generative reranking data
    `{}`

prepare representative reranking data
    `{}`


.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1fzq1iV7-f8hNKFnjMmpVhVxadqPb9IXk?usp=sharing
    :alt: Open In Colab


- Point-wise style prompt:

    "Passage: {text}\nPlease write a question based on this passage."

- Point-wise style prompt:

    "Passage: {text}\nQuery: {query}\nDoes the passage answer the query? Answer 'Yes' or 'No'"

- pairwise style prompt:

    """Given a query "{query}", which of the following two passages is more relevant to the query?

    Passage A: "{doc1}"

    Passage B: "{doc2}"

    Output Passage A or Passage B:"""

- listwise style prompt:

    I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."

- set-wise style prompt:

    Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
    + passages + '\n\nOutput only the passage label of the most relevant passage:'


**Cross encoder reranking**

.. code-block:: shell

    MODEL_NAME="BAAI/bge-reranker-base"
    TRAIN_DATA="/t2_ranking.jsonl"
    OUTPUT_DIR="/t2_output"

    torchrun --nproc_per_node 1 \
      -m retrievals.pipelines.rerank \
      --output_dir $OUTPUT_DIR \
      --overwrite_output_dir \
      --model_name_or_path $MODEL_NAME \
      --model_type cross-encoder \
      --do_train \
      --data_name_or_path $TRAIN_DATA \
      --positive_key positive \
      --negative_key negative \
      --learning_rate 2e-5 \
      --fp16 \
      --num_train_epochs 3 \
      --per_device_train_batch_size 64 \
      --dataloader_drop_last True \
      --max_length 512 \
      --save_total_limit 1 \
      --logging_steps 100


**Colbert reranking**

.. code-block:: shell

    MODEL_NAME='hfl/chinese-roberta-wwm-ext'
    TRAIN_DATA="/t2_ranking.jsonl"
    OUTPUT_DIR="/t2_output"

    torchrun --nproc_per_node 1 \
      --module retrievals.pipelines.rerank \
      --output_dir $OUTPUT_DIR \
      --overwrite_output_dir \
      --model_name_or_path $MODEL_NAME \
      --tokenizer_name $MODEL_NAME \
      --model_type colbert \
      --do_train \
      --data_name_or_path $TRAIN_DATA \
      --positive_key positive \
      --negative_key negative \
      --learning_rate 1e-4 \
      --bf16 \
      --num_train_epochs 3 \
      --per_device_train_batch_size 64 \
      --dataloader_drop_last True \
      --max_length 256 \
      --train_group_size 4 \
      --unfold_each_positive false \
      --save_total_limit 1 \
      --logging_steps 100 \
      --use_inbatch_negative false


**LLM reranking**

.. code-block:: shell

    MODEL_NAME="Qwen/Qwen2-1.5B-Instruct"
    TRAIN_DATA="/t2_ranking.jsonl"
    OUTPUT_DIR="/t2_output"

    torchrun --nproc_per_node 1 \
        -m retrievals.pipelines.rerank \
        --output_dir ${OUTPUT_DIR} \
        --overwrite_output_dir \
        --model_name_or_path $MODEL_NAME \
        --model_type llm \
        --causal_lm True \
        --use_lora True \
        --data_name_or_path $TRAIN_DATA \
        --task_prompt "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'." \
        --query_instruction "A: " \
        --document_instruction 'B: ' \
        --positive_key positive \
        --negative_key negative \
        --learning_rate 2e-4 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 16 \
        --dataloader_drop_last True \
        --max_len 256 \
        --train_group_size 4 \
        --logging_steps 10 \
        --save_steps 20000 \
        --save_total_limit 1 \
        --bf16


Reference
-------------------

- https://github.com/ielab/llm-rankers/tree/main

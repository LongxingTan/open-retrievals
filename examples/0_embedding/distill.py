from datasets import load_dataset
from transformers import (
    AdamW,
    AutoTokenizer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from retrievals import (
    AutoModelForEmbedding,
    DistillTrainer,
    PairwiseModel,
    RetrievalCollator,
)

teacher_model_name = "intfloat/multilingual-e5-large"
student_model_name = "BAAI/bge-small-en"


train_dataset = load_dataset('shibing624/nli_zh', 'STS-B')['train']


teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, use_fast=False)
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, use_fast=False)

teacher_model = AutoModelForEmbedding.from_pretrained(teacher_model_name, pooling_method="mean")
student_model = AutoModelForEmbedding.from_pretrained(student_model_name, pooling_method="mean")

# Freeze teacher model
for param in teacher_model.parameters():
    param.requires_grad = False

batch_size = 32
epochs = 3
learning_rate = 5e-5

optimizer = AdamW(student_model.parameters(), lr=learning_rate)
num_train_steps = int(len(train_dataset) / batch_size * epochs)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0.05 * num_train_steps, num_training_steps=num_train_steps
)

training_args = TrainingArguments(
    output_dir='./distillation_checkpoints',
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    remove_unused_columns=False,
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=1000,
    load_best_model_at_end=True,
)


trainer = DistillTrainer(
    teacher_model=teacher_model,
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=RetrievalCollator(student_tokenizer, keys=['sentence1', 'sentence2'], max_lengths=[32, 128]),
)
trainer.optimizer = optimizer
trainer.scheduler = scheduler
trainer.train()

# Save the final student model
# student_model.save_pretrained('./distilled_model')
# student_tokenizer.save_pretrained('./distilled_model')

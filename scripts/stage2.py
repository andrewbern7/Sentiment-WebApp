#!/usr/bin/env python3
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1️⃣ Reproducibility
set_seed(42)  # fix random seeds for reproducibility

# 2️⃣ Paths & Hyperparameters
dataset_name   = "tweet_eval"
task_name      = "sentiment"
model          = "BertTweet/models/Sent140/checkpoint-5625"
output_dir     = "BertTweet/models/TweetEval"
num_labels     = 3          # classes: 0=neg,1=neu,2=pos
batch_size     = 32         # per‐device batch
epochs         = 4
lr_backbone    = 2e-5       # smaller LR for backbone
lr_head        = 2e-4       # larger LR for classification head
weight_decay   = 0.01       # regularization
warmup_ratio   = 0.1        # 10% warmup
max_length     = 128
grad_acc_steps = 2          # accumulate gradients for effective batch=64
num_workers    = 8

# 3️⃣ Load TweetEval sentiment split
raw_datasets = load_dataset(dataset_name, task_name)  # train/validation/test
# 4️⃣ Tokenization & Preprocessing
tokenizer = AutoTokenizer.from_pretrained(model)
def preprocess_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

tokenized = raw_datasets.map(
    preprocess_fn,
    batched=True,
    num_proc=num_workers,
    remove_columns=["text"]
)  # fast, parallel tokenization

# 5️⃣ Data Collator & Formatting
data_collator = DataCollatorWithPadding(tokenizer)
tokenized["train"].set_format("torch", columns=["input_ids","attention_mask","label"])
tokenized["validation"].set_format("torch", columns=["input_ids","attention_mask","label"])

# 6️⃣ Load Model
model = AutoModelForSequenceClassification.from_pretrained(
    model,
    num_labels=num_labels
)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# 7️⃣ TrainingArguments with warmup, scheduler, mixed precision
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=lr_backbone,
    weight_decay=weight_decay,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type="linear",
    gradient_accumulation_steps=grad_acc_steps,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True
)

# 8️⃣ Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    return {
        "eval_accuracy":  acc,
        "eval_precision": precision,
        "eval_recall":    recall,
        "eval_f1":        f1
    }

# 9️⃣ Trainer setup & launch
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 🔟 Train!
trainer.train()

# 1️⃣1️⃣ Final eval & save
metrics = trainer.evaluate()
print("Evaluation results:", metrics)
trainer.save_model(output_dir)
print(f"Model and tokenizer saved to {output_dir}")

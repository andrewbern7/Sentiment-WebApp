#!/usr/bin/env python3
import os
import pandas as pd
import torch
from datasets import Dataset
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
set_seed(42)

# 2️⃣ Paths & Hyperparameters
data_dir = os.path.expanduser('~/bert-training/bert_sentiment/data/tweeteval')
output_dir = os.path.expanduser('~/bert-training/bert_sentiment/models/basic/tweeteval_finetuned')
base_model_dir = os.path.expanduser('~/bert-training/bert_sentiment/models/bases/sent140_base')
num_labels = 3
batch_size = 512
epochs = 3 # Upped the epoch count as this introduces a new neutral class
learning_rate = 3e-5 #
weight_decay = 0.01
num_workers = 24

# 3️⃣ Load & preprocess data
df_train = pd.read_csv(f"{data_dir}/train.csv")
df_val = pd.read_csv(f"{data_dir}/val.csv")
df_train = df_train.rename(columns={'label': 'labels'})
df_val = df_val.rename(columns={'label': 'labels'})

train_ds = Dataset.from_pandas(df_train, preserve_index=False)
eval_ds = Dataset.from_pandas(df_val, preserve_index=False)

# 4️⃣ Tokenization
tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

def tokenize(batch):
    texts = [str(t) if isinstance(t, str) else "" for t in batch["text"]]
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_ds = train_ds.map(tokenize, batched=True, num_proc=4, remove_columns=['text'])
eval_ds = eval_ds.map(tokenize, batched=True, num_proc=4, remove_columns=['text'])

# 5️⃣ Format & Data Collator
train_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
eval_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
collator = DataCollatorWithPadding(tokenizer)

# 6️⃣ Load model from previously trained checkpoint
model = AutoModelForSequenceClassification.from_pretrained(base_model_dir, num_labels=num_labels)

# 7️⃣ Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    bf16=True,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='steps',
    logging_steps=100,
    dataloader_num_workers=num_workers,
    dataloader_persistent_workers=True,
    dataloader_prefetch_factor=16,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model='eval_accuracy',
    lr_scheduler_type="linear", #
    warmup_ratio=0.1, #
)

# 8️⃣ Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'eval_accuracy': acc,
        'eval_precision': precision,
        'eval_recall': recall,
        'eval_f1': f1
    }

# 9️⃣ Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics
)

# 🔟 Train & Save
trainer.train()
metrics = trainer.evaluate()
print("Manual eval returned:", metrics)
trainer.save_model(output_dir)
print(f"Model and tokenizer saved to {output_dir}")

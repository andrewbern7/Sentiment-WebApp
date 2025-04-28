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
data_path     = os.path.expanduser('~/bert-training/bert_sentiment/data/cleaned_sentiment140.csv')
output_dir    = 'BertTweet/models/Sent140'
model_name    = 'vinai/bertweet-large'
num_labels    = 3        # three-way head (0, 1, 2)
batch_size    = 256
epochs        = 3
learning_rate = 5e-5
weight_decay  = 0.01
num_workers   = 24
max_length    = 128

# 3️⃣ Load & preprocess data
df = pd.read_csv(data_path)
df = df.rename(columns={'target': 'labels'})

# 4️⃣ Build HF Dataset and split
full_ds = Dataset.from_pandas(df, preserve_index=False)
split   = full_ds.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = split['train'], split['test']

# 5️⃣ Tokenization
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize(batch):
    texts = [t if isinstance(t, str) else "" for t in batch["text"]]
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

train_ds = train_ds.map(
    tokenize,
    batched=True,
    num_proc=num_workers,
    remove_columns=['text']
)
eval_ds = eval_ds.map(
    tokenize,
    batched=True,
    num_proc=num_workers,
    remove_columns=['text']
)

# 6️⃣ Format & Data Collator
train_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
eval_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
collator = DataCollatorWithPadding(tokenizer)

# 7️⃣ Load & compile model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

# 8️⃣ Training arguments
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
    metric_for_best_model='eval_accuracy'
)

# 9️⃣ Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    return {
        'eval_accuracy': acc,
        'eval_precision': precision,
        'eval_recall': recall,
        'eval_f1': f1
    }

# 🔟 Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics
)

# 1️⃣1️⃣ Train & Save
trainer.train()
metrics = trainer.evaluate()
print("Manual eval returned:", metrics)
trainer.save_model(output_dir)
print(f"Model and tokenizer saved to {output_dir}")

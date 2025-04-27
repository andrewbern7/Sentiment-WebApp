#!/usr/bin/env python3
# DO NOT REPLICATE OFF OF THIS. THIS WAS THE FIRST MULTI-CLASSIFIER and is heavily modified
import os
import re
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 🚫 Suppress tokenizer fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1️⃣ Reproducibility
set_seed(42)

# 2️⃣ Paths & Hyperparameters
data_path      = os.path.expanduser('~/bert-training/bert_sentiment/data/goemotions/train.csv')
output_dir     = 'models/basic/goemotions_finetuned'
base_model_dir = os.path.expanduser('~/bert-training/bert_sentiment/models/bases/tweeteval_base')
num_labels     = 28
batch_size     = 512
epochs         = 6
learning_rate  = 5e-4
weight_decay   = 0.01
num_workers    = 24

# 3️⃣ Load & preprocess data
df = pd.read_csv(data_path)
df['labels'] = df['labels'].apply(lambda x: list(map(int, re.findall(r'\d+', str(x)))))

# Convert to HF Dataset
dataset = Dataset.from_pandas(df)

# 4️⃣ Tokenization & Label Binarization
tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

def tokenize_and_encode(batch):
    texts = [str(t) for t in batch["text"]]
    tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=128)
    # Multi-hot encode as float32
    binarized = torch.zeros((len(batch["labels"]), num_labels), dtype=torch.float32)
    for i, lbls in enumerate(batch["labels"]):
        for l in lbls:
            binarized[i][l] = 1.0
    tokenized["labels"] = binarized
    return tokenized

# Apply tokenization and encoding
dataset = dataset.map(tokenize_and_encode, batched=True, remove_columns=dataset.column_names)
dataset.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

# 5️⃣ Split the data
split = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = split['train'], split['test']

# --- Add custom collate function to return a dict ---
def collate_fn(batch):
    # batch: list of tuples (input_ids, attention_mask, labels)
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch]).float()
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# Build PyTorch DataLoaders with custom collate_fn to preserve mapping
train_loader = DataLoader(
    TensorDataset(
        torch.stack([ex['input_ids'] for ex in train_ds]),
        torch.stack([ex['attention_mask'] for ex in train_ds]),
        torch.stack([ex['labels'] for ex in train_ds])
    ),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=collate_fn
)

eval_loader = DataLoader(
    TensorDataset(
        torch.stack([ex['input_ids'] for ex in eval_ds]),
        torch.stack([ex['attention_mask'] for ex in eval_ds]),
        torch.stack([ex['labels'] for ex in eval_ds])
    ),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=collate_fn
)

# 6️⃣ Load model
model = AutoModelForSequenceClassification.from_pretrained(
    base_model_dir,
    problem_type="multi_label_classification",
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

# 7️⃣ TrainingArguments
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

# 8️⃣ Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.3).int().numpy()
    labels = labels.astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'eval_accuracy': acc,
        'eval_precision': precision,
        'eval_recall': recall,
        'eval_f1': f1
    }

# 9️⃣ Custom Trainer to use our DataLoaders
class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return train_loader

    def get_eval_dataloader(self, eval_dataset=None):
        return eval_loader

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics
)

# 🔟 Train & Save
trainer.train()
metrics = trainer.evaluate()
print("Manual eval returned:", metrics)
trainer.save_model(output_dir)
print(f"Model and tokenizer saved to {output_dir}")

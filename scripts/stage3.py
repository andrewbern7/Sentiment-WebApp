#!/usr/bin/env python3
import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict
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

# 2️⃣ Hyperparameters & Paths
goemotions_name = "go_emotions"
core_emotions   = ["anger", "joy", "sadness", "optimism"]
stage2_ckpt     = "BertTweet/models/TweetEval/checkpoint-1426"
output_dir      = "BertTweet/models/bertweet-core-emotion"
num_labels      = len(core_emotions)
batch_size      = 32
epochs          = 5
lr_backbone     = 2e-5
lr_head         = 2e-4
weight_decay    = 0.01
warmup_ratio    = 0.1
max_length      = 128
grad_acc_steps  = 1
num_workers     = 8

# 3️⃣ Load GoEmotions and filter to core labels
raw = load_dataset(goemotions_name)["train"]  # single split
label2id = {lbl: i for i, lbl in enumerate(raw.features["labels"].feature.names)}
core_ids = [label2id[e] for e in core_emotions]

def filter_and_relabel(example):
    orig = example["labels"]
    core_hot = [1 if idx in orig and idx in core_ids else 0 for idx in core_ids]
    return {"labels": core_hot}

filtered = raw.map(
    filter_and_relabel,
    batched=False,
    remove_columns=["labels"]
)

# 4️⃣ Random 80/10/10 split using HF methods (no stratification)
splits_tmp = filtered.train_test_split(test_size=0.2, seed=42)
val_test = splits_tmp["test"].train_test_split(test_size=0.5, seed=42)
splits = DatasetDict({
    "train":      splits_tmp["train"],
    "validation": val_test["train"],
    "test":       val_test["test"]
})

# 5️⃣ Tokenization
tokenizer = AutoTokenizer.from_pretrained(
    "vinai/bertweet-large",
    local_files_only=True
)

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

tokenized = splits.map(
    tokenize_fn,
    batched=True,
    num_proc=num_workers,
    remove_columns=["text"]
)

# 6️⃣ Formatting & Data Collator
for split in ["train", "validation"]:
    tokenized[split].set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
collator = DataCollatorWithPadding(tokenizer)

# 7️⃣ Load Stage 2 Checkpoint & Adapt Head
model = AutoModelForSequenceClassification.from_pretrained(
    stage2_ckpt,
    num_labels=num_labels,
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True,
    local_files_only=True
)
model.gradient_checkpointing_enable()

# 8️⃣ Optimizer with Exponential Layer-wise LR Decay
no_decay = ["bias", "LayerNorm.weight"]
def get_optimizer_params(model):
    layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
    params = []
    layer_decay = 0.9
    for i, layer in enumerate(reversed(layers)):
        lr = lr_backbone * (layer_decay ** i)
        for name, param in layer.named_parameters():
            params.append({
                "params": [param],
                "lr": lr,
                "weight_decay": 0.0 if any(nd in name for nd in no_decay) else weight_decay
            })
    for name, param in model.classifier.named_parameters():
        params.append({
            "params": [param],
            "lr": lr_head,
            "weight_decay": 0.0 if any(nd in name for nd in no_decay) else weight_decay
        })
    return params

optimizer = torch.optim.AdamW(get_optimizer_params(model))

# 9️⃣ TrainingArguments (with grad clipping)
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_acc_steps,
    learning_rate=lr_backbone,
    weight_decay=weight_decay,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type="cosine_with_restarts",
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_micro_f1",
    greater_is_better=True,
    max_grad_norm=1.0
)

# 1️⃣0️⃣ Composite & Focal Loss
from torch.nn import MultiLabelSoftMarginLoss

def focal_loss(logits, targets, gamma: float = 2.0):
    prob = torch.sigmoid(logits)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    return (ce * ((1 - p_t) ** gamma)).mean()

# 1️⃣1️⃣ Metrics
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.25).astype(int)
    y_true, y_pred = labels.flatten(), preds.flatten()
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="micro")
    acc = accuracy_score(y_true, y_pred)
    return {"eval_micro_f1": f1, "eval_precision": p, "eval_recall": r, "eval_accuracy": acc}

# 1️⃣2️⃣ Custom Trainer override loss
torch.autograd.set_detect_anomaly(True)
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        loss = MultiLabelSoftMarginLoss()(logits, labels) + focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 1️⃣3️⃣ Trainer & Train
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None)
)

trainer.train()
metrics = trainer.evaluate()
print("Stage 3 evaluation:", metrics)
trainer.save_model(output_dir)
print(f"Model and tokenizer saved to {output_dir}")
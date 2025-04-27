#!/usr/bin/env python3
# train_tweeteval_emotion_multitask.py

import os, torch
import pandas as pd
from pathlib import Path
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score

# 1️⃣ Reproducibility
os.environ["TOKENIZERS_PARALLELISM"] = "false"
set_seed(84)

# 2️⃣ Paths & Hyperparameters
clean_dir      = Path("data/tweeteval/emotions/clean")
output_dir     = "models/tweeteval_multitask"
base_sent_ckpt = "models/bases/tweeteval_base"  # your best sentiment checkpoint
num_sentiment_labels = 3  # negative / neutral / positive
num_emotion_labels   = 4  # anger, joy, sadness, optimism
batch_size     = 128
epochs         = 6
lr_backbone    = 2.5e-5
lr_head        = 2.5e-4
weight_decay   = 0.01
num_workers    = 8
threshold      = 0.3

# 3️⃣ Load cleaned emotion data
dfs = {split: pd.read_csv(clean_dir/f"{split}.csv") for split in ["train","val","test"]}
for split, df in dfs.items():
    # integer emotion label 0–3
    df["emotion_label"] = df["label"]  # from your cleaner script
    dfs[split] = df

# 4️⃣ Tokenizer & encoding
tokenizer = AutoTokenizer.from_pretrained(base_sent_ckpt)
def tokenize(batch):
    # Extract texts as a Python list (not a pandas Series or numpy array)
    texts = batch["text"]
    if not isinstance(texts, list):
        texts = texts.tolist()
    
    # Now safe to call the tokenizer
    tokens = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    # Single-label integer
    tokens["labels"] = batch["label"]
    return tokens


hf_dsets = {
    split: Dataset.from_pandas(df)
            .map(tokenize, batched=True, remove_columns=list(df.columns))
            .with_format("torch", columns=["input_ids","attention_mask","emotion_label"])
    for split, df in dfs.items()
}

train_loader = DataLoader(hf_dsets["train"], batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
eval_loader  = DataLoader(hf_dsets["val"],   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

# 5️⃣ Build Multi-Task Model
class MultiTaskBert(nn.Module):
    def __init__(self, sent_ckpt):
        super().__init__()
        # Shared encoder
        self.bert = AutoModel.from_pretrained(sent_ckpt)
        hidden = self.bert.config.hidden_size

        # Sentiment head: load from pretrained SentEval checkpoint
        sent_model = AutoModelForSequenceClassification.from_pretrained(
            sent_ckpt, num_labels=num_sentiment_labels
        )
        self.sent_head = nn.Linear(hidden, num_sentiment_labels)
        # copy pretrained weights
        self.sent_head.weight.data.copy_(sent_model.classifier.weight.data)
        self.sent_head.bias.data.copy_(  sent_model.classifier.bias.data  )

        # Emotion head: new randomly initialized head
        self.emot_head = nn.Linear(hidden, num_emotion_labels)

    def forward(self, input_ids, attention_mask=None):
        # BERT returns (last_hidden_state, pooler_output)
        pooled = self.bert(input_ids, attention_mask=attention_mask)[1]
        return {
            "sentiment_logits": self.sent_head(pooled),
            "emotion_logits":   self.emot_head(pooled)
        }

model = MultiTaskBert(base_sent_ckpt)

# 6️⃣ Optimizer & Scheduler
optimizer = AdamW([
    {"params": model.bert.parameters(),    "lr": lr_backbone},
    {"params": model.emot_head.parameters(),"lr": lr_head}
], weight_decay=weight_decay)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = int(0.1 * total_steps),
    num_training_steps= total_steps
)

# 7️⃣ TrainingArguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="emotion_micro_f1",
    greater_is_better=True
)

# 8️⃣ Custom Trainer
from torch.nn import CrossEntropyLoss

# sentiment loss (we do NOT re-train sentiment head on emotion data)
sent_loss_fn = CrossEntropyLoss()
# emotion loss
emot_loss_fn = CrossEntropyLoss()

class MTTrainer(Trainer):
    def get_train_dataloader(self):     return train_loader
    def get_eval_dataloader(self, ds): return eval_loader
    def create_optimizer_and_scheduler(self, _):
        return optimizer, scheduler

    def compute_loss(self, model, inputs, return_outputs=False):
        # pull out inputs
        labels_emo = inputs.pop("emotion_label")
        outputs = model(**inputs)
        sent_logits = outputs["sentiment_logits"]
        emot_logits = outputs["emotion_logits"]
        # loss for emotion only
        loss = emot_loss_fn(emot_logits, labels_emo)
        return (loss, outputs) if return_outputs else loss

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        # Trainer passes logits & labels from compute_loss outputs,
        # so unpack: logits=outputs, labels=labels_emo
        # but we need separate heads; override evaluation loop instead
        raise NotImplementedError("Metrics overridden below")

# We’ll handle metrics manually after training

trainer = MTTrainer(
    model=model,
    args=training_args,
    compute_metrics=None,  # we’ll evaluate after
)

if __name__=="__main__":
    trainer.train()
    # After training, run inference on val/test sets:
    def eval_split(dataloader):
        all_sent, all_emot, all_lab = [], [], []
        for batch in dataloader:
            with torch.no_grad():
                out = model(batch["input_ids"].to(model.bert.device),
                            attention_mask=batch["attention_mask"].to(model.bert.device))
            all_sent.append(out["sentiment_logits"].cpu())
            all_emot.append(out["emotion_logits"].cpu())
            all_lab.append(batch["emotion_label"])
        import numpy as np
        sent_logits = torch.cat(all_sent).numpy()
        emot_logits = torch.cat(all_emot).numpy()
        true_lab    = torch.cat(all_lab).numpy()
        # sentiment accuracy
        sent_preds   = sent_logits.argmax(axis=1)
        # emotion micro-F1
        emot_preds   = emot_logits.argmax(axis=1)
        emo_f1       = f1_score(true_lab, emot_preds, average="micro")
        sent_acc     = accuracy_score(true_lab, sent_preds)  # NOTE: here we only have emotion labels
        print(f"→ Emotion micro-F1: {emo_f1:.4f}")
        print(f"→ Sentiment acc (not trained): {sent_acc:.4f}")

    print("**Validation results**")
    eval_split(eval_loader)

    print("**Test results**")
    test_loader = DataLoader(hf_dsets["test"], batch_size=batch_size, shuffle=False)
    eval_split(test_loader)

    trainer.save_model(output_dir)

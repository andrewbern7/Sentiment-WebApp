#!/usr/bin/env python3
import os, re, pandas as pd, torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import BCEWithLogitsLoss, MultiLabelSoftMarginLoss
from torch.optim import AdamW
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import hamming_loss, f1_score

# 1️⃣ Reproducibility
os.environ["TOKENIZERS_PARALLELISM"] = "false"
set_seed(84)

# 2️⃣ Paths & Hyperparams
data_path      = os.path.expanduser('~/bert-training/bert_sentiment/data/goemotions/train.csv')
output_dir     = 'models/basic/goemotions_finetuned'
base_model_dir = os.path.expanduser('~/bert-training/bert_sentiment/models/bases/tweeteval_base')
num_labels, batch_size, epochs = 28, 512, 6
learning_rate, weight_decay = 2.5e-4, 0.01
num_workers = 24

# 3️⃣ Load & preprocess data
df = pd.read_csv(data_path)
df['labels'] = df['labels'].apply(lambda x: list(map(int, re.findall(r'\d+', str(x)))))

# ─────────── Step 1: Smoothed & capped class weights ───────────
N = len(df)                     # total samples
C = num_labels                  # number of classes  
max_w = 10.0                    # cap to prevent extreme weights

# count positives per class
pos_counts = [0]*C
for lbls in df['labels']:
    for l in lbls:
        pos_counts[l] += 1

# compute pos_weight_i = min(N / (C * pos_counts[i]), max_w) s
# Smoothing alogorithm for rarer labels
pos_weights = [
    min((N / (C * p)) if p > 0 else max_w, max_w)
    for p in pos_counts
]
pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32)

# Convert to HF Dataset
dataset = Dataset.from_pandas(df)

# 4️⃣ Tokenize & multi-hot encode
tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
def tokenize_and_encode(batch):
    texts = [str(t) for t in batch["text"]]
    tok = tokenizer(texts, truncation=True, padding="max_length", max_length=128)
    binarized = torch.zeros((len(batch["labels"]), num_labels), dtype=torch.float32)
    for i, lbls in enumerate(batch["labels"]):
        for l in lbls:
            binarized[i][l] = 1.0
    tok["labels"] = binarized
    return tok

dataset = dataset.map(tokenize_and_encode, batched=True, remove_columns=dataset.column_names)
dataset.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

# 5️⃣ Split
split = dataset.train_test_split(test_size=0.1, seed=84)
train_ds, eval_ds = split['train'], split['test']

# 6️⃣ Collate to dict
def collate_fn(batch):
    input_ids    = torch.stack([item[0] for item in batch])
    attention_ms = torch.stack([item[1] for item in batch])
    labels       = torch.stack([item[2] for item in batch]).float()
    return {'input_ids':input_ids, 'attention_mask':attention_ms, 'labels':labels}

train_loader = DataLoader(
    TensorDataset(
        torch.stack([ex['input_ids']    for ex in train_ds]),
        torch.stack([ex['attention_mask'] for ex in train_ds]),
        torch.stack([ex['labels']         for ex in train_ds])
    ),
    batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True,
    collate_fn=collate_fn
)
eval_loader = DataLoader(
    TensorDataset(
        torch.stack([ex['input_ids']    for ex in eval_ds]),
        torch.stack([ex['attention_mask'] for ex in eval_ds]),
        torch.stack([ex['labels']         for ex in eval_ds])
    ),
    batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True,
    collate_fn=collate_fn
)

# 7️⃣ Model
model = AutoModelForSequenceClassification.from_pretrained(
    base_model_dir,
    problem_type="multi_label_classification",
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

# 8️⃣ Optimizer & Scheduler (unchanged)
optimizer = AdamW([
    {'params': model.bert.parameters(),        'lr': learning_rate * 0.1},
    {'params': model.classifier.parameters(), 'lr': learning_rate},
], weight_decay=weight_decay)

total_steps = len(train_loader) * epochs
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = int(0.1 * total_steps),
    num_training_steps= total_steps
)  # 10% warm-up, then linear decay

# 9️⃣ TrainingArguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    bf16=True,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='steps',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='micro_f1',
    greater_is_better=True,
    max_grad_norm=1.0
)

# 🔟 Metrics & Trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.15).int().numpy()
    labels = labels.astype(int)
    return {
        'hamming_loss':  hamming_loss(labels, preds),
        'hamming_score': 1 - hamming_loss(labels, preds),
        'micro_f1':      f1_score(labels, preds, average='micro')
    }

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return train_loader
    def get_eval_dataloader(self, eval_dataset=None):
        return eval_loader
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        return self.optimizer, self.lr_scheduler
        
    # Weighted BCE (with pos_weight) + Soft Margin
    loss_fn_bce  = BCEWithLogitsLoss(pos_weight=pos_weights_tensor)
    loss_fn_soft = MultiLabelSoftMarginLoss()
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        device  = logits.device
    
        # Weighted BCE + Soft Margin
        weight     = pos_weights_tensor.to(device)
        loss_bce   = BCEWithLogitsLoss(pos_weight=weight)
        loss_soft  = MultiLabelSoftMarginLoss()
        loss       = loss_bce(logits, labels) + 0.5*loss_soft(logits, labels)
        return (loss, outputs) if return_outputs else loss


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    optimizers=(optimizer, scheduler)
)

trainer.train()
metrics = trainer.evaluate()
print("Final metrics:", metrics)
trainer.save_model(output_dir)

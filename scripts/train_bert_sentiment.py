#!/usr/bin/env python3
import pandas as pd
import torch
import logging
import os
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    DataCollatorWithPadding,
    set_seed
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from multiprocessing import cpu_count

# === Debug & Performance-Enhanced Training Script (Optimized) ===
# - In-memory caching, BF16, gradient checkpointing, persistent workers,
#   data prefetch, dynamic padding, and eager compilation.

# 1️⃣ Seed and Logging Setup
set_seed(42)
logs_dir = '/home/ubuntu/bert-training/bert_sentiment/logs'
cache_dir = '/home/ubuntu/bert-training/bert_sentiment/data/cache'
model_dir = '/home/ubuntu/bert-training/bert_sentiment/models'
for d in [logs_dir, cache_dir, model_dir]: os.makedirs(d, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(logs_dir, 'train_debug.log'),
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logging.info("=== Optimized Debug Training Script Started ===")

# 2️⃣ System Information
logging.info(f"CPU cores available: {cpu_count()}")
try:
    import psutil
    logging.info(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
except ImportError:
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemTotal'):
                logging.info(f"Total RAM: {int(line.split()[1])/1024/1024:.2f} GB")
gpu_count = torch.cuda.device_count()
logging.info(f"GPU devices detected: {gpu_count}")
for i in range(gpu_count):
    logging.info(f" - GPU {i}: {torch.cuda.get_device_name(i)}")

# 3️⃣ Enable TF32 & cuDNN autotune
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# 4️⃣ Initialize Tokenizer
logging.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 5️⃣ Data Loading & Preprocessing
data_path = '/home/ubuntu/bert-training/bert_sentiment/data/training.1600000.processed.noemoticon.csv'
if os.path.isdir(os.path.join(cache_dir, 'train')):
    logging.info("Loading tokenized dataset from cache...")
    train_ds = load_from_disk(os.path.join(cache_dir, 'train'))
    eval_ds  = load_from_disk(os.path.join(cache_dir, 'eval'))
else:
    logging.info("Reading raw CSV and preprocessing...")
    df = pd.read_csv(
        data_path,
        encoding='latin-1',
        header=None,
        usecols=[0,5],
        names=['target','text']
    )
    df['target'] = df['target'].map({0:0, 2:1, 4:2})
    df = df.rename(columns={'target':'labels'})
    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.shuffle(seed=42)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split['train'], split['test']

    num_proc = cpu_count()
    logging.info(f"Tokenizing with {num_proc} processes and keeping in memory...")
    def tokenize_fn(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length')
    train_ds = train_ds.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        keep_in_memory=True,
        remove_columns=['text']
    )
    eval_ds = eval_ds.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        keep_in_memory=True,
        remove_columns=['text']
    )
    train_ds.save_to_disk(os.path.join(cache_dir, 'train'))
    eval_ds.save_to_disk(os.path.join(cache_dir, 'eval'))

# 6️⃣ Dataset Formatting
train_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
eval_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
data_collator = DataCollatorWithPadding(tokenizer)

# 7️⃣ Model Loading & Compilation
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
try:
    model = torch.compile(model, backend='eager')
    logging.info("Model compiled with eager backend for stability")
except Exception:
    logging.info("torch.compile unavailable, continuing without it.")

# 8️⃣ Metrics Function
def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
    acc = accuracy_score(p.label_ids, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

# 9️⃣ Training Arguments (Optimized for A100)
training_args = TrainingArguments(
    output_dir=model_dir,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir=logs_dir,
    logging_steps=100,
    save_total_limit=2,
    num_train_epochs=3,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=5e-5,
    weight_decay=0.01,
    bf16=True,
    gradient_checkpointing=True,
    dataloader_num_workers=16,
    dataloader_persistent_workers=True,
    dataloader_prefetch_factor=8,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    report_to=[]
)
logging.info(f"Training config: {training_args}")

# 🔟 Trainer Setup & Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
logging.info("Starting training...")
trainer.train()
logging.info("Training complete.")

# 1️⃣1️⃣ Save & Evaluate
trainer.save_model(os.path.join(model_dir, 'final_model'))
metrics = trainer.evaluate()
logging.info(f"Evaluation metrics: {metrics}")
with open(os.path.join(model_dir, 'final_metrics.txt'), 'w') as f:
    for k, v in metrics.items(): f.write(f"{k}: {v:.4f}\n")
logging.info("Metrics saved and script finished.")

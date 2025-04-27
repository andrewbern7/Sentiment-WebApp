# goemotions_setup.py
# Step 1: Download and clean GoEmotions dataset with label formatting like [17, 15]

import os
import re
import string
import pandas as pd
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Setup
nltk.download('punkt')
nltk.download('stopwords')

# Constants
OUTPUT_DIR = os.path.expanduser('~/bert-training/bert_sentiment/data/goemotions')
os.makedirs(OUTPUT_DIR, exist_ok=True)
STOPWORDS = set(stopwords.words('english'))
MENTION_PATTERN = r"@\w+"
HASHTAG_PATTERN = r"#\w+"
LINK_PATTERN = r"https?://\S+|www\.\S+"

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(LINK_PATTERN, '', text)
    text = re.sub(MENTION_PATTERN, '', text)
    text = re.sub(HASHTAG_PATTERN, '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in STOPWORDS and word.isalpha()]
    return ' '.join(tokens)

# Convert integer list to string for CSV
def listify_labels(example):
    example['labels'] = str(example['labels'])
    return example

# Load and process GoEmotions dataset
print("Downloading GoEmotions dataset...")
dataset = load_dataset("go_emotions")

print("Cleaning text and formatting labels...")
for split in ['train', 'validation', 'test']:
    dataset[split] = dataset[split].map(lambda ex: {'text': clean_text(ex['text'])})
    dataset[split] = dataset[split].map(listify_labels)
    df = pd.DataFrame(dataset[split])
    df.to_csv(os.path.join(OUTPUT_DIR, f"{split}.csv"), index=False)

print("✅ GoEmotions cleaned and saved to:", OUTPUT_DIR)

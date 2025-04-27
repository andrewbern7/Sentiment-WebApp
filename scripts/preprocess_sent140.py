#!/usr/bin/env python3
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os
nltk.download('punkt')
nltk.download('stopwords')

# File paths
RAW_CSV = os.path.expanduser("~/bert-training/bert_sentiment/data/training.1600000.processed.noemoticon.csv")
CLEAN_CSV = os.path.expanduser("~/bert-training/bert_sentiment/data/cleaned_sentiment140.csv")

# Load raw data
print("Loading raw Sentiment140 dataset...")
df = pd.read_csv(
    RAW_CSV,
    encoding='latin-1',
    header=None,
    names=["target", "id", "date", "flag", "user", "text"]
)

# We're only using sentiment and text columns for training
df = df[["target", "text"]]

# Keep only 0 and 4 classes, map them into 0 and 2
# Reserve 1 for future neutral sentiment
print("Remapping target labels...")
df = df[df['target'].isin([0, 4])]
df['target'] = df['target'].map({0: 0, 4: 2})

# Compile regex patterns
MENTION_PATTERN = r"@\w+"
HASHTAG_PATTERN = r"#\w+"
LINK_PATTERN = r"https?://\S+|www\.\S+"
STOPWORDS = set(stopwords.words('english'))

# Cleaning function
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove links, mentions, hashtags
    text = re.sub(LINK_PATTERN, '', text)
    text = re.sub(MENTION_PATTERN, '', text)
    text = re.sub(HASHTAG_PATTERN, '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in STOPWORDS and word.isalpha()]
    return ' '.join(tokens)

print("Cleaning tweets...")
df['text'] = df['text'].apply(clean_text)

print(f"Saving cleaned dataset to: {CLEAN_CSV}")
df.to_csv(CLEAN_CSV, index=False)
print("✅ Cleaning complete.")

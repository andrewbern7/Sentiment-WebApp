#!/usr/bin/env python3
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pathlib import Path

nltk.download("punkt")
nltk.download("stopwords")

MENTION_PATTERN = r"@\w+"
HASHTAG_PATTERN = r"#\w+"
LINK_PATTERN = r"https?://\S+|www\.\S+"
STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(LINK_PATTERN, "", text)
    text = re.sub(MENTION_PATTERN, "", text)
    text = re.sub(HASHTAG_PATTERN, "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in STOPWORDS and word.isalpha()]
    return " ".join(tokens)

# Paths
raw_path = Path("../data/raw_datasets/tweeteval")
cleaned_path = Path("../data/tweeteval")
cleaned_path.mkdir(parents=True, exist_ok=True)

for split in ["train", "val", "test"]:
    df = pd.read_csv(raw_path / f"{split}.csv")
    df["text"] = df["text"].astype(str).map(clean_text)
    df.to_csv(cleaned_path / f"{split}.csv", index=False)

print("Cleaned TweetEval saved to data/tweeteval/")
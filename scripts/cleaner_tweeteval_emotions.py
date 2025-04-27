#!/usr/bin/env python3
# cleaner_tweeteval_emotions.py

import re, string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import nlpaug.augmenter.word as naw  # for EDA swap augmentation

# 0. Download stopwords once
nltk.download("stopwords")

# 1. Patterns & tokenizer
MENTION_PATTERN = r"@\w+"
HASHTAG_PATTERN = r"#\w+"
LINK_PATTERN    = r"https?://\S+|www\.\S+"
STOPWORDS       = set(stopwords.words("english"))
tknzr           = TweetTokenizer(strip_handles=True, reduce_len=True)

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(LINK_PATTERN,    "", text)
    text = re.sub(MENTION_PATTERN, "", text)
    text = re.sub(HASHTAG_PATTERN, "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = tknzr.tokenize(text)
    tokens = [w for w in tokens if w.isalpha() and w not in STOPWORDS]
    return " ".join(tokens)

# 2. EDA augmenter
eda = naw.RandomWordAug(action="swap")

# 3. File paths & params
raw_dir      = "data/tweeteval/emotions"
clean_dir    = "data/tweeteval/emotion"
splits       = ["train", "val", "test"]
augment_frac = 0.2  # 20% augmentation

# 4. Label mapping
label_map = {
    0: "anger",
    1: "joy",
    2: "sadness",
    3: "optimism"
}
emotion_cols = list(label_map.values())

for split in splits:
    # a) load raw CSV with columns: text,label
    df = pd.read_csv(f"{raw_dir}/{split}.csv")
    
    # b) clean the text
    df["text"] = df["text"].astype(str).map(clean_text)
    
    # c) map numeric label → emotion string
    df["emotion"] = df["label"].map(label_map)
    
    # d) one-hot encode the four emotions
    one_hot = pd.get_dummies(df["emotion"])[emotion_cols]
    df = pd.concat([df, one_hot], axis=1)
    
    # e) compute per-emotion frequencies & find minority labels
    freqs = df[emotion_cols].sum()
    median_freq = freqs.median()
    minority_labels = freqs[freqs < median_freq].index.tolist()
    
    # f) select rows containing any minority emotion, augment 20% of them
    mask = df[minority_labels].sum(axis=1) > 0
    to_augment = df[mask]
    aug_samples = to_augment.sample(frac=augment_frac, random_state=42)
    aug_texts = aug_samples["text"].map(lambda t: eda.augment(t))
    aug_df = aug_samples.copy()
    aug_df["text"] = aug_texts
    
    # g) concatenate augmented examples and save
    df = pd.concat([df, aug_df], ignore_index=True)
    df.to_csv(f"{clean_dir}/{split}.csv", index=False)
    
    print(f"→ {split}: {len(df)} samples (including {len(aug_df)} augmented)")

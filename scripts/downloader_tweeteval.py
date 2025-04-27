#!/usr/bin/env python3
from datasets import load_dataset
import pandas as pd
from pathlib import Path

output_dir = Path("data/tweeteval/emotions/")
output_dir.mkdir(parents=True, exist_ok=True)

# Load TweetEval sentiment dataset
dataset = load_dataset("tweet_eval", "emotion")

# Save splits to CSV
dataset["train"].to_pandas().to_csv(output_dir / "train.csv", index=False)
dataset["validation"].to_pandas().to_csv(output_dir / "val.csv", index=False)
dataset["test"].to_pandas().to_csv(output_dir / "test.csv", index=False)

print("TweetEval raw CSVs saved to data/tweeteval/emtotions")

import json
from transformers import pipeline

# 1) Load both pipelines once at startup
sentiment = pipeline(
    "sentiment-analysis",
    model="models/TweetEval/checkpoint-1426",
    tokenizer="models/TweetEval/checkpoint-1426",
    device=-1  # Force CPU use
)
emotions = pipeline(
    "text-classification",
    model="models/bertweet-core-emotion/checkpoint-2172",
    tokenizer="models/bertweet-core-emotion/checkpoint-2172",
    return_all_scores=True,
    device=-1
)

def main():
    # prompt user for input
    text = input("Enter a sentence to analyze: ").strip()

    if not text:
        print("No input provided.")
        return

    # run inference
    sent = sentiment(text)[0]
    emo = emotions(text)[0]

    # emit combined JSON
    result = {
        "sentiment": sent,
        "emotions": emo
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

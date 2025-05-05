import json
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Label mapping for emotions and sentiment
emotion_labels = {
    "LABEL_0": "anger",
    "LABEL_1": "joy",
    "LABEL_2": "sadness",
    "LABEL_3": "fear"
}

sentiment_labels = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

# Load pipelines with CPU-only
sentiment = pipeline(
    "sentiment-analysis",
    model="models/TweetEval/checkpoint-1426",
    tokenizer="models/TweetEval/checkpoint-1426",
    device=-1
)
emotions = pipeline(
    "text-classification",
    model="models/bertweet-core-emotion/checkpoint-2172",
    tokenizer="models/bertweet-core-emotion/checkpoint-2172",
    return_all_scores=True,
    device=-1
)

def main():
    text = input("Enter a sentence to analyze: ").strip()

    if not text:
        print("No input provided.")
        return

    # Run sentiment analysis
    sent = sentiment(text)[0]
    sent["label"] = sentiment_labels.get(sent["label"], sent["label"])

    # Run emotion classification and decode labels
    raw_emotions = emotions(text)[0]
    decoded_emotions = [
        {
            "emotion": emotion_labels.get(item["label"], item["label"]),
            "score": item["score"]
        }
        for item in raw_emotions
    ]

    decoded_emotions.sort(key=lambda x: x["score"], reverse=True)

    # Output results
    result = {
        "sentiment": sent,
        "emotions": decoded_emotions
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

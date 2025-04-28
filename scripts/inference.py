import sys, json
from transformers import pipeline

# 1) Load both pipelines once at startup
sentiment = pipeline(
    "sentiment-analysis",
    model="models/TweetEval/checkpoint-1426",
    tokenizer="models/TweetEval/checkpoint-1426",
)
emotions = pipeline(
    "text-classification",
    model="models/bertweet-emotions",
    tokenizer="models/bertweet-emotions",
    return_all_scores=True,
)

def main():
    # read {"text": "..."} from stdin
    data = json.load(sys.stdin)
    text = data.get("text", "")
    # run inference
    sent = sentiment(text)[0]
    emo = emotions(text)[0]
    # emit combined JSON
    print(json.dumps({
        "sentiment": sent,
        "emotions": emo
    }))

if __name__ == "__main__":
    main()

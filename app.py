from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains by default

# Label mappings
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

# Load models
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="models/TweetEval/checkpoint-1426",
    tokenizer="models/TweetEval/checkpoint-1426",
    device=-1
)

emotion_pipeline = pipeline(
    "text-classification",
    model="models/bertweet-core-emotion/checkpoint-2172",
    tokenizer="models/bertweet-core-emotion/checkpoint-2172",
    return_all_scores=True,
    device=-1
)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    # Run sentiment
    sent = sentiment_pipeline(text)[0]
    sent["label"] = sentiment_labels.get(sent["label"], sent["label"])

    # Run emotions
    raw_emotions = emotion_pipeline(text)[0]
    decoded_emotions = [
        {
            "emotion": emotion_labels.get(item["label"], item["label"]),
            "score": item["score"]
        }
        for item in raw_emotions
    ]
    decoded_emotions.sort(key=lambda x: x["score"], reverse=True)

    return jsonify({
        "sentiment": sent,
        "emotions": decoded_emotions
    })

if __name__ == "__main__":
    app.run(debug=True)

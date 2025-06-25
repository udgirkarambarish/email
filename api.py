from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Constants
MAX_LEN = 200

# Load model
model = load_model("superImprovedDL.keras")

# Load or create tokenizer and label encoder
def load_or_create_tokenizer_and_labels():
    if (not os.path.exists("tokenizer.json") or os.path.getsize("tokenizer.json") == 0 or
        not os.path.exists("label_encoder.npy") or os.path.getsize("label_encoder.npy") == 0):

        def load_and_label_data(filepath, label):
            df = pd.read_csv(filepath)
            df['label'] = label
            return df

        dataframes = [
            load_and_label_data("dataset/preprocessed_spam.csv", "spam"),
            load_and_label_data("dataset/preprocessed_promotions.csv", "promotions"),
            load_and_label_data("dataset/preprocessed_updates.csv", "updates"),
            load_and_label_data("dataset/preprocessed_social.csv", "social"),
        ]

        df = pd.concat(dataframes, ignore_index=True)
        df['text'] = df['Subject'].astype(str) + " " + df['Body'].astype(str)

        # Tokenizer creation assumes data is already preprocessed
        tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
        tokenizer.fit_on_texts(df['text'])
        with open("tokenizer.json", "w") as f:
            f.write(tokenizer.to_json())

        label_encoder = LabelEncoder()
        label_encoder.fit(df['label'])
        np.save("label_encoder.npy", label_encoder.classes_)
        label_classes = label_encoder.classes_
    else:
        with open("tokenizer.json") as f:
            tokenizer = tokenizer_from_json(f.read())
        label_classes = np.load("label_encoder.npy", allow_pickle=True)

    return tokenizer, label_classes

# Load tokenizer and label classes
tokenizer, label_classes = load_or_create_tokenizer_and_labels()

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "Please provide 'text' field"}), 400

    text = data["text"]
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)

    prediction = model.predict(padded)
    class_index = np.argmax(prediction)
    predicted_label = label_classes[class_index]
    confidence = float(np.max(prediction))

    return jsonify({
        "label": str(predicted_label),
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

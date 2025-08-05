# Author: Sean Sjahrial
# Title: Cybersecurity RAG Assistant – Predict CLI (Baseline Classifier)
# Description: Predicts attack phase labels from pentest report text using a trained LinearSVC model.
# GitHub: https://github.com/isnakie
# License: MIT

# ------------------------------------------------------------------------------
# This CLI tool is part of the legacy supervised learning branch of the project.
# It loads a TF-IDF vectorizer and LinearSVC model to predict the most likely
# attack phase or label given a raw input sentence from a penetration test report.
# ------------------------------------------------------------------------------

import joblib
import json
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Load trained model and vectorizer ---
model = joblib.load("models/LinearSVC_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# --- Load label mapping ---
with open("models/label_map.json") as f:
    label_map = json.load(f)

# --- Invert map for easy lookup ---
id_to_label = {int(k): v for k, v in label_map.items()}

def predict(text):
    """Vectorize the input text and return the model’s predicted label."""
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return prediction

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(":: Missing input text.")
        print("Usage: python predict.py \"The attacker used PowerShell to disable antivirus.\"")
        sys.exit(1)

    input_text = sys.argv[1]
    pred_label = predict(input_text)
    print(f"\n:: Predicted label → {pred_label}")

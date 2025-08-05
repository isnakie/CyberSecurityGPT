import joblib
import json
import sys

from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
model = joblib.load("models/LinearSVC_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Load label map
with open("models/label_map.json") as f:
    label_map = json.load(f)

# Invert label map to get ID → Label
id_to_label = {int(k): v for k, v in label_map.items()}

def predict(text):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return prediction

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️  Please provide a text string as an argument.")
        print("Usage: python predict.py \"The attacker used PowerShell to disable antivirus.\"")
        sys.exit(1)

    input_text = sys.argv[1]
    pred_label = predict(input_text)
    print(f"\n📌 Predicted label: {pred_label}")

# train_baseline_model.py
# Author: Sean Sjahrial
# Title: Cybersecurity RAG Assistant
# Description: Part of UC Berkeley MICS Machine Learning Course (2025)
# GitHub: https://github.com/isnakie
# Description: Train and evaluate a baseline text classification model on pentest data
# License: MIT

import joblib
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ------------------------------------------------------------------------------
# Load vectorized training/test data
# ------------------------------------------------------------------------------

X_train = joblib.load("data/processed/X_train.pkl")
X_test = joblib.load("data/processed/X_test.pkl")
y_train = joblib.load("data/processed/y_train.pkl")
y_test = joblib.load("data/processed/y_test.pkl")

# ------------------------------------------------------------------------------
# Train model
# ------------------------------------------------------------------------------

# Available options:
# clf = LogisticRegression(max_iter=1000)
# clf = MultinomialNB()
clf = LinearSVC(class_weight="balanced")

clf.fit(X_train, y_train)

# ------------------------------------------------------------------------------
# Evaluate model
# ------------------------------------------------------------------------------

y_pred = clf.predict(X_test)

print("Model training complete.")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ------------------------------------------------------------------------------
# Save model
# ------------------------------------------------------------------------------

joblib.dump(clf, "models/LinearSVC_model.pkl")
print("Model saved to models/LinearSVC_model.pkl")

# ------------------------------------------------------------------------------
# Log misclassified samples
# ------------------------------------------------------------------------------

test_texts = joblib.load("data/processed/X_test_texts.pkl")
misclassified_idx = np.where(y_pred != y_test)[0]

misclassified = pd.DataFrame({
    "text": [test_texts[i] for i in misclassified_idx],
    "true_label": [y_test.iloc[i] for i in misclassified_idx],
    "predicted_label": [y_pred[i] for i in misclassified_idx]
})

misclassified.to_csv("logs/misclassified_samples.csv", index=False)
print("Misclassified samples saved to logs/misclassified_samples.csv")

# ------------------------------------------------------------------------------
# Save label map
# ------------------------------------------------------------------------------

encoder = LabelEncoder()
encoder.fit(y_train)

label_map = {i: label for i, label in enumerate(encoder.classes_)}
with open("models/label_map.json", "w") as f:
    json.dump(label_map, f, indent=4)

print("Label map saved to models/label_map.json")

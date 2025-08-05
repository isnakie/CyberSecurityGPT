# Author: Sean Sjahrial
# Title: Cybersecurity RAG Assistant – TF-IDF Vectorizer
# Description: Converts labeled pentest report text into TF-IDF vectors for classification models.
# GitHub: https://github.com/isnakie
# License: MIT

# ------------------------------------------------------------------------------
# This script vectorizes pre-cleaned text from penetration test reports using a
# TF-IDF pipeline. It supports model training and inference for the legacy
# classifier component of the project. Saved files are used downstream by the
# supervised ML baseline.
# ------------------------------------------------------------------------------

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

print(":: Vectorization script started")

# --- Load processed train/test data ---
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# --- Initialize TF-IDF Vectorizer ---
vectorizer = TfidfVectorizer(
    max_features=10000,       # Limit vocab size
    ngram_range=(1, 3),       # Use unigrams, bigrams, and trigrams
    min_df=2,                 # Ignore very rare terms
    stop_words="english"      # Remove common stopwords
)

# --- Vectorize text data ---
X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])

# --- Save features and labels ---
joblib.dump(X_train, "data/processed/X_train.pkl")
joblib.dump(X_test, "data/processed/X_test.pkl")
joblib.dump(train_df["label"], "data/processed/y_train.pkl")
joblib.dump(test_df["label"], "data/processed/y_test.pkl")

# --- Save the vectorizer and test text for review/debugging ---
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(test_df["text"].tolist(), "data/processed/X_test_texts.pkl")

print(":: ✅ Text vectorized and saved to disk.")

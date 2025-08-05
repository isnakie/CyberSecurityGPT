# vectorize_data.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
print("test")
# Load split data
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# Initialize the vectorizer
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    min_df=2,  # ignore rare terms
    stop_words='english'
    )

# Fit on training data and transform both train and test
X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])

# Save transformed data and vectorizer
joblib.dump(X_train, "data/processed/X_train.pkl")
joblib.dump(X_test, "data/processed/X_test.pkl")
joblib.dump(train_df['label'], "data/processed/y_train.pkl")
joblib.dump(test_df['label'], "data/processed/y_test.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# Save original test text for later inspection
joblib.dump(test_df['text'].tolist(), "data/processed/X_test_texts.pkl")


print("✅ Text successfully vectorized and saved.")
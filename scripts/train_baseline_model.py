import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json


# Load vectorized data and labels
X_train = joblib.load("data/processed/X_train.pkl")
X_test = joblib.load("data/processed/X_test.pkl")
y_train = joblib.load("data/processed/y_train.pkl")
y_test = joblib.load("data/processed/y_test.pkl")

# Train the classifier
#clf = LogisticRegression(max_iter=1000) # Logistic Regression
#clf = MultinomialNB() # Multinomial Naive Bayes
clf = LinearSVC(class_weight='balanced') # Linear SVC

clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print("✅ Model trained.")
print("🔍 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(clf, "models/LinearSVC_model.pkl")
print("💾 Model saved to models/LinearSVC_model.pkl")

# Further analysis of model
# Load original test text (must be saved in vectorize_data.py)
test_texts = joblib.load("data/processed/X_test_texts.pkl")

# Log misclassified samples
misclassified_idx = np.where(y_pred != y_test)[0]

misclassified = pd.DataFrame({
    "text": [test_texts[i] for i in misclassified_idx],
    "true_label": [y_test.iloc[i] for i in misclassified_idx],
    "predicted_label": [y_pred[i] for i in misclassified_idx]
})

misclassified.to_csv("logs/misclassified_samples.csv", index=False)
print("📄 Misclassified samples saved to logs/misclassified_samples.csv")

# Further Decoding
# Create and fit label encoder
encoder = LabelEncoder()
encoder.fit(y_train)

# Save label map
label_map = {i: label for i, label in enumerate(encoder.classes_)}
with open("models/label_map.json", "w") as f:
    json.dump(label_map, f, indent=4)

print("📘 Label map saved to models/label_map.json")
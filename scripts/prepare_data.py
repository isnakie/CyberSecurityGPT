# prepare_data.py
print("Script started")
import pandas as pd
import re
from sklearn.model_selection import train_test_split
print("Loading CSV...")
# Load the CSV
df = pd.read_csv("G:\Security Tools\- Project Talos\pentest_nlp_project\data\processed\Larger_Group_Labels.csv")

# Basic text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
    return text

df["clean_text"] = df["text"].astype(str).apply(clean_text)

#num_labels = df['label'].nunique()
#test_size = max(0.2, num_labels / len(df))  # ensure each class is represented
# Split into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Save the processed sets
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("✅ Preprocessing and splitting complete.")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
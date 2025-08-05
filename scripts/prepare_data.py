# Author: Sean Sjahrial
# Title: Cybersecurity RAG Assistant – Data Preparation
# Description: This script preprocesses and splits labeled pentest report text for use in supervised ML tasks.
# GitHub: https://github.com/isnakie
# License: MIT

# ------------------------------------------------------------------------------
# This script is part of the legacy pentest report classifier. It loads labeled
# findings from a CSV, cleans the text, and splits the dataset into train/test.
# While the project has pivoted to RAG, this pipeline may be used to re-integrate
# classification alongside retrieval for future hybrid models.
# ------------------------------------------------------------------------------

import pandas as pd
import re
from sklearn.model_selection import train_test_split

print(":: Script started")

# --- Load raw labeled findings ---
print(":: Loading CSV data ...")
df = pd.read_csv("data/processed/Larger_Group_Labels.csv")  # Update path if needed

# --- Text cleaning ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)       # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text

df["clean_text"] = df["text"].astype(str).apply(clean_text)

# --- Train/test split ---
print(":: Splitting into train/test sets ...")
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# --- Save to disk ---
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

# --- Done ---
print(":: ✅ Preprocessing and split complete.")
print(f":: Train shape: {train_df.shape}")
print(f":: Test shape:  {test_df.shape}")

import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# --- File Paths ---
index_path = "data/cyber_threats/mitre_faiss.index"
metadata_path = "data/cyber_threats/index_metadata.pkl"

# --- Load FAISS and metadata ---
print("📂 Loading FAISS index and metadata...")
index = faiss.read_index(index_path)
with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

# --- Load SentenceTransformer model ---
print("🔍 Loading embedding model...")
model = SentenceTransformer("all-mpnet-base-v2")  # Must match the one used for indexing

# --- Query loop ---
def query_loop():
    print("\n💬 Ask a cybersecurity question (or type 'exit'):\n")
    while True:
        user_input = input("🧠> ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        query_vec = model.encode([user_input])
        D, I = index.search(np.array(query_vec), k=3)  # top-3 matches

        print("\n🔎 Top Matches:\n")
        for rank, idx in enumerate(I[0]):
            entry = metadata[idx]
            print(f"#{rank + 1}: {entry['id']} — {entry['metadata']['name']}")
            print(f"   🔗 CWE ID: {entry['metadata']['cwe_id']}")
            print(f"   📘 Excerpt:\n{entry['text'][:500]}...\n")
            print("-" * 80)

if __name__ == "__main__":
    query_loop()

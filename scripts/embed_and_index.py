# Author: Sean Sjahrial
# Title: Cybersecurity RAG Assistant – MITRE FAISS Index Builder
# Description: Embeds MITRE CWE documents and builds a FAISS index with associated metadata.
# GitHub: https://github.com/isnakie
# License: MIT

# ------------------------------------------------------------------------------
# This legacy script converts a MITRE CWE JSONL knowledge base into a FAISS index.
# It uses Sentence-Transformers for vector embeddings, stores results as .index and
# .pkl files, and enables fast semantic search of CWE documents.
# ------------------------------------------------------------------------------

import json
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- File paths ---
jsonl_path = "data/cyber_threats/mitre_cwe_knowledge_base.jsonl"
index_path = "data/cyber_threats/mitre_faiss.index"
metadata_path = "data/cyber_threats/index_metadata.pkl"

# --- Load MITRE CWE entries from JSONL ---
documents = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        documents.append(doc)

# --- Load embedding model ---
print(":: Loading sentence transformer model ...")
model = SentenceTransformer("all-mpnet-base-v2")

# --- Generate vector embeddings ---
print(":: Embedding MITRE CWE documents ...")
texts = [doc["content"] for doc in documents]
embeddings = model.encode(texts, show_progress_bar=True)

# --- Build FAISS index from embeddings ---
print(":: Building FAISS index ...")
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# --- Save index and metadata ---
print(":: Saving FAISS index and metadata ...")
os.makedirs(os.path.dirname(index_path), exist_ok=True)
faiss.write_index(index, index_path)

metadata = [{"id": doc["id"], "metadata": doc["metadata"], "text": doc["content"]} for doc in documents]
with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)

print(f"\n:: Indexing complete. Total documents: {len(documents)}")
print(f"   ├── Index saved to:     {index_path}")
print(f"   └── Metadata saved to:  {metadata_path}")

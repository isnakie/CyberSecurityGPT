import json
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# File paths
jsonl_path = "data/cyber_threats/mitre_cwe_knowledge_base.jsonl"
index_path = "data/cyber_threats/mitre_faiss.index"
metadata_path = "data/cyber_threats/index_metadata.pkl"

# Load your JSONL knowledge base
documents = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        documents.append(doc)

# Use Sentence Transformers for embedding
print("🔍 Loading sentence transformer...")
model = SentenceTransformer("all-mpnet-base-v2")

# Generate embeddings
print("🧠 Embedding documents...")
texts = [doc["content"] for doc in documents]
embeddings = model.encode(texts, show_progress_bar=True)

# Build FAISS index
print("📦 Building FAISS index...")
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and metadata
print("💾 Saving index and metadata...")
os.makedirs(os.path.dirname(index_path), exist_ok=True)
faiss.write_index(index, index_path)

metadata = [{"id": doc["id"], "metadata": doc["metadata"], "text": doc["content"]} for doc in documents]
with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)

print(f"✅ Done! Saved {len(documents)} entries to:")
print(f"   • FAISS index: {index_path}")
print(f"   • Metadata:    {metadata_path}")

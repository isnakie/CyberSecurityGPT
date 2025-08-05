# Author: Sean Sjahrial
# Title: Cybersecurity RAG Assistant
# Description: Part of UC Berkeley MICS Machine Learning Course (2025)
# GitHub: https://github.com/isnakie
# Description: Converts a combined JSONL cybersecurity knowledge base (STIGs + MITRE CWEs)
# into a FAISS index and metadata pickle file for fast vector similarity search.
# License: MIT

import os
import json
import faiss
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Load all JSONL lines into memory as a list of dictionaries
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# Create and populate a flat FAISS index from dense vectors
def build_faiss_index(embeddings):
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def main():
    jsonl_path = "data/embeddings/combined_cybersecurity_knowledge_base.jsonl"
    index_path = "data/embeddings/combined_faiss.index"
    metadata_path = "data/embeddings/combined_metadata.pkl"

    print(f":: Loading JSONL from {jsonl_path}")
    entries = load_jsonl(jsonl_path)

    print(":: Initializing embedding model ...")
    model = SentenceTransformer("all-mpnet-base-v2")

    print(":: Encoding entries into dense vectors ...")
    texts = [entry["text"] for entry in entries]
    embeddings = model.encode(texts, show_progress_bar=True)

    print(":: Building FAISS index ...")
    index = build_faiss_index(embeddings)

    print(":: Saving FAISS index and metadata ...")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    # Build structured metadata for fast retrieval and display
    metadata = []
    for entry in entries:
        entry_id = entry.get("id") or entry.get("cwe_id") or entry.get("vuln_id") or "N/A"
        source = entry.get("source", "Unknown")
        base_title = entry.get("title") or entry.get("name") or "N/A"

        # Human-readable title that includes source-specific ID formatting
        if source.upper() == "MITRE" and entry_id != "N/A":
            title = f"CWE-{entry_id}: {base_title}"
        elif source.upper() == "STIG" and entry_id != "N/A":
            title = f"{entry_id}: {base_title}"
        else:
            title = f"{entry_id}: {base_title}"

        metadata.append({
            "id": entry_id,
            "title": title,
            "severity": entry.get("severity", ""),
            "source": source,
            "text": entry.get("text", "")
        })

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f":: FAISS index saved to: {index_path}")
    print(f":: Metadata saved to:   {metadata_path}")
    print(f":: Total entries indexed: {len(metadata)}")

if __name__ == "__main__":
    main()


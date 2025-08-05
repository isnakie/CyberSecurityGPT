# Author: Sean Sjahrial
# Title: Cybersecurity RAG Assistant
# Description: Part of UC Berkeley MICS Machine Learning Course (2025)
# GitHub: https://github.com/isnakie
# Description: CLI script to run standalone queries against a FAISS index built from STIG + MITRE CWE data.
# It uses sentence embeddings for semantic search and displays results with metadata and matched text.
# License: MIT

import argparse
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load both the FAISS index and the associated metadata
def load_index_and_metadata(index_path, metadata_path):
    print(":: Loading FAISS index and metadata ...")
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# Perform semantic search using the embedder
def search_index(query, model, index, metadata, top_k=5):
    embedding = model.encode([query])
    D, I = index.search(np.array(embedding).astype("float32"), top_k)
    return [metadata[i] for i in I[0]], D[0]

# Display search results, showing full text for the top hit and snippets for the rest
def display_results(results, distances):
    print()
    for i, (item, score) in enumerate(zip(results, distances)):
        meta = item.get("metadata", {})  # legacy support
        title = meta.get("title") or item.get("title") or "N/A"
        source = meta.get("source") or item.get("source") or "Unknown"
        text = item.get("text", "").strip()

        print(f"Result {i+1}")
        print(f"  Title   : {title}")
        print(f"  Source  : {source}")
        print(f"  Distance: {score:.4f}")

        if i == 0:
            print(f"\n  Full Match:\n{text}\n")
        else:
            snippet = (text[:400] + "...") if len(text) > 400 else text
            print(f"  Snippet:\n  {snippet}\n")

# Main CLI loop
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index", default="data/embeddings/combined_faiss.index", help="Path to FAISS index"
    )
    parser.add_argument(
        "--metadata", default="data/embeddings/combined_metadata.pkl", help="Path to metadata pickle"
    )
    parser.add_argument(
        "--model", default="all-mpnet-base-v2", help="SentenceTransformer model to use"
    )
    args = parser.parse_args()

    index, metadata = load_index_and_metadata(args.index, args.metadata)
    print(f":: Loading embedding model: {args.model} ...")
    model = SentenceTransformer(args.model)

    print("\n=== FAISS Search Console ===")
    while True:
        query = input("\n>> Enter your cybersecurity question (or type 'exit'): ").strip()
        if query.lower() in ("exit", "quit"):
            break

        print("\n:: Searching index ...")
        results, distances = search_index(query, model, index, metadata)
        display_results(results, distances)

if __name__ == "__main__":
    main()

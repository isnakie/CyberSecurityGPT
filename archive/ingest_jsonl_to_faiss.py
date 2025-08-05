# scripts/ingest/ingest_jsonl_to_faiss.py

import json
import uuid
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


def load_jsonl(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def embed_texts(texts, model_name="all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize


def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Cosine sim
    index.add(embeddings)
    return index


def main():
    jsonl_path = Path("data/STIGs/stig_traditional_security_checklist_v2r6.jsonl")
    index_out = Path("data/embeddings/stig_faiss.index")
    meta_out = Path("data/embeddings/stig_metadata.csv")
    index_out.parent.mkdir(parents=True, exist_ok=True)

    entries = load_jsonl(jsonl_path)

    texts = []
    metadatas = []

    for entry in entries:
        parts = [
            entry.get("title", ""),
            entry.get("description", ""),
            entry.get("check", ""),
            entry.get("fix", "")
        ]
        combined = "\n\n".join([p.strip() for p in parts if p.strip()])
        texts.append(combined)

        metadatas.append({
            "id": str(uuid.uuid4()),
            "vuln_id": entry.get("vuln_id", ""),
            "rule_id": entry.get("rule_id", ""),
            "severity": entry.get("severity", ""),
            "title": entry.get("title", "")
        })

    embeddings = embed_texts(texts)
    index = build_faiss_index(embeddings)

    faiss.write_index(index, str(index_out))
    pd.DataFrame(metadatas).to_csv(meta_out, index=False)

    print(f"✅ Saved {len(metadatas)} STIG entries to:")
    print(f"    • FAISS index:     {index_out}")
    print(f"    • Metadata (CSV):  {meta_out}")


if __name__ == "__main__":
    main()

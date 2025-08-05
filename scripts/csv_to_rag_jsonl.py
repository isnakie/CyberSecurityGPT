# Author: Sean Sjahrial
# Title: Cybersecurity RAG Assistant – MITRE CSV to JSONL Converter (Legacy)
# Description: Converts cleaned MITRE CWE CSV data into a RAG-compatible JSONL format.
# GitHub: https://github.com/isnakie
# License: MIT

# ------------------------------------------------------------------------------
# This script reads a cleaned MITRE CWE Top 25 CSV file and transforms it into
# a structured `.jsonl` format suitable for RAG-style semantic search and FAISS indexing.
# Fields are combined into a single text block under the "content" key.
# ------------------------------------------------------------------------------

import pandas as pd
import json
import os

# --- Paths ---
input_csv = "data/cyber_threats/mitre_cwe_clean.csv"
output_jsonl = "data/cyber_threats/mitre_cwe_knowledge_base.jsonl"
os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

def csv_to_rag_jsonl(input_csv, output_jsonl):
    df = pd.read_csv(input_csv)
    documents = []

    for _, row in df.iterrows():
        # Extract and structure important fields
        content_parts = [
            f"CWE-ID: {row.get('CWE-ID', '')}",
            f"Name: {row.get('Name', '')}",
            f"Description and Notes:\n{row.get('Description && Notes', '')}",
            f"Introduction:\n{row.get('Modes or Phase of Introduction', '')}",
            f"Detection Methods:\n{row.get('Detection Methods', '')}",
            f"Potential Mitigations:\n{row.get('Potential Mitigations', '')}",
            f"Observed Examples:\n{row.get('Observed Examples', '')}",
            f"Common Consequences:\n{row.get('Common Consequences', '')}",
        ]

        combined = "\n\n".join([p for p in content_parts if p.strip()])

        documents.append({
            "id": f"CWE-{row.get('CWE-ID', '')}",
            "content": combined,
            "metadata": {
                "cwe_id": row.get('CWE-ID', ''),
                "name": row.get('Name', '')
            }
        })

    # Save as JSON Lines format
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f":: Saved {len(documents)} MITRE entries to {output_jsonl}")

if __name__ == "__main__":
    csv_to_rag_jsonl(input_csv, output_jsonl)

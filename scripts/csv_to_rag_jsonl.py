import pandas as pd
import json
import os

# Load your cleaned MITRE CWE CSV

input_csv = "data/cyber_threats/mitre_cwe_clean.csv"
output_jsonl = "data/cyber_threats/mitre_cwe_knowledge_base.jsonl"
os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

def csv_to_rag_jsonl(input_csv, output_jsonl):
    df = pd.read_csv(input_csv)
    documents = []

    for _, row in df.iterrows():
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

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(documents)} RAG-ready entries to {output_jsonl}")

if __name__ == "__main__":
    csv_to_rag_jsonl(input_csv, output_jsonl)

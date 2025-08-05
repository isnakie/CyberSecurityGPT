# Author: Sean Sjahrial
# Title: Cybersecurity RAG Assistant
# Description: Part of UC Berkeley MICS Machine Learning Course (2025)
# GitHub: https://github.com/isnakie
# Description: Converts MITRE or STIG CSV files into JSONL format for downstream ingestion to FAISS.
# Supports both Top 25 CWEs and DISA Traditiona Security STIGs in a unified semantic format.
# License: MIT

"""
Usage:

Convert MITRE CSV
> python scripts/ingest/convert_csv_to_jsonl.py data/cyber_threats/mitre_cwe_clean.csv data/embeddings/mitre_cwe_knowledge_base.jsonl --format mitre

Convert STIG CSV
> python scripts/ingest/convert_csv_to_jsonl.py data/STIGs/stig_traditional_security_checklist_v2r6_flat.csv data/embeddings/stig_traditional_security_checklist_v2r6.jsonl --format stig
"""

import argparse
import csv
import json
from pathlib import Path

# --- Converts MITRE CWE CSV into structured JSONL format ---
def convert_mitre_csv_to_jsonl(input_path, output_path):
    records = []

    with open(input_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Extract fields with fallback to empty string
            cwe_id = row.get("CWE-ID", "").strip()
            name = row.get("Name", "").strip()
            description = row.get("Full Description", "").strip()
            likelihood = row.get("Modes or Phase of Introduction", "").strip()
            consequences = row.get("Common Consequences", "").strip()
            detection = row.get("Detection Methods", "").strip()
            mitigations = row.get("Potential Mitigations", "").strip()
            examples = row.get("Observed Examples", "").strip()

            # Full raw text used for embedding
            text = f"{name}. {description} Likelihood: {likelihood}. Consequences: {consequences}. Mitigations: {mitigations}. Detection: {detection}. Examples: {examples}"

            record = {
                "cwe_id": cwe_id,
                "name": name,
                "description": description,
                "likelihood": likelihood,
                "consequences": consequences,
                "mitigations": mitigations,
                "detection": detection,
                "examples": examples,
                "source": "MITRE",
                "text": text.strip()
            }
            records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in records:
            f.write(json.dumps(entry) + "\n")

    print(f":: Saved {len(records)} MITRE records to {output_path}")

# --- Converts STIG CSV into structured JSONL format ---
def convert_stig_csv_to_jsonl(input_path, output_path):
    records = []

    with open(input_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            vuln_id = row.get("vuln_id", "").strip()
            title = row.get("title", "").strip()
            description = row.get("description", "").strip()
            check = row.get("check", "").strip()
            fix = row.get("fix", "").strip()
            severity = row.get("severity", "").strip()

            text = f"{title}. {description} Check: {check} Fix: {fix}"

            record = {
                "vuln_id": vuln_id,
                "title": title,
                "description": description,
                "check": check,
                "fix": fix,
                "severity": severity,
                "source": "STIG",
                "text": text.strip()
            }
            records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in records:
            f.write(json.dumps(entry) + "\n")

    print(f":: Saved {len(records)} STIG records to {output_path}")

# --- Entry point ---
def main():
    parser = argparse.ArgumentParser(description="Convert MITRE or STIG CSV to JSONL format for FAISS ingestion")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("output_jsonl", help="Path to output JSONL file")
    parser.add_argument("--format", choices=["mitre", "stig"], required=True, help="Specify which CSV format to convert")

    args = parser.parse_args()

    if args.format == "mitre":
        convert_mitre_csv_to_jsonl(args.input_csv, args.output_jsonl)
    elif args.format == "stig":
        convert_stig_csv_to_jsonl(args.input_csv, args.output_jsonl)

if __name__ == "__main__":
    main()

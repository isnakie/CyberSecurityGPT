# Author: Sean Sjahrial
# Title: Cybersecurity RAG Assistant
# Description: Part of UC Berkeley MICS Machine Learning Course (2025)
# GitHub: https://github.com/isnakie
# Description: Fetches detailed DISA Traditional Security STIG vulnerability data from the Trackr.live API
# and saves to structured CSV. Used for downstream conversion to JSONL and vectorized ingestion in the
# Cybersecurity RAG pipeline.
# License: MIT

import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re
import argparse
import time

# --- Normalize file-safe names ---
def slugify(text):
    return re.sub(r'\W+', '_', text).lower()

# --- Strip extra line breaks and whitespace ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()

# --- Fetch list of vulnerability IDs for the STIG checklist ---
def fetch_stig_summary(title, version, release):
    url = f"https://cyber.trackr.live/api/stig/{title}/{version}/{release}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json().get("requirements", {})

# --- Retrieve full vulnerability entry from Trackr API ---
def fetch_stig_details(title, version, release, vuln_id):
    url = f"https://cyber.trackr.live/api/stig/{title}/{version}/{release}/{vuln_id}"
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"!! Failed to fetch {vuln_id}: {e}")
        return {}

# --- Save detailed entries as flat CSV for later processing ---
def save_to_csv(records, outdir, title, version, release):
    df = pd.DataFrame(records)
    slug = slugify(title)
    outpath = Path(outdir) / f"stig_{slug}_v{version}r{release}_flat.csv"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False)
    print(f":: Saved {len(df)} STIG entries to {outpath}")
    return outpath

# --- Entry point ---
def main():
    parser = argparse.ArgumentParser(description="Fetch detailed STIG entries from Trackr.live API")
    parser.add_argument("--title", default="Traditional_Security_Checklist", help="STIG checklist title slug")
    parser.add_argument("--version", default="2", help="STIG version (e.g. 2)")
    parser.add_argument("--release", default="6", help="STIG release number (e.g. 6)")
    parser.add_argument("--outdir", default="data/STIGs", help="Output directory for CSV")
    args = parser.parse_args()

    print(f":: Fetching detailed STIG data for {args.title} v{args.version}r{args.release} ...")

    requirements = fetch_stig_summary(args.title, args.version, args.release)
    records = []

    for vid in tqdm(requirements, desc=":: Retrieving vulnerabilities"):
        details = fetch_stig_details(args.title, args.version, args.release, vid)
        if not details:
            continue

        # Structure into flat record format
        record = {
            "vuln_id": details.get("id", vid),
            "rule_id": details.get("rule", ""),
            "severity": details.get("severity", ""),
            "title": clean_text(details.get("requirement-title", "")),
            "description": clean_text(details.get("requirement-description", "")),
            "check": clean_text(details.get("check-text", "")),
            "fix": clean_text(details.get("fix-text", "")),
        }
        records.append(record)
        time.sleep(0.05)  # be polite to the API

    save_to_csv(records, args.outdir, args.title, args.version, args.release)

if __name__ == "__main__":
    main()

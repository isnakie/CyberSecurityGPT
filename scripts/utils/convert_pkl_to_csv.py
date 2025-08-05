# scripts/utils/convert_pkl_to_csv.py

import pickle
import pandas as pd
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert FAISS metadata .pkl to CSV for inspection.")
    parser.add_argument("pkl_file", help="Path to metadata .pkl file")
    parser.add_argument("csv_file", help="Path to save resulting .csv file")
    args = parser.parse_args()

    # Load .pkl
    print(f"📦 Loading metadata from: {args.pkl_file}")
    with open(args.pkl_file, "rb") as f:
        metadata = pickle.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(metadata)

    # Save to CSV
    Path(args.csv_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv_file, index=False)
    print(f"✅ Saved CSV to: {args.csv_file}")

if __name__ == "__main__":
    main()

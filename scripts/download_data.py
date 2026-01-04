import argparse
import os
from pathlib import Path

import pandas as pd

# Public copy of UCI Heart dataset in CSV form (same columns as your heart.csv style dataset)
RAW_URL = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"

def main(out: str):
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from:\n{RAW_URL}")
    df = pd.read_csv(RAW_URL)

    # Basic sanity checks (fail fast if URL returns unexpected content)
    required_cols = {"age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                     "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Downloaded CSV is missing columns: {sorted(missing)}")

    df.to_csv(out_path, index=False)
    print(f"Saved dataset to: {out_path.resolve()}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/heart.csv", help="Output CSV path")
    args = parser.parse_args()
    main(args.out)

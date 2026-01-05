import argparse
from pathlib import Path

import pandas as pd

# Public copy of the UCI Heart dataset in CSV form
RAW_URL = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"

REQUIRED_COLS = {
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
}


def main(out: str):
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from: {RAW_URL}")
    df = pd.read_csv(RAW_URL)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Downloaded CSV missing columns: {sorted(missing)}")

    df.to_csv(out_path, index=False)
    print(f"Saved dataset to: {out_path.resolve()} (shape={df.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/heart.csv", help="Output CSV path")
    args = parser.parse_args()
    main(args.out)

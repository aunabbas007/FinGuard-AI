
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "v2"


def main():
    X = pd.read_csv(DATA_DIR / "features.csv")
    y = pd.read_csv(DATA_DIR / "labels.csv")

    print("========== FEATURE VALIDATION ==========")

    print("\nFeature shape:", X.shape)
    print("Label shape:", y.shape)

    print("\nFirst five rows:")
    print(X.head())

    print("\nMissing values:", X.isnull().sum().sum())

    print("\nInfinite values:", X.isin([float("inf"), float("-inf")]).sum().sum())

    print("\nDuplicate rows:", X.duplicated().sum())

    print("\nFraud distribution:")
    print(y["is_fraud"].value_counts())

    print("\nTarget leakage check:")

    forbidden = ["is_fraud", "fraud_type"]

    for column in forbidden:
        print(f"{column}: {column in X.columns}")

    print("\nFeature data types:")
    print(X.dtypes.value_counts())

    print("\nFeature validation completed!")


if __name__ == "__main__":
    main()

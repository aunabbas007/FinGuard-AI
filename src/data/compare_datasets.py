
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

V1_PATH = ROOT / "data" / "raw" / "transactions.csv"
V2_PATH = ROOT / "data" / "raw" / "v2" / "transactions.csv"


def inspect_dataset(name, path):
    df = pd.read_csv(path)

    print(f"\n========== {name} ==========")
    print("Shape:", df.shape)

    print("\nAmount statistics:")
    print(
        df.groupby("is_fraud")["amount"]
        .agg(["mean", "median", "min", "max"])
        .round(2)
    )

    print("\nFraud rate by hour:")
    print(
        (df.groupby("hour")["is_fraud"].mean() * 100)
        .round(2)
    )

    print("\nFraud rate by channel:")
    print(
        (df.groupby("channel")["is_fraud"].mean() * 100)
        .round(2)
    )

    return df


def main():
    v1 = inspect_dataset("V1 BASELINE", V1_PATH)
    v2 = inspect_dataset("V2 IMPROVED", V2_PATH)

    print("\n========== COMPARISON ==========")

    print("V1 fraud rate:", v1["is_fraud"].mean())
    print("V2 fraud rate:", v2["is_fraud"].mean())

    print(
        "\nV1 normal transaction median:",
        v1.loc[v1["is_fraud"] == 0, "amount"].median()
    )

    print(
        "V2 normal transaction median:",
        v2.loc[v2["is_fraud"] == 0, "amount"].median()
    )


if __name__ == "__main__":
    main()

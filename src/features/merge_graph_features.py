
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = ROOT / "data/processed/v2/features.csv"
LABEL_PATH = ROOT / "data/processed/v2/labels.csv"
GRAPH_PATH = ROOT / "data/processed/v2/graph_features.csv"

OUTPUT_PATH = ROOT / "data/processed/v2/combined_features.csv"


def main():
    print("Loading datasets...")

    features = pd.read_csv(FEATURE_PATH)
    labels = pd.read_csv(LABEL_PATH)
    graph_features = pd.read_csv(GRAPH_PATH)

    # Attach transaction IDs to the original feature matrix.
    if len(features) != len(labels):
        raise ValueError("Feature and label row counts do not match.")

    features.insert(0, "transaction_id", labels["transaction_id"])

    # Verify transaction IDs before merging.
    if features["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs in original features.")

    if graph_features["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs in graph features.")

    if set(features["transaction_id"]) != set(graph_features["transaction_id"]):
        raise ValueError("Transaction IDs do not match between datasets.")

    # Merge graph features using transaction ID.
    combined = features.merge(
        graph_features,
        on="transaction_id",
        how="inner",
        validate="one_to_one",
        sort=False
    )

    # Confirm original feature order is preserved.
    if not combined["transaction_id"].equals(features["transaction_id"]):
        raise ValueError("Transaction ordering changed during merge.")

    combined.to_csv(OUTPUT_PATH, index=False)

    print("\nFEATURE INTEGRATION COMPLETED")
    print("=" * 60)
    print("Transactions:", len(combined))
    print("Predictive features:", len(combined.columns) - 1)
    print("Missing values:", combined.isnull().sum().sum())
    print("Duplicate transactions:", combined["transaction_id"].duplicated().sum())
    print("Saved:", OUTPUT_PATH)

    print("\nFeature sample:")
    print(combined.head().to_string(index=False))


if __name__ == "__main__":
    main()


from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.fraud.risk_scoring import FraudRiskScorer


ROOT = Path(__file__).resolve().parents[2]


def main():

    features = pd.read_csv(
        ROOT / "data/processed/v2/features.csv"
    )

    scorer = FraudRiskScorer()

    sample = features.head(10)

    results = scorer.score_transactions(sample)

    print("\nFIN GUARD AI - RISK SCORING")
    print("=" * 75)

    print(results.to_string(index=False))

    print("\nRisk category distribution:")
    print(results["risk_level"].value_counts())

    print("\nFlagged transactions:")
    print(results["is_flagged"].sum())


if __name__ == "__main__":
    main()

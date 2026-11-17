
from pathlib import Path

import numpy as np
import pandas as pd


# Reproducibility: same seed produces the same dataset.
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

NUM_CUSTOMERS = 500
NUM_TRANSACTIONS = 10_000
FRAUD_RATE = 0.02

COUNTRIES = ["India", "UAE", "UK", "USA", "Singapore"]
CURRENCIES = ["INR", "AED", "GBP", "USD", "SGD"]

CHANNELS = ["mobile", "web", "ATM", "POS"]
TRANSACTION_TYPES = ["transfer", "payment", "withdrawal"]
MERCHANT_CATEGORIES = [
    "retail", "groceries", "travel", "electronics",
    "restaurants", "fuel", "utilities"
]


def generate_customers():
    customer_ids = [f"CUST{i:04d}" for i in range(NUM_CUSTOMERS)]

    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "age": rng.integers(18, 75, NUM_CUSTOMERS),
        "country": rng.choice(COUNTRIES, NUM_CUSTOMERS),
        "account_tenure_months": rng.integers(1, 240, NUM_CUSTOMERS),
        "customer_segment": rng.choice(
            ["standard", "premium", "business"],
            NUM_CUSTOMERS,
            p=[0.65, 0.25, 0.10]
        )
    })

    return customers


def generate_accounts(customers):
    accounts = pd.DataFrame({
        "account_id": [f"ACC{i:04d}" for i in range(NUM_CUSTOMERS)],
        "customer_id": customers["customer_id"],
        "account_type": rng.choice(
            ["savings", "current"],
            NUM_CUSTOMERS,
            p=[0.75, 0.25]
        ),
        "balance": np.round(
            rng.lognormal(mean=10, sigma=1, size=NUM_CUSTOMERS),
            2
        ),
        "currency": rng.choice(CURRENCIES, NUM_CUSTOMERS)
    })

    return accounts


def generate_normal_transactions(accounts):
    account_ids = accounts["account_id"].to_numpy()

    sender_indices = rng.integers(
        0, len(account_ids), NUM_TRANSACTIONS
    )

    receiver_indices = rng.integers(
        0, len(account_ids), NUM_TRANSACTIONS
    )

    # Prevent an account from sending money to itself.
    receiver_indices = (
        receiver_indices + (receiver_indices == sender_indices)
    ) % len(account_ids)

    start_date = pd.Timestamp("2026-01-01")

    timestamps = (
        start_date
        + pd.to_timedelta(
            rng.integers(0, 180 * 24 * 60, NUM_TRANSACTIONS),
            unit="m"
        )
    )

    transactions = pd.DataFrame({
        "transaction_id": [
            f"TXN{i:06d}" for i in range(NUM_TRANSACTIONS)
        ],
        "timestamp": timestamps,
        "sender_account_id": account_ids[sender_indices],
        "receiver_account_id": account_ids[receiver_indices],
        "amount": np.round(
            rng.lognormal(mean=7.5, sigma=1.0, size=NUM_TRANSACTIONS),
            2
        ),
        "channel": rng.choice(CHANNELS, NUM_TRANSACTIONS),
        "transaction_type": rng.choice(
            TRANSACTION_TYPES, NUM_TRANSACTIONS
        ),
        "merchant_category": rng.choice(
            MERCHANT_CATEGORIES, NUM_TRANSACTIONS
        ),
        "sender_country": rng.choice(
            ["India", "India", "India", "UAE", "UK"],
            NUM_TRANSACTIONS
        ),
        "receiver_country": rng.choice(
            ["India", "India", "India", "UAE", "UK"],
            NUM_TRANSACTIONS
        ),
        "device_id": [
            f"DEV{value:04d}"
            for value in rng.integers(1, 800, NUM_TRANSACTIONS)
        ],
        "is_new_device": rng.choice(
            [0, 1], NUM_TRANSACTIONS, p=[0.95, 0.05]
        ),
        "is_fraud": 0,
        "fraud_type": "none"
    })

    return transactions


def inject_fraud_patterns(transactions):
    fraud_count = int(NUM_TRANSACTIONS * FRAUD_RATE)

    selected_indices = rng.choice(
        transactions.index,
        size=fraud_count,
        replace=False
    )

    # Split fraud examples across four simulated scenarios.
    scenarios = np.array_split(selected_indices, 4)

    # Scenario 1: Account takeover.
    indices = scenarios[0]
    transactions.loc[indices, "is_new_device"] = 1
    transactions.loc[indices, "channel"] = "web"
    transactions.loc[indices, "timestamp"] = (
        transactions.loc[indices, "timestamp"].dt.normalize()
        + pd.Timedelta(hours=2)
    )
    transactions.loc[indices, "fraud_type"] = "account_takeover"

    # Scenario 2: Unusually large transfers.
    indices = scenarios[1]
    transactions.loc[indices, "amount"] = np.round(
        rng.uniform(150_000, 500_000, len(indices)), 2
    )
    transactions.loc[indices, "fraud_type"] = "unusual_large_transfer"

    # Scenario 3: Structuring-like transaction amounts.
    indices = scenarios[2]
    transactions.loc[indices, "amount"] = np.round(
        rng.uniform(8_000, 9_900, len(indices)), 2
    )
    transactions.loc[indices, "fraud_type"] = "structuring_pattern"

    # Scenario 4: Unusual cross-border activity.
    indices = scenarios[3]
    transactions.loc[indices, "sender_country"] = "India"
    transactions.loc[indices, "receiver_country"] = rng.choice(
        ["USA", "UK", "Singapore"], len(indices)
    )
    transactions.loc[indices, "is_new_device"] = 1
    transactions.loc[indices, "fraud_type"] = "unusual_cross_border"

    transactions.loc[selected_indices, "is_fraud"] = 1

    return transactions


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic banking data...")

    customers = generate_customers()
    accounts = generate_accounts(customers)

    transactions = generate_normal_transactions(accounts)
    transactions = inject_fraud_patterns(transactions)

    # Derive transaction hour from timestamp.
    transactions["hour"] = transactions["timestamp"].dt.hour

    customers.to_csv(RAW_DATA_DIR / "customers.csv", index=False)
    accounts.to_csv(RAW_DATA_DIR / "accounts.csv", index=False)
    transactions.to_csv(RAW_DATA_DIR / "transactions.csv", index=False)

    print("\nDataset generation completed!")
    print(f"Customers: {len(customers)}")
    print(f"Accounts: {len(accounts)}")
    print(f"Transactions: {len(transactions)}")

    print("\nFraud distribution:")
    print(transactions["fraud_type"].value_counts())

    print("\nFraud percentage:")
    print(f"{transactions['is_fraud'].mean() * 100:.2f}%")

    print(f"\nFiles saved to: {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()

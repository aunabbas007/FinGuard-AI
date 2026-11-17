
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 2026
rng = np.random.default_rng(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "v2"

NUM_CUSTOMERS = 500
NUM_TRANSACTIONS = 10_000
FRAUD_RATE = 0.02

COUNTRIES = ["India", "UAE", "UK", "USA", "Singapore"]

CURRENCY_MAP = {
    "India": "INR",
    "UAE": "AED",
    "UK": "GBP",
    "USA": "USD",
    "Singapore": "SGD",
}

CHANNELS = ["mobile", "web", "ATM", "POS"]

MERCHANT_CATEGORIES = [
    "retail",
    "groceries",
    "travel",
    "electronics",
    "restaurants",
    "fuel",
    "utilities",
]


def generate_customers():
    customers = pd.DataFrame({
        "customer_id": [
            f"CUST{i:04d}" for i in range(NUM_CUSTOMERS)
        ],
        "age": rng.integers(18, 75, NUM_CUSTOMERS),
        "country": rng.choice(
            COUNTRIES,
            NUM_CUSTOMERS,
            p=[0.65, 0.12, 0.08, 0.08, 0.07]
        ),
        "account_tenure_months": rng.integers(
            1, 240, NUM_CUSTOMERS
        ),
        "customer_segment": rng.choice(
            ["standard", "premium", "business"],
            NUM_CUSTOMERS,
            p=[0.65, 0.25, 0.10]
        ),
        "typical_transaction_amount": np.round(
            rng.lognormal(mean=7.5, sigma=0.8, size=NUM_CUSTOMERS),
            2
        ),
        "typical_monthly_transactions": rng.integers(
            5, 60, NUM_CUSTOMERS
        )
    })

    return customers


def generate_accounts(customers):
    accounts = pd.DataFrame({
        "account_id": [
            f"ACC{i:04d}" for i in range(NUM_CUSTOMERS)
        ],
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
        "currency": customers["country"].map(CURRENCY_MAP)
    })

    return accounts


def generate_normal_transactions(customers, accounts):
    account_ids = accounts["account_id"].to_numpy()
    customer_ids = customers["customer_id"].to_numpy()

    customer_count = len(customers)

    # Each transaction gets a sender account.
    sender_indices = rng.integers(
        0, customer_count, NUM_TRANSACTIONS
    )

    receiver_indices = rng.integers(
        0, customer_count, NUM_TRANSACTIONS
    )

    # Prevent self-transfers.
    receiver_indices = (
        receiver_indices
        + (receiver_indices == sender_indices)
    ) % customer_count

    # Each customer has a different typical transaction amount.
    customer_baselines = customers[
        "typical_transaction_amount"
    ].to_numpy()

    base_amounts = customer_baselines[sender_indices]

    # Transaction amounts vary around each customer's baseline.
    amounts = base_amounts * rng.lognormal(
        mean=0,
        sigma=0.65,
        size=NUM_TRANSACTIONS
    )

    
    # Most normal transactions happen during the day.
    hour_probabilities = np.array([
        0.01, 0.005, 0.003, 0.003, 0.005, 0.01,
        0.025, 0.045, 0.065, 0.075, 0.075, 0.075,
        0.075, 0.075, 0.075, 0.075, 0.075, 0.065,
        0.055, 0.045, 0.025, 0.015, 0.01, 0.01
    ])

    # Normalize probabilities so their sum equals 1.
    hour_probabilities /= hour_probabilities.sum()

    hours = rng.choice(
        np.arange(24),
        NUM_TRANSACTIONS,
        p=hour_probabilities
    )


    days = rng.integers(0, 180, NUM_TRANSACTIONS)
    minutes = rng.integers(0, 60, NUM_TRANSACTIONS)

    timestamps = (
        pd.Timestamp("2026-01-01")
        + pd.to_timedelta(days, unit="D")
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
    )

    # Each account has a small set of familiar devices.
    device_ids = [
        [f"DEV{i:04d}A", f"DEV{i:04d}B"]
        for i in range(NUM_CUSTOMERS)
    ]

    devices = []
    is_new_device = []

    for sender_index in sender_indices:
        if rng.random() < 0.03:
            devices.append(
                f"NEWDEV{rng.integers(100000, 999999)}"
            )
            is_new_device.append(1)
        else:
            devices.append(
                rng.choice(device_ids[sender_index])
            )
            is_new_device.append(0)

    sender_countries = customers["country"].to_numpy()

    receiver_countries = []

    for sender_index in sender_indices:
        if rng.random() < 0.08:
            other_countries = [
                country for country in COUNTRIES
                if country != sender_countries[sender_index]
            ]
            receiver_countries.append(rng.choice(other_countries))
        else:
            receiver_countries.append(
                sender_countries[sender_index]
            )

    transactions = pd.DataFrame({
        "transaction_id": [
            f"TXN{i:06d}" for i in range(NUM_TRANSACTIONS)
        ],
        "timestamp": timestamps,
        "sender_account_id": account_ids[sender_indices],
        "receiver_account_id": account_ids[receiver_indices],
        "amount": np.round(amounts, 2),
        "channel": rng.choice(
            CHANNELS,
            NUM_TRANSACTIONS,
            p=[0.55, 0.25, 0.10, 0.10]
        ),
        "transaction_type": rng.choice(
            ["transfer", "payment", "withdrawal"],
            NUM_TRANSACTIONS,
            p=[0.55, 0.35, 0.10]
        ),
        "merchant_category": rng.choice(
            MERCHANT_CATEGORIES,
            NUM_TRANSACTIONS
        ),
        "sender_country": sender_countries[sender_indices],
        "receiver_country": receiver_countries,
        "device_id": devices,
        "is_new_device": is_new_device,
        "is_fraud": 0,
        "fraud_type": "none"
    })

    return transactions


def inject_fraud_patterns(transactions, customers):
    fraud_count = int(NUM_TRANSACTIONS * FRAUD_RATE)

    selected = rng.choice(
        transactions.index,
        size=fraud_count,
        replace=False
    )

    scenarios = np.array_split(selected, 4)

    baselines = customers[
        "typical_transaction_amount"
    ].to_numpy()

    account_ids = [
        f"ACC{i:04d}" for i in range(NUM_CUSTOMERS)
    ]

    # 1. Account takeover:
    # New device and unusual transaction amount.
    indices = scenarios[0]

    for index in indices:
        sender = transactions.at[index, "sender_account_id"]
        customer_index = int(sender[-4:])

        baseline = baselines[customer_index]

        transactions.at[index, "device_id"] = (
            f"ATTACKDEV{rng.integers(100000, 999999)}"
        )
        transactions.at[index, "is_new_device"] = 1
        transactions.at[index, "amount"] = round(
            baseline * rng.uniform(2, 6), 2
        )
        transactions.at[index, "channel"] = rng.choice(
            ["web", "mobile"]
        )
        transactions.at[index, "fraud_type"] = "account_takeover"

    # 2. Unusual large transfers:
    # Amount is elevated relative to the customer's baseline.
    indices = scenarios[1]

    for index in indices:
        sender = transactions.at[index, "sender_account_id"]
        customer_index = int(sender[-4:])

        baseline = baselines[customer_index]

        transactions.at[index, "amount"] = round(
            baseline * rng.uniform(2.5, 10), 2
        )
        transactions.at[index, "fraud_type"] = (
            "unusual_large_transfer"
        )

    # 3. Structuring-like activity:
    # Ten groups, each containing five rapid transfers.
    indices = scenarios[2]

    groups = np.array_split(indices, 10)

    for group in groups:
        sender_index = int(rng.integers(0, NUM_CUSTOMERS))
        sender_account = account_ids[sender_index]

        base_time = (
            pd.Timestamp("2026-01-01")
            + pd.Timedelta(days=int(rng.integers(0, 180)))
            + pd.Timedelta(hours=int(rng.integers(7, 22)))
        )

        for index in group:
            transactions.at[index, "sender_account_id"] = (
                sender_account
            )
            transactions.at[index, "timestamp"] = (
                base_time
                + pd.Timedelta(minutes=int(rng.integers(0, 30)))
            )
            transactions.at[index, "amount"] = round(
                rng.uniform(1_000, 7_000), 2
            )
            transactions.at[index, "fraud_type"] = (
                "structuring_pattern"
            )

    # 4. Unusual cross-border activity.
    indices = scenarios[3]

    for index in indices:
        sender_country = transactions.at[index, "sender_country"]

        alternatives = [
            country for country in COUNTRIES
            if country != sender_country
        ]

        transactions.at[index, "receiver_country"] = (
            rng.choice(alternatives)
        )
        transactions.at[index, "fraud_type"] = (
            "unusual_cross_border"
        )

    transactions.loc[selected, "is_fraud"] = 1

    return transactions


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating FinGuard AI V2 dataset...")

    customers = generate_customers()
    accounts = generate_accounts(customers)

    transactions = generate_normal_transactions(
        customers,
        accounts
    )

    transactions = inject_fraud_patterns(
        transactions,
        customers
    )

    transactions["hour"] = (
        pd.to_datetime(transactions["timestamp"]).dt.hour
    )

    transactions["day_of_week"] = (
        pd.to_datetime(transactions["timestamp"]).dt.dayofweek
    )

    customers.to_csv(
        OUTPUT_DIR / "customers.csv", index=False
    )

    accounts.to_csv(
        OUTPUT_DIR / "accounts.csv", index=False
    )

    transactions.to_csv(
        OUTPUT_DIR / "transactions.csv", index=False
    )

    print("\nV2 generation completed!")
    print("Customers:", len(customers))
    print("Accounts:", len(accounts))
    print("Transactions:", len(transactions))

    print("\nFraud distribution:")
    print(transactions["fraud_type"].value_counts())

    print(
        "\nFraud percentage:",
        f"{transactions['is_fraud'].mean() * 100:.2f}%"
    )

    print("\nFiles saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

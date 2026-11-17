
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw" / "v2"
OUTPUT_DIR = ROOT / "data" / "processed" / "v2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():
    transactions = pd.read_csv(
        DATA_DIR / "transactions.csv"
    )

    customers = pd.read_csv(
        DATA_DIR / "customers.csv"
    )

    accounts = pd.read_csv(
        DATA_DIR / "accounts.csv"
    )

    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"]
    )

    return transactions, customers, accounts


# --------------------------------------------------
# TEMPORAL VELOCITY FEATURES
# --------------------------------------------------

def add_velocity_features(df):
    """
    Calculate previous transaction activity for each sender.

    The current transaction is excluded.
    Only transactions with strictly earlier timestamps
    are counted.
    """

    df = df.sort_values(
        ["sender_account_id", "timestamp", "transaction_id"]
    ).copy()

    df["transactions_last_hour"] = 0
    df["transactions_last_24h"] = 0
    df["minutes_since_previous"] = np.nan

    for _, group in df.groupby(
        "sender_account_id",
        sort=False
    ):

        times = (
            group["timestamp"]
            .astype("int64")
            .to_numpy()
        )

        hour_counts = []
        day_counts = []
        previous_minutes = []

        for i, current_time in enumerate(times):

            previous_times = times[:i]

            one_hour_ago = (
                current_time - 3_600_000_000_000
            )

            one_day_ago = (
                current_time - 86_400_000_000_000
            )

            hour_start = np.searchsorted(
                previous_times,
                one_hour_ago,
                side="left"
            )

            day_start = np.searchsorted(
                previous_times,
                one_day_ago,
                side="left"
            )

            # Count only transactions in the preceding window.
            # Exclude transactions at the current timestamp.
            hour_counts.append(
                np.sum(
                    (previous_times >= one_hour_ago)
                    & (previous_times < current_time)
                )
            )

            day_counts.append(
                np.sum(
                    (previous_times >= one_day_ago)
                    & (previous_times < current_time)
                )
            )

            if i == 0:
                previous_minutes.append(np.nan)
            else:
                difference = (
                    current_time - times[i - 1]
                )

                previous_minutes.append(
                    difference / 60_000_000_000
                )

        df.loc[
            group.index,
            "transactions_last_hour"
        ] = hour_counts

        df.loc[
            group.index,
            "transactions_last_24h"
        ] = day_counts

        df.loc[
            group.index,
            "minutes_since_previous"
        ] = previous_minutes

    return df


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def build_features(
    transactions,
    customers,
    accounts
):

    df = transactions.copy()

    # --------------------------------------------------
    # ACCOUNT INFORMATION
    # --------------------------------------------------

    df = df.merge(
        accounts[
            [
                "account_id",
                "customer_id",
                "balance",
                "account_type"
            ]
        ],
        left_on="sender_account_id",
        right_on="account_id",
        how="left",
        validate="many_to_one"
    )

    # --------------------------------------------------
    # CUSTOMER INFORMATION
    # --------------------------------------------------

    df = df.merge(
        customers[
            [
                "customer_id",
                "typical_transaction_amount",
                "typical_monthly_transactions",
                "account_tenure_months",
                "customer_segment"
            ]
        ],
        on="customer_id",
        how="left",
        validate="many_to_one"
    )

    # --------------------------------------------------
    # TIME FEATURES
    # --------------------------------------------------

    df["hour"] = df["timestamp"].dt.hour

    df["day_of_week"] = (
        df["timestamp"].dt.day_of_week
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------
    # GEOGRAPHICAL FEATURES
    # --------------------------------------------------

    df["is_cross_border"] = (
        df["sender_country"]
        != df["receiver_country"]
    ).astype(int)

    # --------------------------------------------------
    # TRANSACTION AMOUNT FEATURES
    # --------------------------------------------------

    df["amount_log"] = np.log1p(
        df["amount"].clip(lower=0)
    )

    df["amount_vs_customer_average"] = (
        df["amount"]
        / df["typical_transaction_amount"].clip(lower=1)
    )

    df["balance_to_transaction_ratio"] = (
        df["balance"]
        / df["amount"].clip(lower=1)
    )

    # --------------------------------------------------
    # DEVICE FEATURES
    # --------------------------------------------------

    df["new_device_flag"] = (
        df["is_new_device"].astype(int)
    )

    # --------------------------------------------------
    # VELOCITY FEATURES
    # --------------------------------------------------

    df = add_velocity_features(df)

    # --------------------------------------------------
    # SEPARATE TARGET LABELS
    # --------------------------------------------------

    labels = df[
        [
            "transaction_id",
            "is_fraud",
            "fraud_type"
        ]
    ].copy()

    # --------------------------------------------------
    # REMOVE IDENTIFIERS AND TARGET LEAKAGE
    # --------------------------------------------------

    excluded_columns = [
        "transaction_id",
        "timestamp",
        "sender_account_id",
        "receiver_account_id",
        "device_id",
        "fraud_type",
        "is_fraud",
        "account_id",
        "customer_id",
        "sender_country",
        "receiver_country"
    ]

    features = df.drop(
        columns=excluded_columns,
        errors="ignore"
    )

    # --------------------------------------------------
    # ENCODE CATEGORICAL FEATURES
    # --------------------------------------------------

    categorical_columns = [
        "channel",
        "transaction_type",
        "merchant_category",
        "account_type",
        "customer_segment"
    ]

    features = pd.get_dummies(
        features,
        columns=categorical_columns,
        dtype=int
    )

    # --------------------------------------------------
    # HANDLE INVALID VALUES
    # --------------------------------------------------

    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    features = features.fillna(0)

    return features, labels


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("Loading FinGuard AI V2 data...")

    transactions, customers, accounts = load_data()

    print("Building features...")

    features, labels = build_features(
        transactions,
        customers,
        accounts
    )

    # Preserve the original transaction ordering.
    # This keeps features and labels aligned with the
    # transaction source file for subsequent processing.
    original_ids = transactions["transaction_id"].tolist()

    feature_ids = labels["transaction_id"].tolist()

    feature_order = pd.Series(
        range(len(feature_ids)),
        index=feature_ids
    ).loc[original_ids].to_numpy()

    features = features.iloc[feature_order].reset_index(drop=True)
    labels = labels.iloc[feature_order].reset_index(drop=True)

    # --------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------

    features.to_csv(
        OUTPUT_DIR / "features.csv",
        index=False
    )

    labels.to_csv(
        OUTPUT_DIR / "labels.csv",
        index=False
    )

    print("\nFeature engineering completed!")

    print("Feature matrix:", features.shape)
    print("Label matrix:", labels.shape)

    print("\nGenerated features:")

    for column in features.columns:
        print("-", column)

    print(
        "\nMissing values:",
        features.isnull().sum().sum()
    )

    print(
        "\nInfinite values:",
        np.isinf(
            features.select_dtypes(include=np.number)
        ).sum().sum()
    )

    print(
        "\nFraud distribution:"
    )

    print(
        labels["is_fraud"].value_counts()
    )

    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

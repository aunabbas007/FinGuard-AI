from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def inspect_data(df):
    print("\n========== DATA OVERVIEW ==========")
    print("Shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate transaction IDs:")
    print(df["transaction_id"].duplicated().sum())

    print("\nFraud distribution:")
    print(df["fraud_type"].value_counts())

    print("\nFraud percentage:")
    print(f"{df['is_fraud'].mean() * 100:.2f}%")


def analyze_amounts(df):
    print("\n========== TRANSACTION AMOUNTS ==========")

    print(
        df.groupby("is_fraud")["amount"]
        .describe()
    )

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="amount",
        hue="is_fraud",
        bins=60,
        element="step",
        stat="density",
        common_norm=False
    )

    plt.title("Transaction Amount Distribution")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "transaction_amounts.png")
    plt.close()


def analyze_fraud_types(df):
    plt.figure(figsize=(10, 6))

    fraud_df = df[df["is_fraud"] == 1]

    sns.countplot(
        data=fraud_df,
        x="fraud_type"
    )

    plt.title("Fraud Distribution by Type")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fraud_types.png")
    plt.close()


def analyze_channels(df):
    channel_rates = (
        df.groupby("channel")["is_fraud"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\n========== FRAUD RATE BY CHANNEL ==========")
    print((channel_rates * 100).round(2))

    plt.figure(figsize=(9, 5))

    sns.barplot(
        x=channel_rates.index,
        y=channel_rates.values
    )

    plt.title("Fraud Rate by Transaction Channel")
    plt.ylabel("Fraud Rate")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fraud_by_channel.png")
    plt.close()


def analyze_hourly_activity(df):
    hourly = (
        df.groupby("hour")["is_fraud"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 5))

    sns.lineplot(
        data=hourly,
        x="hour",
        y="is_fraud",
        marker="o"
    )

    plt.title("Fraud Rate by Hour")
    plt.ylabel("Fraud Rate")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fraud_by_hour.png")
    plt.close()


def analyze_cross_border(df):
    df["is_cross_border"] = (
        df["sender_country"] != df["receiver_country"]
    ).astype(int)

    cross_border_rates = (
        df.groupby("is_cross_border")["is_fraud"]
        .mean()
    )

    print("\n========== CROSS-BORDER FRAUD RATE ==========")
    print(cross_border_rates)


def main():
    df = load_data()

    inspect_data(df)
    analyze_amounts(df)
    analyze_fraud_types(df)
    analyze_channels(df)
    analyze_hourly_activity(df)
    analyze_cross_border(df)

    print("\nEDA completed!")
    print("Charts saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

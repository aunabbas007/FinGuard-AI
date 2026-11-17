from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRANSACTION_PATH = ROOT / "data/raw/v2/transactions.csv"
OUTPUT_DIR = ROOT / "data/processed/v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class GraphFeatureBuilder:
    """Build strictly historical account-network features per transaction."""

    def __init__(self):
        self.transactions = pd.read_csv(TRANSACTION_PATH)
        required = {
            "transaction_id", "timestamp", "sender_account_id",
            "receiver_account_id", "amount"
        }
        missing = required - set(self.transactions.columns)
        if missing:
            raise ValueError(f"Transactions missing columns: {sorted(missing)}")

        if self.transactions["transaction_id"].duplicated().any():
            raise ValueError("Duplicate transaction IDs detected.")

        self.transactions["timestamp"] = pd.to_datetime(
            self.transactions["timestamp"], errors="raise"
        )
        self.transactions["amount"] = pd.to_numeric(
            self.transactions["amount"], errors="raise"
        )
        if self.transactions[["sender_account_id", "receiver_account_id"]].isna().any().any():
            raise ValueError("Sender or receiver account IDs contain missing values.")

        self.outgoing_24h = defaultdict(deque)
        self.incoming_24h = defaultdict(deque)
        self.outgoing_30d = defaultdict(deque)
        self.incoming_30d = defaultdict(deque)
        self.pair_history_30d = defaultdict(deque)
        self.features = []

        self.window_24h = pd.Timedelta(hours=24)
        self.window_30d = pd.Timedelta(days=30)

    @staticmethod
    def remove_expired(history, cutoff):
        while history and history[0][0] <= cutoff:
            history.popleft()

    def build_features(self):
        print("\nBuilding temporal graph features...")
        df = self.transactions.copy()
        df["original_position"] = range(len(df))
        df = df.sort_values(["timestamp", "transaction_id"])

        for timestamp, group in df.groupby("timestamp", sort=True):
            cutoff_24h = timestamp - self.window_24h
            cutoff_30d = timestamp - self.window_30d
            batch = []

            # Read history before adding any transaction from this timestamp.
            for row in group.itertuples(index=False):
                sender = row.sender_account_id
                receiver = row.receiver_account_id

                sender_out_24h = self.outgoing_24h[sender]
                sender_in_24h = self.incoming_24h[sender]
                receiver_in_24h = self.incoming_24h[receiver]
                sender_out_30d = self.outgoing_30d[sender]
                receiver_in_30d = self.incoming_30d[receiver]
                reverse_history = self.pair_history_30d[(receiver, sender)]

                for history in (
                    sender_out_24h, sender_in_24h, receiver_in_24h
                ):
                    self.remove_expired(history, cutoff_24h)

                for history in (
                    sender_out_30d, receiver_in_30d, reverse_history
                ):
                    self.remove_expired(history, cutoff_30d)

                batch.append({
                    "original_position": row.original_position,
                    "transaction_id": row.transaction_id,
                    "sender_outgoing_count_24h": len(sender_out_24h),
                    "sender_outgoing_amount_24h": sum(e[2] for e in sender_out_24h),
                    "sender_incoming_count_24h": len(sender_in_24h),
                    "sender_unique_receivers_30d": len({e[1] for e in sender_out_30d}),
                    "receiver_incoming_count_24h": len(receiver_in_24h),
                    "receiver_incoming_amount_24h": sum(e[2] for e in receiver_in_24h),
                    "receiver_unique_senders_30d": len({e[1] for e in receiver_in_30d}),
                    "previous_reverse_transfer": int(bool(reverse_history)),
                })

            # Update all histories only after evaluating the entire timestamp batch.
            for row in group.itertuples(index=False):
                sender = row.sender_account_id
                receiver = row.receiver_account_id
                amount = float(row.amount)
                event_out = (timestamp, receiver, amount)
                event_in = (timestamp, sender, amount)

                self.outgoing_24h[sender].append(event_out)
                self.incoming_24h[receiver].append(event_in)
                self.outgoing_30d[sender].append(event_out)
                self.incoming_30d[receiver].append(event_in)
                self.pair_history_30d[(sender, receiver)].append(event_out)

            self.features.extend(batch)

        result = pd.DataFrame(self.features)
        result = result.sort_values("original_position").drop(columns="original_position")

        expected_ids = self.transactions["transaction_id"].tolist()
        if result["transaction_id"].tolist() != expected_ids:
            raise RuntimeError("Output transaction order does not match source data.")
        if result.isna().any().any():
            raise RuntimeError("Generated graph features contain missing values.")

        output_path = OUTPUT_DIR / "graph_features.csv"
        result.to_csv(output_path, index=False)

        print("\nGraph feature generation completed.")
        print("Transactions processed:", len(result))
        print("Graph features generated:", len(result.columns) - 1)
        print("Missing values:", int(result.isna().sum().sum()))
        print("Saved:", output_path)
        return result


def main():
    builder = GraphFeatureBuilder()
    features = builder.build_features()
    print("\nGRAPH FEATURE SAMPLE")
    print("=" * 80)
    print(features.head(10).to_string(index=False))
    print("\nGraph feature generation successful!")


if __name__ == "__main__":
    main()

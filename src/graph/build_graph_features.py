
from pathlib import Path
from collections import defaultdict, deque

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TRANSACTION_PATH = ROOT / "data/raw/v2/transactions.csv"
OUTPUT_DIR = ROOT / "data/processed/v2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class GraphFeatureBuilder:

    def __init__(self):

        self.transactions = pd.read_csv(TRANSACTION_PATH)

        self.transactions["timestamp"] = pd.to_datetime(
            self.transactions["timestamp"]
        )

        self.outgoing = defaultdict(deque)
        self.incoming = defaultdict(deque)
        self.pair_history = defaultdict(deque)

        self.features = []

        self.window_24h = pd.Timedelta(hours=24)
        self.window_30d = pd.Timedelta(days=30)

    def remove_old_events(self, history, cutoff):

        while history and history[0][0] <= cutoff:
            history.popleft()

    def build_features(self):

        print("\nBuilding temporal graph features...")

        df = self.transactions.copy()

        df["original_position"] = range(len(df))

        df = df.sort_values(
            ["timestamp", "transaction_id"]
        )

        # Process transactions with identical timestamps as one batch.
        for timestamp, group in df.groupby("timestamp", sort=True):

            cutoff_24h = timestamp - self.window_24h
            cutoff_30d = timestamp - self.window_30d

            batch_features = []

            # Calculate features using strictly earlier transactions.
            for row in group.itertuples(index=False):

                sender = row.sender_account_id
                receiver = row.receiver_account_id

                sender_out = self.outgoing[sender]
                sender_in = self.incoming[sender]

                receiver_in = self.incoming[receiver]
                receiver_out = self.outgoing[receiver]

                sender_pair_history = self.pair_history[
                    (sender, receiver)
                ]

                reverse_pair_history = self.pair_history[
                    (receiver, sender)
                ]

                # Keep only events inside the relevant time windows.
                self.remove_old_events(sender_out, cutoff_24h)
                self.remove_old_events(sender_in, cutoff_24h)
                self.remove_old_events(receiver_in, cutoff_24h)
                self.remove_old_events(receiver_out, cutoff_24h)

                self.remove_old_events(
                    sender_pair_history,
                    cutoff_30d
                )

                self.remove_old_events(
                    reverse_pair_history,
                    cutoff_30d
                )

                # Count distinct counterparties in the previous 30 days.
                sender_unique_receivers = len({
                    event[1]
                    for event in sender_out
                    if event[0] > cutoff_30d
                })

                receiver_unique_senders = len({
                    event[1]
                    for event in receiver_in
                    if event[0] > cutoff_30d
                })

                batch_features.append({
                    "original_position": row.original_position,
                    "transaction_id": row.transaction_id,

                    "sender_outgoing_count_24h": len(sender_out),

                    "sender_outgoing_amount_24h": sum(
                        event[2] for event in sender_out
                    ),

                    "sender_incoming_count_24h": len(sender_in),

                    "sender_unique_receivers_30d": (
                        sender_unique_receivers
                    ),

                    "receiver_incoming_count_24h": len(receiver_in),

                    "receiver_incoming_amount_24h": sum(
                        event[2] for event in receiver_in
                    ),

                    "receiver_unique_senders_30d": (
                        receiver_unique_senders
                    ),

                    "previous_reverse_transfer": int(
                        len(reverse_pair_history) > 0
                    )
                })

            # Update histories only after every transaction at this
            # timestamp has been evaluated.
            for row in group.itertuples(index=False):

                sender = row.sender_account_id
                receiver = row.receiver_account_id
                amount = float(row.amount)

                self.outgoing[sender].append(
                    (timestamp, receiver, amount)
                )

                self.incoming[receiver].append(
                    (timestamp, sender, amount)
                )

                self.pair_history[(sender, receiver)].append(
                    (timestamp, sender, amount)
                )

            self.features.extend(batch_features)

        # Restore the original transaction order.
        result = pd.DataFrame(self.features)

        result = result.sort_values(
            "original_position"
        ).drop(columns="original_position")

        output_path = OUTPUT_DIR / "graph_features.csv"

        result.to_csv(output_path, index=False)

        print("\nGraph feature generation completed.")

        print("Transactions processed:", len(result))

        print("Graph features generated:", len(result.columns) - 1)

        print("Saved:", output_path)

        return result


def main():

    builder = GraphFeatureBuilder()

    features = builder.build_features()

    print("\nGRAPH FEATURE SAMPLE")
    print("=" * 80)

    print(features.head(10).to_string(index=False))

    print("\nMissing values:")
    print(features.isnull().sum())

    print("\nGraph feature generation successful!")


if __name__ == "__main__":
    main()

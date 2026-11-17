
from pathlib import Path

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

TRANSACTION_PATH = ROOT / "data/raw/v2/transactions.csv"

OUTPUT_DIR = ROOT / "data/processed/v2/graph"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class AccountNetwork:

    def __init__(self):

        self.transactions = pd.read_csv(TRANSACTION_PATH)

        self.graph = nx.DiGraph()

        self.validate_data()

    def validate_data(self):

        required_columns = [
            "transaction_id",
            "sender_account_id",
            "receiver_account_id",
            "amount",
            "timestamp"
        ]

        missing = set(required_columns) - set(self.transactions.columns)

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.transactions["amount"] = pd.to_numeric(
            self.transactions["amount"],
            errors="raise"
        )

        if self.transactions["amount"].isna().any():
            raise ValueError("Transaction amounts contain missing values.")

        if (self.transactions["amount"] < 0).any():
            raise ValueError("Negative transaction amounts detected.")

        self.transactions["timestamp"] = pd.to_datetime(
            self.transactions["timestamp"],
            errors="raise"
        )

        if self.transactions[
            ["sender_account_id", "receiver_account_id"]
        ].isna().any().any():
            raise ValueError("Sender or receiver account IDs are missing.")

    def build_graph(self):

        print("\nBuilding account network...")

        for row in self.transactions.itertuples(index=False):

            sender = row.sender_account_id
            receiver = row.receiver_account_id
            amount = float(row.amount)

            # Add account nodes.
            self.graph.add_node(sender)
            self.graph.add_node(receiver)

            # Aggregate repeated transactions between accounts.
            if self.graph.has_edge(sender, receiver):

                edge = self.graph[sender][receiver]

                edge["transaction_count"] += 1
                edge["total_amount"] += amount
                edge["max_amount"] = max(
                    edge["max_amount"],
                    amount
                )

            else:

                self.graph.add_edge(
                    sender,
                    receiver,
                    transaction_count=1,
                    total_amount=amount,
                    max_amount=amount
                )

        print("Account network created successfully.")

        print(f"Total accounts: {self.graph.number_of_nodes()}")
        print(f"Unique directed account relationships: {self.graph.number_of_edges()}")

    def calculate_graph_metrics(self):

        print("\nCalculating graph metrics...")

        graph = self.graph

        # Number of incoming and outgoing relationships.
        in_degree = dict(graph.in_degree())
        out_degree = dict(graph.out_degree())

        # Total incoming and outgoing transaction values.
        in_amount = dict(
            graph.in_degree(weight="total_amount")
        )

        out_amount = dict(
            graph.out_degree(weight="total_amount")
        )

        # PageRank identifies structurally influential accounts.
        pagerank = nx.pagerank(
            graph,
            weight="total_amount"
        )

        # Betweenness centrality measures how often a node
        # lies on shortest paths between other nodes.
        betweenness = nx.betweenness_centrality(
            graph,
            normalized=True,
            weight=None
        )

        # Strongly connected components identify groups
        # where every account can reach every other account.
        component_sizes = {}

        for component_id, component in enumerate(
            nx.strongly_connected_components(graph)
        ):

            for account in component:
                component_sizes[account] = len(component)

        records = []

        for account in graph.nodes():

            records.append({
                "account_id": account,
                "incoming_connections": in_degree[account],
                "outgoing_connections": out_degree[account],
                "total_connections": (
                    in_degree[account] + out_degree[account]
                ),
                "money_received": in_amount[account],
                "money_sent": out_amount[account],
                "net_flow": (
                    in_amount[account] - out_amount[account]
                ),
                "pagerank": pagerank[account],
                "betweenness_centrality": betweenness[account],
                "strongly_connected_component_size": (
                    component_sizes[account]
                )
            })

        metrics = pd.DataFrame(records)

        metrics = metrics.sort_values(
            "total_connections",
            ascending=False
        )

        output_path = OUTPUT_DIR / "account_graph_metrics.csv"

        metrics.to_csv(output_path, index=False)

        print("Graph metrics saved:", output_path)

        return metrics

    def find_reciprocal_relationships(self):

        print("\nFinding reciprocal account relationships...")

        reciprocal_pairs = []

        for sender, receiver in self.graph.edges():

            if (
                sender < receiver
                and self.graph.has_edge(receiver, sender)
            ):

                forward = self.graph[sender][receiver]
                backward = self.graph[receiver][sender]

                reciprocal_pairs.append({
                    "account_a": sender,
                    "account_b": receiver,
                    "a_to_b_amount": forward["total_amount"],
                    "b_to_a_amount": backward["total_amount"],
                    "a_to_b_count": forward["transaction_count"],
                    "b_to_a_count": backward["transaction_count"]
                })

        reciprocal_df = pd.DataFrame(reciprocal_pairs)

        output_path = OUTPUT_DIR / "reciprocal_relationships.csv"

        reciprocal_df.to_csv(output_path, index=False)

        print(f"Reciprocal relationships found: {len(reciprocal_df)}")
        print("Saved:", output_path)

        return reciprocal_df

    
    def detect_communities(self):

        print("\nDetecting account communities...")

        # Convert the directed transaction network into an
        # undirected network for community detection.
        undirected_graph = nx.Graph()

        for sender, receiver, data in self.graph.edges(data=True):

            if undirected_graph.has_edge(sender, receiver):

                undirected_graph[sender][receiver]["transaction_count"] += (
                    data["transaction_count"]
                )

                undirected_graph[sender][receiver]["total_amount"] += (
                    data["total_amount"]
                )

            else:

                undirected_graph.add_edge(
                    sender,
                    receiver,
                    transaction_count=data["transaction_count"],
                    total_amount=data["total_amount"]
                )

        # Identify communities using transaction frequency.
        communities = nx.community.greedy_modularity_communities(
            undirected_graph,
            weight="transaction_count"
        )

        records = []

        for community_id, community in enumerate(communities):

            for account in community:

                records.append({
                    "account_id": account,
                    "community_id": community_id,
                    "community_size": len(community)
                })

        community_df = pd.DataFrame(records)

        output_path = OUTPUT_DIR / "account_communities.csv"

        community_df.to_csv(output_path, index=False)

        print(f"Communities detected: {len(communities)}")

        print("\nCommunity sizes:")

        for community_id, community in enumerate(communities):

            print(
                f"Community {community_id}: "
                f"{len(community)} accounts"
            )

        print("\nCommunity assignments saved:", output_path)

        return community_df

    
    def profile_communities(self, community_df):

        print("\nProfiling account communities...")

        # Map every account to its detected community.
        account_to_community = dict(
            zip(
                community_df["account_id"],
                community_df["community_id"]
            )
        )

        profiles = {}

        for community_id in community_df["community_id"].unique():

            accounts = community_df[
                community_df["community_id"] == community_id
            ]["account_id"].tolist()

            profiles[community_id] = {
                "community_id": community_id,
                "account_count": len(accounts),
                "internal_relationships": 0,
                "internal_transactions": 0,
                "internal_amount": 0.0,
                "external_relationships": 0,
                "external_transactions": 0,
                "external_amount": 0.0
            }

        # Examine every directed relationship.
        for sender, receiver, data in self.graph.edges(data=True):

            sender_community = account_to_community[sender]
            receiver_community = account_to_community[receiver]

            if sender_community == receiver_community:

                profile = profiles[sender_community]

                profile["internal_relationships"] += 1
                profile["internal_transactions"] += (
                    data["transaction_count"]
                )
                profile["internal_amount"] += (
                    data["total_amount"]
                )

            else:

                sender_profile = profiles[sender_community]

                sender_profile["external_relationships"] += 1
                sender_profile["external_transactions"] += (
                    data["transaction_count"]
                )
                sender_profile["external_amount"] += (
                    data["total_amount"]
                )

        # Calculate internal directed graph density.
        for profile in profiles.values():

            n = profile["account_count"]

            possible_edges = n * (n - 1)

            if possible_edges > 0:

                profile["internal_density"] = (
                    profile["internal_relationships"]
                    / possible_edges
                )

            else:
                profile["internal_density"] = 0.0

        summary = pd.DataFrame(list(profiles.values()))

        summary["internal_amount"] = summary[
            "internal_amount"
        ].round(2)

        summary["external_amount"] = summary[
            "external_amount"
        ].round(2)

        summary["internal_density"] = summary[
            "internal_density"
        ].round(4)

        summary = summary.sort_values(
            "internal_density",
            ascending=False
        )

        output_path = OUTPUT_DIR / "community_summary.csv"

        summary.to_csv(output_path, index=False)

        print("\nCOMMUNITY PROFILE")
        print("=" * 100)

        print(summary.to_string(index=False))

        print("\nCommunity profile saved:", output_path)

        return summary



    def visualize_network(self, top_n=40):

        print("\nGenerating account network visualization...")

        # Select the accounts with the highest number of connections.
        degree = dict(self.graph.degree())

        top_accounts = sorted(
            degree,
            key=degree.get,
            reverse=True
        )[:top_n]

        subgraph = self.graph.subgraph(top_accounts).copy()

        if subgraph.number_of_nodes() == 0:
            print("No accounts available for visualization.")
            return

        plt.figure(figsize=(15, 11))

        positions = nx.spring_layout(
            subgraph,
            seed=42,
            weight="total_amount"
        )

        node_sizes = [
            300 + degree[node] * 100
            for node in subgraph.nodes()
        ]

        nx.draw_networkx_nodes(
            subgraph,
            positions,
            node_size=node_sizes
        )

        nx.draw_networkx_edges(
            subgraph,
            positions,
            arrows=True,
            arrowsize=12,
            width=1.2,
            alpha=0.6
        )

        nx.draw_networkx_labels(
            subgraph,
            positions,
            font_size=8
        )

        plt.title("FinGuard AI - Account Transaction Network")
        plt.axis("off")
        plt.tight_layout()

        output_path = OUTPUT_DIR / "account_network.png"

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("Network visualization saved:", output_path)


def main():

    network = AccountNetwork()

    network.build_graph()

    metrics = network.calculate_graph_metrics()

    reciprocal = network.find_reciprocal_relationships()

    communities = network.detect_communities()

    community_profiles = network.profile_communities(
        communities
    )

    network.visualize_network(top_n=40)

    print("\nTOP 10 CONNECTED ACCOUNTS")
    print("=" * 75)

    print(
        metrics.head(10).to_string(index=False)
    )

    print("\nRECIPROCAL RELATIONSHIPS")
    print("=" * 75)

    print(
        reciprocal.head(10).to_string(index=False)
    )


    print("\nCOMMUNITY SUMMARY")
    print(
        communities.groupby("community_id")["account_id"]
        .count()
        .sort_values(ascending=False)
    )

    print("\nAccount graph analytics completed!")


if __name__ == "__main__":
    main()


from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/v2/combined_features.csv"
LABEL_PATH = ROOT / "data/processed/v2/labels.csv"

MODEL_PATH = ROOT / "models/fraud_detection_graph_model.joblib"
THRESHOLD_PATH = ROOT / "models/graph_thresholds.json"

TRANSACTION_PATH = ROOT / "data/raw/v2/transactions.csv"


def select_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    # F1 is calculated at each available threshold.
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    best_index = f1_scores.argmax()

    return float(thresholds[best_index]), float(f1_scores[best_index])


def main():
    
    print("Loading graph-enhanced dataset...")

    data = pd.read_csv(DATA_PATH)
    labels = pd.read_csv(LABEL_PATH)
    transactions = pd.read_csv(TRANSACTION_PATH)

    # Establish the actual chronological transaction order.
    chronological_order = transactions[
        ["transaction_id", "timestamp"]
    ].copy()

    chronological_order["timestamp"] = pd.to_datetime(
        chronological_order["timestamp"]
    )

    chronological_order = chronological_order.sort_values(
        ["timestamp", "transaction_id"]
    )

    # Attach features and labels using transaction IDs.
    data = chronological_order.merge(
        data,
        on="transaction_id",
        how="inner",
        validate="one_to_one",
        sort=False
    )

    data = data.merge(
        labels[["transaction_id", "is_fraud"]],
        on="transaction_id",
        how="inner",
        validate="one_to_one",
        sort=False
    )

    if len(data) != 10000:
        raise ValueError("Unexpected transaction count.")

    if data["is_fraud"].isnull().any():
        raise ValueError("Missing fraud labels.")

    # Remove timestamp and transaction ID from model inputs.
    X = data.drop(
        columns=["transaction_id", "timestamp", "is_fraud"]
    )

    y = data["is_fraud"]


    # Chronological 70/15/15 split.
    train_end = int(len(data) * 0.70)
    validation_end = int(len(data) * 0.85)

    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:validation_end]
    X_test = X.iloc[validation_end:]

    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:validation_end]
    y_test = y.iloc[validation_end:]

    print("\nDATA SPLIT")
    print("=" * 60)
    print("Features:", X.shape[1])
    print("Training:", len(X_train), "| Fraud:", int(y_train.sum()))
    print("Validation:", len(X_val), "| Fraud:", int(y_val.sum()))
    print("Testing:", len(X_test), "| Fraud:", int(y_test.sum()))

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining graph-enhanced Random Forest...")
    model.fit(X_train, y_train)

    # Validation-based model selection and threshold tuning.
    val_probabilities = model.predict_proba(X_val)[:, 1]

    val_pr_auc = average_precision_score(y_val, val_probabilities)
    val_roc_auc = roc_auc_score(y_val, val_probabilities)

    threshold, val_f1 = select_threshold(y_val, val_probabilities)


precision, recall, thresholds = precision_recall_curve(
    y_val, val_probabilities
)

print("\nVALIDATION THRESHOLD ANALYSIS")
print("=" * 65)

for target_recall in [0.60, 0.70, 0.80, 0.90]:
    eligible = [
        i for i, r in enumerate(recall[:-1])
        if r >= target_recall
    ]

    if not eligible:
        print(f"Recall target {target_recall:.0%}: unavailable")
        continue

    # Choose the highest-precision threshold meeting the recall target.
    best_index = max(
        eligible,
        key=lambda i: precision[i]
    )

    print(
        f"Target recall: {target_recall:.0%} | "
        f"Threshold: {thresholds[best_index]:.4f} | "
        f"Precision: {precision[best_index]:.4f} | "
        f"Recall: {recall[best_index]:.4f}"
    )


    print("\nVALIDATION RESULTS")
    print("=" * 60)
    print(f"PR-AUC: {val_pr_auc:.4f}")
    print(f"ROC-AUC: {val_roc_auc:.4f}")
    print(f"Selected threshold: {threshold:.4f}")
    print(f"Validation F1: {val_f1:.4f}")

    # Final evaluation on untouched test data.
    test_probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (test_probabilities >= threshold).astype(int)

    test_pr_auc = average_precision_score(y_test, test_probabilities)
    test_roc_auc = roc_auc_score(y_test, test_probabilities)

    print("\nFINAL TEST RESULTS")
    print("=" * 60)
    print(f"PR-AUC: {test_pr_auc:.4f}")
    print(f"ROC-AUC: {test_roc_auc:.4f}")
    print(f"Threshold: {threshold:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Fraud"],
        zero_division=0,
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Save model and decision threshold.
    joblib.dump(model, MODEL_PATH)

    with open(THRESHOLD_PATH, "w") as file:
        json.dump({
            "threshold": threshold,
            "validation_pr_auc": val_pr_auc,
            "validation_roc_auc": val_roc_auc,
            "validation_f1": val_f1,
            "test_pr_auc": test_pr_auc,
            "test_roc_auc": test_roc_auc,
            "feature_count": X.shape[1],
        }, file, indent=4)

    print("\nSaved model:", MODEL_PATH)
    print("Saved thresholds:", THRESHOLD_PATH)


if __name__ == "__main__":
    main()

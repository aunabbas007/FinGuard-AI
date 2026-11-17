from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/processed/v2/combined_features.csv"
LABEL_PATH = ROOT / "data/processed/v2/labels.csv"
TRANSACTION_PATH = ROOT / "data/raw/v2/transactions.csv"
MODEL_PATH = ROOT / "models/fraud_detection_graph_model.joblib"
THRESHOLD_PATH = ROOT / "models/graph_thresholds.json"
RANDOM_STATE = 42


def select_f1_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )
    if len(thresholds) == 0:
        raise ValueError("No classification thresholds could be calculated.")

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )
    index = int(f1_scores.argmax())
    return float(thresholds[index]), float(f1_scores[index])


def report_recall_targets(y_true, probabilities, targets=(0.60, 0.70, 0.80, 0.90)):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )
    print("\nVALIDATION OPERATING-POINT OPTIONS")
    print("=" * 72)

    for target in targets:
        eligible = [i for i, value in enumerate(recall[:-1]) if value >= target]
        if not eligible:
            print(f"Recall >= {target:.0%}: unavailable")
            continue

        # Highest precision among thresholds that satisfy the recall target.
        index = max(eligible, key=lambda i: precision[i])
        print(
            f"Recall target {target:.0%} | "
            f"threshold={thresholds[index]:.4f} | "
            f"precision={precision[index]:.4f} | "
            f"recall={recall[index]:.4f}"
        )


def load_chronological_dataset():
    features = pd.read_csv(DATA_PATH)
    labels = pd.read_csv(LABEL_PATH)
    transactions = pd.read_csv(TRANSACTION_PATH, usecols=["transaction_id", "timestamp"])

    for frame, name in ((features, "features"), (labels, "labels"), (transactions, "transactions")):
        if "transaction_id" not in frame.columns:
            raise ValueError(f"{name} is missing transaction_id.")
        if frame["transaction_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate transaction IDs.")

    order = transactions.copy()
    order["timestamp"] = pd.to_datetime(order["timestamp"], errors="raise")
    order = order.sort_values(["timestamp", "transaction_id"])

    data = order.merge(
        features, on="transaction_id", how="inner",
        validate="one_to_one", sort=False
    )
    data = data.merge(
        labels[["transaction_id", "is_fraud"]],
        on="transaction_id", how="inner",
        validate="one_to_one", sort=False
    )

    if len(data) != len(transactions) or len(data) != len(features) or len(data) != len(labels):
        raise ValueError("Transactions, features, and labels do not contain the same IDs.")

    if data["is_fraud"].isna().any():
        raise ValueError("Fraud labels contain missing values.")

    X = data.drop(columns=["transaction_id", "timestamp", "is_fraud"])
    y = data["is_fraud"].astype(int)

    if X.isna().any().any():
        raise ValueError("Feature matrix contains missing values.")
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("is_fraud must contain only binary values 0 and 1.")

    return X, y


def main():
    print("Loading graph-enhanced dataset...")
    X, y = load_chronological_dataset()

    n = len(X)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    X_train, X_val, X_test = (
        X.iloc[:train_end],
        X.iloc[train_end:validation_end],
        X.iloc[validation_end:],
    )
    y_train, y_val, y_test = (
        y.iloc[:train_end],
        y.iloc[train_end:validation_end],
        y.iloc[validation_end:],
    )

    if y_train.nunique() < 2 or y_val.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Each chronological split must contain both classes.")

    print("\nCHRONOLOGICAL DATA SPLIT")
    print("=" * 60)
    print(f"Features: {X.shape[1]}")
    print(f"Training: {len(X_train)} | Fraud: {int(y_train.sum())}")
    print(f"Validation: {len(X_val)} | Fraud: {int(y_val.sum())}")
    print(f"Testing: {len(X_test)} | Fraud: {int(y_test.sum())}")

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=10,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining graph-enhanced Random Forest...")
    model.fit(X_train, y_train)

    val_probabilities = model.predict_proba(X_val)[:, 1]
    val_pr_auc = average_precision_score(y_val, val_probabilities)
    val_roc_auc = roc_auc_score(y_val, val_probabilities)
    threshold, val_f1 = select_f1_threshold(y_val, val_probabilities)

    print("\nVALIDATION RESULTS")
    print("=" * 60)
    print(f"PR-AUC: {val_pr_auc:.4f}")
    print(f"ROC-AUC: {val_roc_auc:.4f}")
    print(f"F1-selected threshold: {threshold:.4f}")
    print(f"Validation F1: {val_f1:.4f}")
    report_recall_targets(y_val, val_probabilities)

    # The test set is evaluated once at the preselected validation threshold.
    test_probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (test_probabilities >= threshold).astype(int)
    test_pr_auc = average_precision_score(y_test, test_probabilities)
    test_roc_auc = roc_auc_score(y_test, test_probabilities)

    print("\nFINAL TEST RESULTS (F1-SELECTED THRESHOLD)")
    print("=" * 60)
    print(f"PR-AUC: {test_pr_auc:.4f}")
    print(f"ROC-AUC: {test_roc_auc:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(classification_report(
        y_test, predictions, labels=[0, 1],
        target_names=["Normal", "Fraud"], zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    config = {
        "model_name": "RandomForestClassifier",
        "feature_count": int(X.shape[1]),
        "feature_names": list(X.columns),
        "threshold": threshold,
        "threshold_selection": "maximum validation F1",
        "validation_pr_auc": float(val_pr_auc),
        "validation_roc_auc": float(val_roc_auc),
        "validation_f1": float(val_f1),
        "test_pr_auc": float(test_pr_auc),
        "test_roc_auc": float(test_roc_auc),
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "random_state": RANDOM_STATE,
        "note": "Synthetic dataset; scores are not calibrated real-world fraud probabilities.",
    }
    THRESHOLD_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\nSaved model:", MODEL_PATH)
    print("Saved threshold configuration:", THRESHOLD_PATH)


if __name__ == "__main__":
    main()

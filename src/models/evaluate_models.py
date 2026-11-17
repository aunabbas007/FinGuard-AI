"""Fair, reproducible comparison of 36-feature and graph-enhanced fraud models.

Run from the repository root after generating V2 data and both feature matrices:
    python src/models/evaluate_models.py

Both models use identical transaction IDs and chronological 70/15/15 boundaries.
Thresholds are selected on validation data only. The test set is never used for
model or threshold selection. Synthetic scores are not calibrated probabilities.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
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
DATA = ROOT / "data" / "processed" / "v2"
RAW = ROOT / "data" / "raw" / "v2"
MODELS = ROOT / "models"
RANDOM_STATE = 42
SPLIT = (0.70, 0.15, 0.15)


def load_aligned_datasets():
    transactions = pd.read_csv(RAW / "transactions.csv")
    labels = pd.read_csv(DATA / "labels.csv")
    baseline = pd.read_csv(DATA / "features.csv")
    graph = pd.read_csv(DATA / "combined_features.csv")

    for frame, name in (
        (transactions, "transactions"),
        (labels, "labels"),
        (graph, "combined features"),
    ):
        if "transaction_id" not in frame:
            raise ValueError(f"{name} is missing transaction_id.")
        if frame["transaction_id"].isna().any():
            raise ValueError(f"{name} contains null transaction IDs.")
        if frame["transaction_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate transaction IDs.")

    if len(transactions) != len(labels) or len(transactions) != len(baseline):
        raise ValueError("Raw transactions, labels, and baseline features differ in row count.")
    if labels["transaction_id"].tolist() != transactions["transaction_id"].tolist():
        raise ValueError("Label IDs do not match raw transaction order.")
    if graph["transaction_id"].nunique() != len(transactions):
        raise ValueError("Graph feature IDs are not one-to-one.")
    if set(graph["transaction_id"]) != set(transactions["transaction_id"]):
        raise ValueError("Graph feature IDs do not match raw transaction IDs.")

    baseline = baseline.copy()
    baseline.insert(0, "transaction_id", labels["transaction_id"].to_numpy())

    order = transactions[["transaction_id", "timestamp"]].copy()
    order["timestamp"] = pd.to_datetime(order["timestamp"], errors="raise")
    order = order.sort_values(["timestamp", "transaction_id"], kind="mergesort")

    base = order.merge(
        baseline, on="transaction_id", how="left",
        validate="one_to_one", sort=False
    ).merge(
        labels[["transaction_id", "is_fraud"]],
        on="transaction_id", how="left",
        validate="one_to_one", sort=False
    )
    enhanced = order.merge(
        graph, on="transaction_id", how="left",
        validate="one_to_one", sort=False
    ).merge(
        labels[["transaction_id", "is_fraud"]],
        on="transaction_id", how="left",
        validate="one_to_one", sort=False
    )

    if base["transaction_id"].tolist() != enhanced["transaction_id"].tolist():
        raise RuntimeError("Baseline and graph datasets are not identically ordered.")
    if base[["is_fraud"]].isna().any().any() or enhanced.isna().any().any():
        raise ValueError("Missing labels or feature values after alignment.")

    y_base = base["is_fraud"].astype(int)
    y_graph = enhanced["is_fraud"].astype(int)
    if not y_base.equals(y_graph):
        raise RuntimeError("Baseline and graph labels are not identical.")
    if not set(y_base.unique()).issubset({0, 1}):
        raise ValueError("is_fraud must contain only binary 0/1 labels.")

    X_base = base.drop(columns=["transaction_id", "timestamp", "is_fraud"])
    X_graph = enhanced.drop(columns=["transaction_id", "timestamp", "is_fraud"])
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_base.dtypes):
        raise TypeError("Baseline features must all be numeric.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_graph.dtypes):
        raise TypeError("Graph features must all be numeric.")
    if not np.isfinite(X_base.to_numpy(dtype=float)).all():
        raise ValueError("Baseline features contain NaN or infinite values.")
    if not np.isfinite(X_graph.to_numpy(dtype=float)).all():
        raise ValueError("Graph features contain NaN or infinite values.")

    return X_base, X_graph, y_base, order["transaction_id"].reset_index(drop=True)


def chronological_indices(n_rows: int):
    if n_rows < 3:
        raise ValueError("At least three rows are required.")
    train_end = int(n_rows * SPLIT[0])
    validation_end = int(n_rows * (SPLIT[0] + SPLIT[1]))
    if not (0 < train_end < validation_end < n_rows):
        raise ValueError("Chronological split produced an empty partition.")
    return slice(0, train_end), slice(train_end, validation_end), slice(validation_end, n_rows)


def choose_f1_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        raise ValueError("Validation data did not produce any thresholds.")
    f1 = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def threshold_options(y_true, probabilities, recall_targets=(0.60, 0.70, 0.80, 0.90)):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    options = {}
    for target in recall_targets:
        candidates = np.flatnonzero(recall[:-1] >= target)
        if len(candidates) == 0:
            options[str(target)] = None
            continue
        best = int(candidates[np.argmax(precision[candidates])])
        options[str(target)] = {
            "threshold": float(thresholds[best]),
            "precision": float(precision[best]),
            "recall": float(recall[best]),
        }
    return options


def evaluate_one(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    val_prob = model.predict_proba(X_val)[:, 1]
    threshold, val_f1 = choose_f1_threshold(y_val, val_prob)
    test_prob = model.predict_proba(X_test)[:, 1]
    pred = (test_prob >= threshold).astype(int)
    return {
        "name": name,
        "model": model,
        "threshold": threshold,
        "validation": {
            "pr_auc": float(average_precision_score(y_val, val_prob)),
            "roc_auc": float(roc_auc_score(y_val, val_prob)),
            "f1_at_selected_threshold": val_f1,
            "threshold_options": threshold_options(y_val, val_prob),
        },
        "test": {
            "pr_auc": float(average_precision_score(y_test, test_prob)),
            "roc_auc": float(roc_auc_score(y_test, test_prob)),
            "classification_report": classification_report(
                y_test, pred, labels=[0, 1],
                target_names=["normal", "fraud"], output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_test, pred, labels=[0, 1]
            ).tolist(),
        },
        "test_predictions": pred,
    }


def main():
    X_base, X_graph, y, transaction_ids = load_aligned_datasets()
    train_idx, val_idx, test_idx = chronological_indices(len(y))

    y_train, y_val, y_test = y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]
    for name, labels in (("train", y_train), ("validation", y_val), ("test", y_test)):
        if labels.nunique() != 2:
            raise ValueError(f"{name} split must contain both fraud and normal examples.")

    print("Chronological, identical evaluation population")
    print(f"Rows: {len(y)} | train: {len(y_train)} | validation: {len(y_val)} | test: {len(y_test)}")
    print(f"Fraud counts: train={int(y_train.sum())}, validation={int(y_val.sum())}, test={int(y_test.sum())}")
    print(f"Baseline features: {X_base.shape[1]} | graph-enhanced features: {X_graph.shape[1]}")

    params = dict(
        n_estimators=300, class_weight="balanced", max_depth=10,
        min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1,
    )
    common = (
        X_base.iloc[train_idx], y_train,
        X_base.iloc[val_idx], y_val,
        X_base.iloc[test_idx], y_test,
    )
    baseline_result = evaluate_one(
        "Random Forest (baseline)", RandomForestClassifier(**params), *common
    )
    graph_result = evaluate_one(
        "Random Forest (graph-enhanced)",
        RandomForestClassifier(**params),
        X_graph.iloc[train_idx], y_train,
        X_graph.iloc[val_idx], y_val,
        X_graph.iloc[test_idx], y_test,
    )

    summary = {}
    for result in (baseline_result, graph_result):
        print(f"\n{result['name']}")
        print(f"Validation PR-AUC: {result['validation']['pr_auc']:.4f}")
        print(f"Test PR-AUC: {result['test']['pr_auc']:.4f}")
        print(f"Test ROC-AUC: {result['test']['roc_auc']:.4f}")
        print(f"Validation-selected threshold: {result['threshold']:.6f}")
        print("Test fraud metrics:", result["test"]["classification_report"]["fraud"])
        print("Test confusion matrix [ [TN, FP], [FN, TP] ]:", result["test"]["confusion_matrix"])
        summary[result["name"]] = {
            "feature_count": int(
                X_base.shape[1] if result is baseline_result else X_graph.shape[1]
            ),
            "threshold": result["threshold"],
            "validation": result["validation"],
            "test": result["test"],
        }

    # Save reproducible artifacts for both experiments, not just the winner.
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline_result["model"], MODELS / "fraud_detection_model.joblib")
    joblib.dump(graph_result["model"], MODELS / "fraud_detection_graph_model.joblib")
    (MODELS / "ml_evaluation.json").write_text(
        json.dumps({
            "dataset": "synthetic FinGuard AI V2",
            "random_state": RANDOM_STATE,
            "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "split_method": "timestamp ascending, transaction_id ascending; same IDs for both models",
            "transaction_counts": {
                "total": len(y), "train": len(y_train),
                "validation": len(y_val), "test": len(y_test),
            },
            "fraud_counts": {
                "train": int(y_train.sum()),
                "validation": int(y_val.sum()),
                "test": int(y_test.sum()),
            },
            "results": summary,
            "limitations": [
                "Synthetic labels are generated by injected rules.",
                "Model scores are not calibrated real-world fraud probabilities.",
                "Test metrics are for reporting only and must not guide further tuning.",
            ],
        }, indent=2),
        encoding="utf-8",
    )

    # Graph feature importances are useful for diagnostic prioritisation, not causal claims.
    importance = pd.DataFrame({
        "feature": X_graph.columns,
        "importance": graph_result["model"].feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(DATA / "graph_feature_importance.csv", index=False)
    print("\nSaved: models/ml_evaluation.json")
    print("Saved: models/fraud_detection_model.joblib")
    print("Saved: models/fraud_detection_graph_model.joblib")
    print("Saved: data/processed/v2/graph_feature_importance.csv")


if __name__ == "__main__":
    main()

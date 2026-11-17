
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = ROOT / "data/processed/v2/features.csv"
LABEL_PATH = ROOT / "data/processed/v2/labels.csv"
TRANSACTION_PATH = ROOT / "data/raw/v2/transactions.csv"

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading FinGuard AI dataset...")

X = pd.read_csv(FEATURE_PATH)
labels = pd.read_csv(LABEL_PATH)
transactions = pd.read_csv(TRANSACTION_PATH)

y = labels["is_fraud"]

if not (len(X) == len(y) == len(transactions)):
    raise ValueError("Features, labels and transactions have different row counts.")

print(f"Total transactions: {len(X)}")
print(f"Total fraud cases: {int(y.sum())}")


# --------------------------------------------------
# 3. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------

# Recover the original row positions after sorting transactions.
order = (
    transactions
    .assign(_original_position=np.arange(len(transactions)))
    .sort_values(["timestamp", "transaction_id"])["_original_position"]
    .to_numpy()
)

X = X.iloc[order].reset_index(drop=True)
y = y.iloc[order].reset_index(drop=True)

n = len(X)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:validation_end]
y_val = y.iloc[train_end:validation_end]

X_test = X.iloc[validation_end:]
y_test = y.iloc[validation_end:]

print("\nChronological data split:")
print(f"Training:   {len(X_train)} transactions | {int(y_train.sum())} fraud")
print(f"Validation: {len(X_val)} transactions | {int(y_val.sum())} fraud")
print(f"Testing:    {len(X_test)} transactions | {int(y_test.sum())} fraud")


# --------------------------------------------------
# 4. DEFINE MODELS
# --------------------------------------------------

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

if positive_count == 0:
    raise ValueError("Training data contains no fraud examples.")

class_weight_ratio = negative_count / positive_count

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=10,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=class_weight_ratio,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# --------------------------------------------------
# 5. SELECT THRESHOLD USING VALIDATION DATA
# --------------------------------------------------

def find_best_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    if len(thresholds) == 0:
        raise ValueError("Unable to calculate classification thresholds.")

    # Each threshold corresponds to precision[:-1] and recall[:-1].
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    best_index = int(np.argmax(f1_scores))

    return float(thresholds[best_index]), float(f1_scores[best_index])


# --------------------------------------------------
# 6. TRAIN AND COMPARE MODELS
# --------------------------------------------------

results = {}

print("\nTraining models and evaluating validation performance...")

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    # Validation probabilities
    val_probabilities = model.predict_proba(X_val)[:, 1]

    # Select threshold using validation data only
    threshold, validation_f1 = find_best_threshold(
        y_val,
        val_probabilities
    )

    validation_pr_auc = average_precision_score(
        y_val,
        val_probabilities
    )

    validation_roc_auc = roc_auc_score(
        y_val,
        val_probabilities
    )

    results[name] = {
        "model": model,
        "threshold": threshold,
        "validation_f1": validation_f1,
        "validation_pr_auc": validation_pr_auc,
        "validation_roc_auc": validation_roc_auc
    }

    print(f"Validation PR-AUC: {validation_pr_auc:.4f}")
    print(f"Validation ROC-AUC: {validation_roc_auc:.4f}")
    print(f"Selected threshold: {threshold:.4f}")
    print(f"Validation F1: {validation_f1:.4f}")


# --------------------------------------------------
# 7. SELECT BEST MODEL USING VALIDATION PR-AUC
# --------------------------------------------------

best_name = max(
    results,
    key=lambda name: results[name]["validation_pr_auc"]
)

best_model = results[best_name]["model"]
best_threshold = results[best_name]["threshold"]

print("\n" + "=" * 55)
print("SELECTED MODEL")
print("=" * 55)

print(f"Model: {best_name}")
print(f"Validation PR-AUC: {results[best_name]['validation_pr_auc']:.4f}")
print(f"Decision threshold: {best_threshold:.4f}")


# --------------------------------------------------
# 8. FINAL EVALUATION ON UNTOUCHED TEST DATA
# --------------------------------------------------

print("\nEvaluating selected model on untouched test data...")

test_probabilities = best_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

test_pr_auc = average_precision_score(
    y_test,
    test_probabilities
)

test_roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

print("\nFinal Test Results")
print("-" * 55)

print(f"PR-AUC:  {test_pr_auc:.4f}")
print(f"ROC-AUC: {test_roc_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Normal", "Fraud"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_predictions))


# --------------------------------------------------
# 9. SAVE MODEL AND THRESHOLD
# --------------------------------------------------

model_path = MODEL_DIR / "fraud_detection_model.joblib"
threshold_path = MODEL_DIR / "thresholds.json"

joblib.dump(best_model, model_path)

threshold_config = {
    "model_name": best_name,
    "threshold": best_threshold,
    "validation_pr_auc": results[best_name]["validation_pr_auc"],
    "validation_f1": results[best_name]["validation_f1"],
    "test_pr_auc": float(test_pr_auc),
    "test_roc_auc": float(test_roc_auc)
}

with open(threshold_path, "w") as file:
    json.dump(threshold_config, file, indent=4)

print("\nSaved model:", model_path)
print("Saved threshold configuration:", threshold_path)

print("\nFinGuard AI baseline training completed!")

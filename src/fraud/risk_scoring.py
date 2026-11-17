
from pathlib import Path
import json

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models/fraud_detection_model.joblib"
THRESHOLD_PATH = ROOT / "models/thresholds.json"


class FraudRiskScorer:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError("Trained fraud model not found.")

        if not THRESHOLD_PATH.exists():
            raise FileNotFoundError("Threshold configuration not found.")

        self.model = joblib.load(MODEL_PATH)

        with open(THRESHOLD_PATH, "r") as file:
            self.config = json.load(file)

        self.threshold = self.config["threshold"]

        # Recover the feature order expected by the trained model.
        if hasattr(self.model, "feature_names_in_"):
            self.feature_names = list(self.model.feature_names_in_)
        else:
            self.feature_names = None

    def calculate_risk_score(self, probability):

        return round(float(probability) * 100, 2)

    def classify_risk(self, risk_score):

        if risk_score >= 75:
            return "CRITICAL"

        elif risk_score >= 50:
            return "HIGH"

        elif risk_score >= 25:
            return "MEDIUM"

        return "LOW"

    def get_recommendation(self, risk_level, is_flagged):

        if not is_flagged:
            return "No fraud alert triggered; continue monitoring."

        recommendations = {
            "MEDIUM": "Queue for review.",
            "HIGH": "Prioritise fraud investigation.",
            "CRITICAL": "Urgent human investigation required."
        }

        return recommendations.get(
            risk_level,
            "Manual investigation recommended."
        )

    def score_transactions(self, features):

        if self.feature_names is not None:

            missing = set(self.feature_names) - set(features.columns)
            extra = set(features.columns) - set(self.feature_names)

            if missing or extra:
                raise ValueError(
                    f"Feature mismatch. Missing: {missing}, Extra: {extra}"
                )

            features = features[self.feature_names]

        probabilities = self.model.predict_proba(features)[:, 1]

        results = []

        for probability in probabilities:

            risk_score = self.calculate_risk_score(probability)

            risk_level = self.classify_risk(risk_score)

            is_flagged = bool(probability >= self.threshold)

            recommendation = self.get_recommendation(
                risk_level,
                is_flagged
            )

            results.append({
                "model_probability": round(float(probability), 4),
                "risk_score": risk_score,
                "risk_level": risk_level,
                "is_flagged": is_flagged,
                "recommendation": recommendation
            })

        return pd.DataFrame(results)

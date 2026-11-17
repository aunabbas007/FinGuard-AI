import json
from functools import lru_cache
from pathlib import Path
from threading import Lock
import joblib
import pandas as pd

class ModelUnavailableError(RuntimeError):
    pass

class FeatureValidationError(ValueError):
    pass

class FraudModelService:
    def __init__(self, model_path: Path, threshold_path: Path):
        self.model_path, self.threshold_path = Path(model_path), Path(threshold_path)
        self._model = self._threshold = None
        self._lock = Lock()

    @property
    def is_available(self):
        return self.model_path.is_file() and self.threshold_path.is_file()

    def _load(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.is_available:
                raise ModelUnavailableError("Model artifacts are missing. Run the ML evaluation locally and mount the generated models directory.")
            try:
                model = joblib.load(self.model_path)
                config = json.loads(self.threshold_path.read_text(encoding="utf-8"))
                threshold = float(config["threshold"])
            except Exception as exc:
                raise ModelUnavailableError(f"Model artifacts could not be loaded: {exc}") from exc
            if not 0 <= threshold <= 1 or not hasattr(model, "predict_proba"):
                raise ModelUnavailableError("Model or threshold configuration is invalid.")
            self._model, self._threshold = model, threshold

    @property
    def feature_names(self):
        self._load()
        names = getattr(self._model, "feature_names_in_", None)
        if names is None:
            raise ModelUnavailableError("Model has no feature-name metadata.")
        return [str(x) for x in names]

    def predict(self, features: dict[str, float]):
        expected = self.feature_names
        missing, extra = sorted(set(expected)-set(features)), sorted(set(features)-set(expected))
        if missing or extra:
            raise FeatureValidationError(f"Feature schema mismatch. Missing: {missing}; unexpected: {extra}")
        row = pd.DataFrame([[float(features[n]) for n in expected]], columns=expected)
        probability = float(self._model.predict_proba(row)[0, 1])
        score = round(probability * 100, 2)
        level = "CRITICAL" if score >= 75 else "HIGH" if score >= 50 else "MEDIUM" if score >= 25 else "LOW"
        flagged = probability >= self._threshold
        advice = ("Urgent human investigation required." if flagged and level == "CRITICAL"
                  else "Prioritise fraud investigation." if flagged and level == "HIGH"
                  else "Queue for human review." if flagged
                  else "No fraud alert triggered; continue monitoring.")
        importances = getattr(self._model, "feature_importances_", None)
        explanation = []
        if importances is not None:
            ranked = sorted(zip(expected, importances), key=lambda x: float(x[1]), reverse=True)[:8]
            explanation = [{"feature": n, "importance": round(float(v), 6)} for n, v in ranked]
        return {"model_probability": round(probability, 6), "risk_score": score,
                "risk_level": level, "is_flagged": bool(flagged), "recommendation": advice,
                "model_name": type(self._model).__name__, "explanation": explanation}

@lru_cache
def get_model_service():
    from src.api.config import get_settings
    s = get_settings()
    return FraudModelService(s.model_path, s.threshold_path)

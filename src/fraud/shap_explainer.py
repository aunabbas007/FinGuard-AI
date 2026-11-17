
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models/fraud_detection_model.joblib"
FEATURE_PATH = ROOT / "data/processed/v2/features.csv"

OUTPUT_DIR = ROOT / "data/processed/v2/shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class FraudExplainer:

    def __init__(self):

        self.model = joblib.load(MODEL_PATH)

        self.features = pd.read_csv(FEATURE_PATH)

        self.feature_names = list(self.model.feature_names_in_)

        self.features = self.features[self.feature_names]

        # Background data helps SHAP establish a reference prediction.
        self.background = self.features.sample(
            n=min(100, len(self.features)),
            random_state=42
        )

        self.explainer = shap.TreeExplainer(
            self.model,
            data=self.background,
            feature_perturbation="interventional",
            model_output="probability"
        )

    def explain_transactions(self, number_of_transactions=10):

        # Select transactions with the highest predicted fraud probability.
        probabilities = self.model.predict_proba(
            self.features
        )[:, 1]

        selected_indices = np.argsort(probabilities)[
            -number_of_transactions:
        ][::-1]

        sample = self.features.iloc[selected_indices]

        sample_probabilities = probabilities[selected_indices]

        shap_values = self.explainer.shap_values(sample)

        # Handle different SHAP output formats.
        if isinstance(shap_values, list):
            positive_class_values = shap_values[1]

        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            positive_class_values = shap_values[:, :, 1]

        else:
            positive_class_values = shap_values

        print("\nFINGUARD AI - SHAP EXPLANATIONS")
        print("=" * 65)

        for row_number in range(len(sample)):

            contributions = positive_class_values[row_number]

            explanation = pd.DataFrame({
                "feature": self.feature_names,
                "contribution": contributions
            })

            explanation["absolute_contribution"] = (
                explanation["contribution"].abs()
            )

            explanation = explanation.sort_values(
                "absolute_contribution",
                ascending=False
            )

            print(f"\nTransaction {row_number + 1}")
            print(
                f"Fraud probability: "
                f"{sample_probabilities[row_number]:.4f}"
            )

            print("\nTop contributing features:")

            print(
                explanation[
                    ["feature", "contribution"]
                ].head(5).to_string(index=False)
            )

        # Global feature importance summary.
        shap.summary_plot(
            positive_class_values,
            sample,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "shap_summary.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        # Individual explanation for the highest-risk transaction.
        base_value = np.asarray(self.explainer.expected_value).reshape(-1)

        if len(base_value) > 1:
            base_value = base_value[1]
        else:
            base_value = base_value[0]

        waterfall_explanation = shap.Explanation(
            values=positive_class_values[0],
            base_values=float(base_value),
            data=sample.iloc[0].values,
            feature_names=self.feature_names
        )

        shap.plots.waterfall(
            waterfall_explanation,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "highest_risk_explanation.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("\nSHAP explanation completed.")
        print("Charts saved in:", OUTPUT_DIR)


if __name__ == "__main__":

    explainer = FraudExplainer()

    explainer.explain_transactions(
        number_of_transactions=10
    )

from pathlib import Path
from secrets import randbelow
import pandas as pd
from fastapi import APIRouter, HTTPException
from src.api.config import ROOT_DIR
from src.api.services.model_service import ModelUnavailableError, get_model_service

router = APIRouter(prefix="/api/v1/demo", tags=["demo"])

@router.get("/sample")
def demo_sample():
    """Return one synthetic feature row using the exact trained model schema."""
    raw_path = ROOT_DIR / "data" / "raw" / "v2" / "transactions.csv"
    feature_path = ROOT_DIR / "data" / "processed" / "v2" / "features.csv"
    if not raw_path.is_file() or not feature_path.is_file():
        raise HTTPException(status_code=503, detail="Synthetic demo dataset is not available in this deployment.")
    try:
        names = get_model_service().feature_names
        raw = pd.read_csv(raw_path)
        matrix = pd.read_csv(feature_path)
        if len(raw) != len(matrix) or len(raw) == 0:
            raise ValueError("Synthetic transaction and feature datasets are not aligned.")
        if set(names) != set(matrix.columns):
            raise ValueError("Synthetic feature schema does not match the loaded model.")
        index = randbelow(len(raw))
        transaction = raw.iloc[index]
        vector = matrix.iloc[index]
        return {
            "transaction": {
                "external_id": str(transaction["transaction_id"]),
                "amount": float(transaction["amount"]),
                "currency": "INR",
                "sender_account": str(transaction["sender_account_id"]),
                "receiver_account": str(transaction["receiver_account_id"]),
                "channel": str(transaction["channel"]),
            },
            "features": {name: float(vector[name]) for name in names},
            "dataset_notice": "This is a randomly selected synthetic transaction, not real banking data.",
        }
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load demo transaction: {exc}") from exc

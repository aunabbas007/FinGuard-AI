from fastapi import APIRouter, HTTPException
from src.api.services.model_service import ModelUnavailableError, get_model_service

router = APIRouter(prefix="/api/v1/model", tags=["model"])

@router.get("/schema")
def model_schema():
    service = get_model_service()
    try:
        names = service.feature_names
        threshold = service.threshold
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"model": "RandomForestClassifier", "feature_count": len(names),
            "features": names, "threshold": threshold,
            "notice": "Model scores are trained on synthetic data and are not real-world calibrated probabilities."}

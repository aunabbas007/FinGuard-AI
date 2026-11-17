from fastapi import APIRouter
from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi import Depends
from src.api.db import get_db
from src.api.services.model_service import get_model_service

router = APIRouter(tags=["system"])

@router.get("/health")
def health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    model = get_model_service()
    return {"status": "healthy", "service": "FinGuard AI API",
            "database": "connected", "model_artifacts": "available" if model.is_available else "missing"}

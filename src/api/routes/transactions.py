from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from src.api.db import get_db
from src.api.models import Transaction
from src.api.schemas import TransactionCreate, TransactionPage, TransactionRead
from src.api.services.model_service import FeatureValidationError, ModelUnavailableError, get_model_service

router = APIRouter(prefix="/api/v1/transactions", tags=["transactions"])

@router.post("/score", response_model=TransactionRead, status_code=201)
def score_transaction(payload: TransactionCreate, db: Session = Depends(get_db)):
    try:
        result = get_model_service().predict(payload.features)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FeatureValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    item = Transaction(**payload.model_dump(exclude={"features"}),
                       features=payload.features, **result)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item

@router.get("", response_model=TransactionPage)
def list_transactions(limit: int = Query(25, ge=1, le=100),
                      offset: int = Query(0, ge=0),
                      flagged_only: bool = False,
                      db: Session = Depends(get_db)):
    query = select(Transaction)
    count_query = select(func.count()).select_from(Transaction)
    if flagged_only:
        query = query.where(Transaction.is_flagged.is_(True))
        count_query = count_query.where(Transaction.is_flagged.is_(True))
    total = db.scalar(count_query) or 0
    items = db.scalars(query.order_by(Transaction.created_at.desc()).offset(offset).limit(limit)).all()
    return {"items": items, "total": total, "limit": limit, "offset": offset}

@router.get("/{transaction_id}", response_model=TransactionRead)
def get_transaction(transaction_id: int, db: Session = Depends(get_db)):
    item = db.get(Transaction, transaction_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return item

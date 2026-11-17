from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field

class TransactionCreate(BaseModel):
    external_id: str | None = Field(default=None, max_length=100)
    amount: float = Field(gt=0, le=1_000_000_000)
    currency: str = Field(default="INR", min_length=3, max_length=8)
    sender_account: str = Field(min_length=1, max_length=100)
    receiver_account: str = Field(min_length=1, max_length=100)
    channel: str = Field(default="web", max_length=30)
    features: dict[str, float] = Field(description="Numeric feature values matching the trained model's feature schema.")

class TransactionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    external_id: str | None
    amount: float
    currency: str
    sender_account: str
    receiver_account: str
    channel: str
    risk_score: float | None
    risk_level: str
    is_flagged: bool | None
    model_probability: float | None
    recommendation: str | None
    model_name: str | None
    created_at: datetime

class TransactionPage(BaseModel):
    items: list[TransactionRead]
    total: int
    limit: int
    offset: int

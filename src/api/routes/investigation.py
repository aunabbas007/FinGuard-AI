from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from src.api.services.investigation_agent import investigate

router = APIRouter(prefix="/api/v1/investigation", tags=["investigation"])

class InvestigationRequest(BaseModel):
    message: str = Field(min_length=3, max_length=4000)

class InvestigationResponse(BaseModel):
    answer: str
    agent: str = "LangChain"
    retrieval: str = "TF-IDF knowledge retrieval"

@router.post("/chat", response_model=InvestigationResponse)
def investigation_chat(payload: InvestigationRequest):
    try:
        answer = investigate(payload.message)
        return InvestigationResponse(answer=answer)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Investigation agent failed: {type(exc).__name__}: {exc}") from exc

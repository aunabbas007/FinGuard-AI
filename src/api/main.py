from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.api.config import get_settings
from src.api.db import Base, engine
from src.api import models  # noqa: F401
from src.api.routes.health import router as health_router
from src.api.routes.demo import router as demo_router
from src.api.routes.model import router as model_router
from src.api.routes.transactions import router as transactions_router

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(title=settings.app_name, description="Synthetic-data fraud detection and investigation demo API.", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(health_router)
app.include_router(model_router)
app.include_router(demo_router)
app.include_router(transactions_router)

@app.get("/api/v1")
def api_root():
    return {"name": settings.app_name, "docs": "/docs", "health": "/health"}

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

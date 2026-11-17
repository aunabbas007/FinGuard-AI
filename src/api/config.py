from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    app_name: str = "FinGuard AI"
    environment: str = "development"
    database_url: str = "sqlite:///./finguard.db"
    model_path: Path = ROOT_DIR / "models" / "fraud_detection_model.joblib"
    threshold_path: Path = ROOT_DIR / "models" / "thresholds.json"
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    @property
    def allowed_origins(self):
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]

@lru_cache
def get_settings():
    return Settings()

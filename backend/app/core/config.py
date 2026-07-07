from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "Dengue Outbreak Predictor API"
    VERSION: str = "1.0.0"
    MODEL_DIR: str = "../models"
    MODEL_FILENAME: str = "model.joblib"
    SCALER_FILENAME: str = "scaler.joblib"
    FEATURE_NAMES_FILE: str = "feature_names.json"
    METADATA_FILE: str = "metadata.json"
    DEFAULT_THRESHOLD: float = 0.5
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
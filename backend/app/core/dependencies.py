from fastapi import Depends
from app.services.predictor import PredictorService
from app.core.config import settings

def get_predictor_service() -> PredictorService:
    return PredictorService(
        model_dir=settings.MODEL_DIR,
        model_filename=settings.MODEL_FILENAME,
        scaler_filename=settings.SCALER_FILENAME,
        feature_names_file=settings.FEATURE_NAMES_FILE,
        metadata_file=settings.METADATA_FILE,
        default_threshold=settings.DEFAULT_THRESHOLD
    )
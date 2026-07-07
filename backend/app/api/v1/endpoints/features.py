from fastapi import APIRouter, Depends
from app.core.dependencies import get_predictor_service
from app.services.predictor import PredictorService

router = APIRouter()

@router.get("/feature_names")
async def get_feature_names(
    predictor: PredictorService = Depends(get_predictor_service)
):
    return {"feature_names": predictor.feature_names}
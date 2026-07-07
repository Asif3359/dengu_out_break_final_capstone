from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import PredictionInput, PredictionOutput, BatchPredictionInput, BatchPredictionOutput
from app.core.dependencies import get_predictor_service
from app.services.predictor import PredictorService

router = APIRouter()

@router.post("/predict", response_model=PredictionOutput)
async def predict(
    input_data: PredictionInput,
    predictor: PredictorService = Depends(get_predictor_service)
):
    proba, pred = predictor.predict_proba(input_data.features)
    return PredictionOutput(
        outbreak_probability=float(proba),
        prediction=pred,
        threshold_used=predictor.optimal_threshold,
        model_type=predictor.model_type
    )

@router.post("/predict_batch", response_model=BatchPredictionOutput)
async def predict_batch(
    input_data: BatchPredictionInput,
    predictor: PredictorService = Depends(get_predictor_service)
):
    results = predictor.predict_batch(input_data.items)
    return BatchPredictionOutput(results=results)
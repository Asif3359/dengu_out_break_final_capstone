from pydantic import BaseModel
from typing import Dict, Any, List

class PredictionInput(BaseModel):
    features: Dict[str, float]

class PredictionOutput(BaseModel):
    outbreak_probability: float
    prediction: int
    threshold_used: float
    model_type: str

class BatchPredictionInput(BaseModel):
    items: List[Dict[str, float]]

class BatchPredictionOutput(BaseModel):
    results: List[Dict[str, Any]]
from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class PredictionResult(BaseModel):
    rank: int
    class_id: str
    class_name: str
    confidence: float
    confidence_percentage: str

class PredictionResponse(BaseModel):
    status: str
    predictions: List[PredictionResult]
    top_prediction: Optional[PredictionResult] = None

class ErrorResponse(BaseModel):
    status: str
    message: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str = "ResNet50"
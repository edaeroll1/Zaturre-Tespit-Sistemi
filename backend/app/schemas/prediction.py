from pydantic import BaseModel
from typing import Literal
from ..core.config import MODEL_PTH


class PredictionResponse(BaseModel):
    label:          str
    confidence:     float
    normal_prob:    float
    pneumonia_prob: float
    model_name:     str


class ErrorResponse(BaseModel):
    detail: str

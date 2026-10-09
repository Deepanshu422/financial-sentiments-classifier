from typing import Dict, List
from pydantic import BaseModel, Field

class SentimentProbabilities(BaseModel):
    negative: float = Field(..., ge=0.0, le=1.0)
    positive: float = Field(..., ge=0.0, le=1.0)
    neutral: float = Field(..., ge=0.0, le=1.0)

class SinglePredictionRequest(BaseModel):
    sentence: str = Field(
        min_length=3,
        max_length=2000,
        examples=["Operating profit rose by 14% to EUR 5.1M in the second quarter."]
    )

class SinglePredictionResponse(BaseModel):
    sentence: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]

class DocumentPredictionRequest(BaseModel):
    text: str = Field(
        min_length=5,
        examples=["Sales grew 5% in Europe. Operating margin fell due to inflation. Outlook remains steady."],
    )

class DocumentPredictionResponse(BaseModel):
    overall_sentiment: str
    sentence_count: int
    overall_confidence: float
    document_probabilities: Dict[str, float]
    sentence_breakdown: List[SinglePredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

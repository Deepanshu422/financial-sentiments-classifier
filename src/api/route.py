from fastapi import APIRouter, HTTPException, status
from src.api.schemas import (
    DocumentPredictionRequest,
    DocumentPredictionResponse,
    HealthResponse,
    SinglePredictionRequest,
    SinglePredictionResponse,
)
from src.api.service import sentiment_service

router = APIRouter()

@router.get(
    "/health", 
    response_model=HealthResponse, 
    status_code=status.HTTP_200_OK, 
    tags=["System"]
)
async def check_health():
    return HealthResponse(
        status="healthy" if sentiment_service.is_ready else "degraded",
        model_loaded=sentiment_service.is_ready,
    )

@router.post(
    "/predict",
    response_model=SinglePredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
async def predict_sentence(payload: SinglePredictionRequest):
    if not sentiment_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model weights unavailable or still warming up.",
        )
    return sentiment_service.classify_sentence(payload.sentence)

@router.post(
    "/predict-document",
    response_model=DocumentPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_document(payload: DocumentPredictionRequest):
    if not sentiment_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model weights unavailable or still warming up.",
        )
    return sentiment_service.classify_document(payload.text)
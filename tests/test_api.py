import pytest
from pathlib import Path
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.service import sentiment_service


MODEL_PATH = Path("artifacts/final_model/onnx/model_quantized.onnx")


@pytest.fixture(scope="module", autouse=True)
def init_service():
    """Ensure the model weights are loaded locally, or fall back to mock in CI."""
    if MODEL_PATH.exists():
        sentiment_service.load_model()
        yield
    else:
        mock_engine = MagicMock()
        mock_engine.predict_sentence.return_value = {
            "sentence": "Operating profit grew by 15% year on year.",
            "sentiment": "positive",
            "confidence": 0.9842,
            "probabilities": {"negative": 0.01, "neutral": 0.01, "positive": 0.98},
        }
        mock_engine.predict_long_text.return_value = {
            "overall_sentiment": "positive",
            "overall_confidence": 0.9521,
            "document_probabilities": {"negative": 0.02, "neutral": 0.03, "positive": 0.95},
            "sentence_count": 3,
            "sentence_breakdown": [
                {
                    "sentence": "Operating profit grew by 15% year on year.",
                    "sentiment": "positive",
                    "confidence": 0.9842,
                    "probabilities": {"negative": 0.01, "neutral": 0.01, "positive": 0.98},
                },
                {
                    "sentence": "However, operating expenses increased in Europe.",
                    "sentiment": "negative",
                    "confidence": 0.9123,
                    "probabilities": {"negative": 0.91, "neutral": 0.06, "positive": 0.03},
                },
                {
                    "sentence": "Management remains confident about reaching year-end targets.",
                    "sentiment": "positive",
                    "confidence": 0.9600,
                    "probabilities": {"negative": 0.02, "neutral": 0.02, "positive": 0.96},
                },
            ],
        }
        sentiment_service._predictor = mock_engine
        yield
        sentiment_service._predictor = None


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_validation_error(client):
    # String shorter than min_length (3) should fail Pydantic validation
    response = client.post("/predict", json={"sentence": "hi"})
    assert response.status_code == 422


def test_document_validation_error(client):
    # Empty string should fail Pydantic validation (min_length=5)
    response = client.post("/predict-document", json={"text": ""})
    assert response.status_code == 422


def test_predict_single_success(client):
    payload = {"sentence": "Operating profit grew by 15% year on year."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["negative", "neutral", "positive"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert len(data["probabilities"]) == 3


def test_predict_document_success(client):
    payload = {
        "text": (
            "Operating profit grew by 15% year on year. "
            "However, operating expenses increased in Europe. "
            "Management remains confident about reaching year-end targets."
        )
    }
    response = client.post("/predict-document", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "overall_sentiment" in data
    assert "overall_confidence" in data
    assert "document_probabilities" in data
    assert data["sentence_count"] == 3
    assert len(data["sentence_breakdown"]) == 3
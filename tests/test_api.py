import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.service import sentiment_service


@pytest.fixture(scope="module", autouse=True)
def init_service():
    """Ensure the model weights are loaded before running API tests."""
    sentiment_service.load_model()


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
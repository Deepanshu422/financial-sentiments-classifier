from typing import Optional
from src.predict import FinancialSentimentPredictor, get_predictor

class SentimentService:
    """
    Service layer handling model operations and inference delegation.
    """

    def __init__(self):
        self._predictor: Optional[FinancialSentimentPredictor] = None

    def load_model(self) -> None:
        """Loads weights into memory."""
        self._predictor = get_predictor()

    @property
    def is_ready(self) -> bool:
        return self._predictor is not None

    def classify_sentence(self, sentence: str) -> dict:
        if not self.is_ready:
            raise RuntimeError("Model service not initialized.")
        return self._predictor.predict_sentence(sentence)

    def classify_document(self, text: str) -> dict:
        if not self.is_ready:
            raise RuntimeError("Model service not initialized.")
        return self._predictor.predict_long_text(text)


# Single service instance for the microservice container
sentiment_service = SentimentService()
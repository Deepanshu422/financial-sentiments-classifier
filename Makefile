.PHONY: help install install-dev format lint test train serve ui clean \
        clean-artifacts docker-build-train docker-train docker-up docker-down docker-logs

PYTHON := python3
PIP := $(PYTHON) -m pip

help:
	@echo "Available commands:"
	@echo "  make install             Install production dependencies"
	@echo "  make install-dev         Install development and test dependencies"
	@echo "  make format              Format code with Black and isort"
	@echo "  make lint                Run Flake8 linting checks"
	@echo "  make test                Run pytest suite"
	@echo "  make train               Run model training locally"
	@echo "  make serve               Start FastAPI inference server locally"
	@echo "  make ui                  Start Streamlit UI locally"
	@echo "  make clean               Remove bytecode and pytest caches"
	@echo "  make clean-artifacts     Wipe all model checkpoints, weights, and eval metrics"
	@echo "  make docker-build-train  Build Docker image for model training"
	@echo "  make docker-train        Train model inside Docker and output weights to ./artifacts"
	@echo "  make docker-up           Build and start API & UI microservices with docker compose"
	@echo "  make docker-down         Stop and tear down docker compose containers"
	@echo "  make docker-logs         Stream logs from running Docker containers"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

format:
	black src/ tests/
	isort src/ tests/

lint:
	flake8 src/ tests/ --max-line-length=100

test:
	pytest tests/ -v --cov=src

train:
	$(PYTHON) -m src.train

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run src/app_ui.py --server.port 8501

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	rm -rf .coverage htmlcov

clean-artifacts: clean
	@echo "[INFO] Removing generated weights and evaluation artifacts..."
	rm -rf artifacts/checkpoints/*
	rm -rf artifacts/final_model/*
	rm -rf artifacts/logs/*
	rm -f artifacts/eval_results.json
	@echo "[SUCCESS] Artifacts wiped."

# --- Docker Operations ---

docker-build-train:
	@echo "[INFO] Building training Docker image..."
	docker build -t sentiment-trainer:latest -f docker/Dockerfile.train .

docker-train: docker-build-train
	@echo "[INFO] Running training pipeline in an isolated container..."
	docker run --rm \
		-v $(shell pwd)/artifacts:/app/artifacts \
		sentiment-trainer:latest

docker-up:
	@echo "[INFO] Starting containerized services (API + UI)..."
	docker compose up --build -d

docker-down:
	@echo "[INFO] Stopping containerized services..."
	docker compose down

docker-logs:
	docker compose logs -f
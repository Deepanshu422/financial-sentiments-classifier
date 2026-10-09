# 📈 Financial Sentiment Transfer Learning & ONNX Inference

An end-to-end, production-ready NLP system that fine-tunes a `distilbert-base-uncased` transformer on financial commentary (Financial PhraseBank) and deploys it via an **INT8-quantized ONNX engine** for high-throughput, low-latency CPU serving (<200 MB RAM).

Features a decoupled **FastAPI** inference backend, an interactive **Streamlit** dashboard, full test coverage, and a containerized microservice setup.

---

## ⚡ Key Highlights

- **Transfer Learning Backbone:** Fine-tuned `distilbert-base-uncased` across Negative, Neutral, and Positive financial sentiment classes.
- **Ultra-Low Memory Footprint:** Exported and dynamically quantized to INT8 with **ONNX Runtime**, shrinking model size from **~268 MB to ~65 MB** and runtime RAM to **<200 MB** (fits free-tier deployments like Render, Koyeb, and Hugging Face Spaces).
- **Decoupled Architecture:** 
  - **FastAPI Backend:** RESTful endpoints for single headlines and multi-sentence earnings releases.
  - **Streamlit Frontend:** Real-time sentiment metrics, confidence breakdown, and document parsing.
- **Production MLOps Standards:** Clean separation between training and inference environments, automated `pytest` suite, and containerized Docker orchestration.

---

## 📁 Repository Structure

```text
├── artifacts/
│   └── final_model/
│       ├── onnx/
│       │   └── model_quantized.onnx  # INT8-quantized ONNX model (~65MB)
│       ├── model.safetensors         # PyTorch fine-tuned weights
│       ├── config.json
│       └── tokenizer.json
├── docker/
│   ├── Dockerfile.serve              # Production serving container (FastAPI)
│   └── Dockerfile.train              # Dedicated training container
├── src/
│   ├── api/
│   │   ├── main.py                   # FastAPI initialization & lifespans
│   │   ├── routes.py                 # REST route definitions
│   │   ├── schemas.py                # Pydantic request/response models
│   │   └── service.py                # Model loader & service wrapper
│   ├── app_ui.py                     # Streamlit frontend dashboard
│   ├── config.py                     # Global paths & hyperparameter configs
│   ├── dataset_loader.py             # Financial PhraseBank split & tokenization
│   ├── export_onnx.py                # PyTorch -> INT8 ONNX conversion pipeline
│   ├── predict.py                    # ONNX Runtime inference engine
│   └── train.py                      # DistilBERT fine-tuning pipeline
├── tests/
│   ├── test_api.py                   # Endpoint integration tests
│   └── test_dataset.py               # Data processing & tokenization tests
├── docker-compose.yml                # Microservices composition (API + UI)
├── Makefile                          # Task automation CLI
├── requirements.txt                  # Lightweight inference dependencies
└── requirements-dev.txt              # Full training, dev, and testing stack
🚀 QuickstartPrerequisitesPython 3.10+Docker & Docker Compose (optional, for containerization)1. Local Environment SetupBashgit clone [https://github.com/](https://github.com/)<your-username>/financial-sentiment-transfer-learning.git
cd financial-sentiment-transfer-learning

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dev & training dependencies
make install-dev
2. Model Pipeline (Train & Quantize)Bash# 1. Fine-tune DistilBERT on host machine
make train

# 2. Export & Quantize to INT8 ONNX format
python src/export_onnx.py
3. Run Verification TestsBashmake test
4. Run Locally on HostBash# Terminal 1: Start FastAPI backend (Port 8000)
make serve

# Terminal 2: Start Streamlit UI (Port 8501)
make ui
🐳 Docker Container DeploymentDeploy the entire decoupled stack locally or on a cloud virtual server using Docker Compose:Bash# Build and run containers in detached mode
make docker-up

# View real-time service logs
make docker-logs

# Stop services
make docker-down
Interactive API Swagger Docs: http://localhost:8000/docsStreamlit Web Application: http://localhost:8501📡 API ReferenceHealth CheckHTTPGET /health
Response:JSON{
  "status": "healthy",
  "model_loaded": true
}
Classify HeadlineHTTPPOST /predict
Content-Type: application/json

{
  "sentence": "Operating profit rose 14% to EUR 5.1M in the second quarter."
}
Response:JSON{
  "sentence": "Operating profit rose 14% to EUR 5.1M in the second quarter.",
  "sentiment": "positive",
  "confidence": 0.9842,
  "probabilities": {
    "negative": 0.0051,
    "neutral": 0.0107,
    "positive": 0.9842
  }
}
Analyze Document / Financial ReportHTTPPOST /predict-document
Content-Type: application/json

{
  "text": "Operating profit rose by 14% to EUR 5.1M. However, overseas supply chain issues created headwinds in Q2."
}
Response:JSON{
  "overall_sentiment": "neutral",
  "overall_confidence": 0.5421,
  "document_probabilities": {
    "negative": 0.3812,
    "neutral": 0.5421,
    "positive": 0.0767
  },
  "sentence_count": 2,
  "sentence_breakdown": [
    {
      "sentence": "Operating profit rose by 14% to EUR 5.1M.",
      "sentiment": "positive",
      "confidence": 0.9842,
      "probabilities": { ... }
    },
    {
      "sentence": "However, overseas supply chain issues created headwinds in Q2.",
      "sentiment": "negative",
      "confidence": 0.9410,
      "probabilities": { ... }
    }
  ]
}
🌐 Cloud Deployment NotesBackend (Render / Koyeb / AWS EC2):Set the start command to:Bashuvicorn src.api.main:app --host 0.0.0.0 --port $PORT
Frontend (Streamlit Community Cloud):Connect your GitHub repository and point to src/app_ui.py. In App Settings $\to$ Secrets / Environment Variables, add:Ini, TOMLAPI_URL = "[https://your-api-service.onrender.com](https://your-api-service.onrender.com)"


## 🔒 License & Copyright

Copyright (c) 2026 Deepanshu Singh. All rights reserved.

This project is proprietary and intended for personal/internal use only. No part of this repository may be reproduced, distributed, modified, or used for commercial purposes without prior written permission.
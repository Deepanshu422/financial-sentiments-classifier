from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.route import router as api_router
from src.api.service import sentiment_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Application startup: Load model into memory
    print("[INFO] Microservice initializing: Loading model weights...")
    try:
        sentiment_service.load_model()
        print("[INFO] Model successfully loaded.")
    except Exception as exc:
        print(f"[ERROR] Model load failed: {exc}")
    
    yield
    
    # Application shutdown
    print("[INFO] Microservice shutting down...")


def create_app() -> FastAPI:
    """Factory function for FastAPI application instance."""
    app = FastAPI(
        title="Financial Sentiment Microservice",
        description="Production microservice for financial sentiment transfer learning inference.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Middleware setup
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes under /api/v1 (or root)
    app.include_router(api_router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
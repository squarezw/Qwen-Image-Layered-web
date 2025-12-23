"""
FastAPI Main Application
"""
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os

from .config import settings
from .models.model_manager import ModelManager
from .models.inference_queue import InferenceQueue
from .models.schemas import HealthCheckResponse
from .db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown lifecycle management
    """
    # Startup: Load models and initialize services
    print("="*60)
    print("Starting Qwen Image Layered API...")
    print("="*60)

    # Initialize database
    print("Initializing database...")
    try:
        await init_db()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization error: {e}")

    # Initialize model manager
    print("Loading AI models...")
    model_manager = ModelManager()
    try:
        await model_manager.initialize()
        print("Models loaded successfully")
    except Exception as e:
        print(f"Model loading error: {e}")
        print("API will start but inference endpoints will not work")

    # Initialize inference queue
    print("Initializing inference queue...")
    app.state.inference_queue = InferenceQueue()
    print("Inference queue initialized")

    # Ensure storage directory exists
    os.makedirs(settings.file_storage_path, exist_ok=True)
    print(f"Storage directory: {settings.file_storage_path}")

    print("="*60)
    print("Qwen Image Layered API is ready!")
    print(f"Environment: {settings.environment}")
    print(f"GPU Available: {model_manager.get_gpu_info()['available']}")
    print("="*60)

    yield

    # Shutdown: Cleanup
    print("Shutting down Qwen Image Layered API...")


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="API for Qwen Image Layered - Decompose images into editable RGBA layers",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for serving generated files)
if os.path.exists(settings.file_storage_path):
    app.mount("/files", StaticFiles(directory=settings.file_storage_path), name="files")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Qwen Image Layered API",
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# Health check endpoint
@app.get("/api/v1/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint
    Returns API status and GPU information
    """
    model_manager = ModelManager()
    gpu_info = model_manager.get_gpu_info()

    # Get queue size
    queue_size = 0
    if hasattr(app.state, 'inference_queue'):
        queue_size = app.state.inference_queue.get_queue_size()

    return HealthCheckResponse(
        status="healthy" if model_manager.is_ready else "initializing",
        gpu_available=gpu_info["available"],
        gpu_info=gpu_info if gpu_info["available"] else None,
        models_loaded=model_manager.is_ready,
        queue_size=queue_size,
        timestamp=datetime.utcnow()
    )


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An error occurred"
        }
    )


# TODO: Include routers when implemented
# from .api import decomposition, editing, batch, history, websocket
# app.include_router(decomposition.router, prefix="/api/v1/decomposition", tags=["decomposition"])
# app.include_router(editing.router, prefix="/api/v1/editing", tags=["editing"])
# app.include_router(batch.router, prefix="/api/v1/batch", tags=["batch"])
# app.include_router(history.router, prefix="/api/v1/history", tags=["history"])
# app.include_router(websocket.router, prefix="/ws", tags=["websocket"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )

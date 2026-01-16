"""Main FastAPI application entry point."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from backend.app.api import annotations, imports, inference, labeling, projects, training, videos
from backend.app.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Local Video Auto-Label → Human Correct → Fine-Tune Real-Time Detector",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(projects.router, prefix="/api")
app.include_router(videos.router, prefix="/api")
app.include_router(annotations.router, prefix="/api")
app.include_router(labeling.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(inference.router, prefix="/api")
app.include_router(imports.router, prefix="/api")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/api/config")
async def get_config():
    """Get application configuration."""
    return {
        "device": settings.device,
        "default_sample_interval": settings.default_sample_interval_seconds,
        "default_tracking_mode": settings.default_tracking_mode,
        "available_models": ["yolo11n", "yolo11s", "yolo11m", "rfdetr-b", "rfdetr-l"],
    }


@app.on_event("startup")
async def startup():
    """Application startup."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Data directory: {settings.data_dir}")
    logger.info(f"Device: {settings.device}")


@app.on_event("shutdown")
async def shutdown():
    """Application shutdown."""
    from backend.app.database import db_manager
    await db_manager.close_all()
    logger.info("Application shutdown complete")


def run_server():
    """Run the development server."""
    uvicorn.run(
        "backend.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run_server()


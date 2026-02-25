"""GPU cluster connection API routes."""

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from backend.app.config import settings
from backend.app.services.gpu_service import gpu_service

router = APIRouter(prefix="/gpu", tags=["gpu"])


class ConnectRequest(BaseModel):
    password: str


@router.post("/connect")
async def connect(request: ConnectRequest):
    """Establish SSH connection to the GPU cluster."""
    try:
        result = gpu_service.connect(request.password)
        return result
    except Exception as e:
        logger.warning(f"GPU connect failed: {e}")
        raise HTTPException(status_code=400, detail=f"Connection failed: {e}")


@router.post("/disconnect")
async def disconnect():
    """Disconnect from the GPU cluster."""
    gpu_service.disconnect()
    return {"status": "disconnected"}


@router.get("/status")
async def status():
    """Check GPU cluster connection status."""
    connected = gpu_service.is_connected
    return {
        "connected": connected,
        "host": settings.ssh_host,
        "user": settings.ssh_user,
    }

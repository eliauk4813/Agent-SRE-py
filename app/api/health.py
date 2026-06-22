"""Health check API."""

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.elasticsearch_client import elasticsearch_manager

router = APIRouter()


@router.get("/health")
async def health_check():
    """Check service and retrieval backend health."""
    health_data: dict[str, Any] = {
        "service": config.app_name,
        "version": config.app_version,
        "status": "healthy",
    }

    try:
        milvus_healthy = milvus_manager.health_check()
        health_data["milvus"] = {
            "status": "connected" if milvus_healthy else "disconnected",
            "message": "Milvus connection is healthy" if milvus_healthy else "Milvus is unavailable",
        }
    except Exception as exc:
        logger.warning(f"Milvus health check failed: {exc}")
        health_data["milvus"] = {
            "status": "error",
            "message": f"Milvus check failed: {exc}",
        }

    try:
        es_healthy = elasticsearch_manager.health_check()
        health_data["elasticsearch"] = {
            "status": "connected" if es_healthy else "disconnected",
            "message": "Elasticsearch connection is healthy"
            if es_healthy
            else "Elasticsearch is unavailable",
        }
    except Exception as exc:
        logger.warning(f"Elasticsearch health check failed: {exc}")
        health_data["elasticsearch"] = {
            "status": "error",
            "message": f"Elasticsearch check failed: {exc}",
        }

    overall_status = "healthy"
    status_code = 200

    if health_data["milvus"]["status"] != "connected":
        overall_status = "unhealthy"
        status_code = 503
        health_data["error"] = "Milvus is unavailable"
    elif health_data["elasticsearch"]["status"] != "connected":
        overall_status = "unhealthy"
        status_code = 503
        health_data["error"] = "Elasticsearch is unavailable"

    health_data["status"] = overall_status

    return JSONResponse(
        status_code=status_code,
        content={
            "code": status_code,
            "message": "service is healthy" if overall_status == "healthy" else "service is unavailable",
            "data": health_data,
        },
    )

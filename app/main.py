"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api import aiops, chat, file, health
from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.elasticsearch_client import elasticsearch_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    logger.info("=" * 60)
    logger.info(f"{config.app_name} v{config.app_version} starting")
    logger.info(f"Environment: {'debug' if config.debug else 'production'}")
    logger.info(f"Listen: http://{config.host}:{config.port}")
    logger.info(f"Docs: http://{config.host}:{config.port}/docs")

    logger.info("Connecting Milvus...")
    milvus_manager.connect()
    logger.info("Milvus connected")

    logger.info("Connecting Elasticsearch...")
    elasticsearch_manager.connect()
    logger.info("Elasticsearch connected")
    logger.info("=" * 60)

    yield

    logger.info("Closing Milvus and Elasticsearch connections...")
    milvus_manager.close()
    elasticsearch_manager.close()
    logger.info(f"{config.app_name} stopped")


app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description="LangChain based intelligent Q&A and AIOps system",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(file.router, prefix="/api", tags=["file"])
app.include_router(aiops.router, prefix="/api", tags=["aiops"])

static_dir = "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """Return the frontend entrypoint when it exists."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": f"Welcome to {config.app_name} API",
        "version": config.app_version,
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info",
    )

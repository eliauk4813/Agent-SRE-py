"""File upload API."""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from app.services.vector_index_service import vector_index_service

router = APIRouter()

UPLOAD_DIR = Path("./uploads")
ALLOWED_EXTENSIONS = ["md"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a Markdown file and index it automatically."""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename cannot be empty")

        safe_filename = _sanitize_filename(file.filename)
        file_extension = _get_file_extension(safe_filename)
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        file_path = UPLOAD_DIR / safe_filename

        if file_path.exists():
            logger.info(f"Replacing existing file: {file_path}")
            file_path.unlink()

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds limit ({MAX_FILE_SIZE} bytes)",
            )

        file_path.write_bytes(content)
        logger.info(f"File uploaded successfully: {file_path}")

        try:
            logger.info(f"Indexing uploaded Markdown file: {file_path}")
            vector_index_service.index_single_file(str(file_path))
            logger.info(f"Vector index created successfully: {file_path}")
        except Exception as exc:
            logger.error(f"Vector index creation failed: {file_path}, error: {exc}")

        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "message": "success",
                "data": {
                    "filename": safe_filename,
                    "file_path": str(file_path),
                    "size": len(content),
                },
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"File upload failed: {exc}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {exc}") from exc


@router.post("/index_directory")
async def index_directory(directory_path: str = None):
    """Index all Markdown files in a directory."""
    try:
        logger.info(f"Indexing directory: {directory_path or 'uploads'}")
        result = vector_index_service.index_directory(directory_path)
        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "message": "success" if result.success else "partial_success",
                "data": result.to_dict(),
            },
        )
    except Exception as exc:
        logger.error(f"Index directory failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Index directory failed: {exc}") from exc


def _get_file_extension(filename: str) -> str:
    """Return the lowercase file extension without the leading dot."""
    parts = filename.rsplit(".", 1)
    if len(parts) == 2:
        return parts[1].lower()
    return ""


def _sanitize_filename(filename: str) -> str:
    """Normalize file names for local storage."""
    sanitized = filename.replace(" ", "_")
    for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        sanitized = sanitized.replace(char, "_")
    return sanitized

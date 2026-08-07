"""Hybrid document indexing service."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from loguru import logger

from app.services.document_splitter_service import document_splitter_service
from app.services.document_quality_checker import document_quality_checker
from app.services.keyword_index_service import keyword_index_service
from app.services.vector_store_manager import vector_store_manager


class IndexingResult:
    """Indexing result data."""

    def __init__(self) -> None:
        self.success = False
        self.directory_path = ""
        self.total_files = 0
        self.success_count = 0
        self.fail_count = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message = ""
        self.failed_files: Dict[str, str] = {}

    def increment_success_count(self) -> None:
        self.success_count += 1

    def increment_fail_count(self) -> None:
        self.fail_count += 1

    def add_failed_file(self, file_path: str, error: str) -> None:
        self.failed_files[file_path] = error

    def get_duration_ms(self) -> int:
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "directory_path": self.directory_path,
            "total_files": self.total_files,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "duration_ms": self.get_duration_ms(),
            "error_message": self.error_message,
            "failed_files": self.failed_files,
        }


class VectorIndexService:
    """Read Markdown files, split them, and index chunks into Milvus and ES."""

    def __init__(self) -> None:
        self.upload_path = "./uploads"
        logger.info("Hybrid index service initialized")

    def index_directory(self, directory_path: Optional[str] = None) -> IndexingResult:
        """Index all Markdown files within a directory."""
        result = IndexingResult()
        result.start_time = datetime.now()

        try:
            target_path = directory_path if directory_path else self.upload_path
            dir_path = Path(target_path).resolve()

            if not dir_path.exists() or not dir_path.is_dir():
                raise ValueError(f"Directory does not exist or is invalid: {target_path}")

            result.directory_path = str(dir_path)
            files = list(dir_path.glob("*.md"))

            if not files:
                logger.warning(f"No Markdown files found in directory: {target_path}")
                result.total_files = 0
                result.success = True
                result.end_time = datetime.now()
                return result

            result.total_files = len(files)
            logger.info(f"Indexing directory {target_path}, found {len(files)} Markdown files")

            for file_path in files:
                try:
                    self.index_single_file(str(file_path))
                    result.increment_success_count()
                    logger.info(f"Indexed file successfully: {file_path.name}")
                except Exception as exc:
                    result.increment_fail_count()
                    result.add_failed_file(str(file_path), str(exc))
                    logger.error(f"Failed to index file: {file_path.name}, error: {exc}")

            result.success = result.fail_count == 0
            result.end_time = datetime.now()
            logger.info(
                "Directory indexing complete: "
                f"total={result.total_files}, success={result.success_count}, fail={result.fail_count}"
            )
            return result
        except Exception as exc:
            logger.error(f"Failed to index directory: {exc}")
            result.success = False
            result.error_message = str(exc)
            result.end_time = datetime.now()
            return result

    def index_single_file(self, file_path: str) -> None:
        """Index a single Markdown file into both Milvus and Elasticsearch.

        改进后的流程：
        1. 读取文件内容
        2. 文档切分
        3. 🆕 质量检查与清洗（新增）
        4. 生成chunk_id
        5. 索引到 Milvus + Elasticsearch
        """
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            raise ValueError(f"File does not exist: {file_path}")
        if path.suffix.lower() != ".md":
            raise ValueError(f"Only Markdown files are supported: {file_path}")

        logger.info(f"Start hybrid indexing file: {path}")

        try:
            content = path.read_text(encoding="utf-8")
            logger.info(f"Read file: {path}, content length: {len(content)} chars")

            normalized_path = path.as_posix()
            vector_store_manager.delete_by_source(normalized_path)
            keyword_index_service.delete_by_source(normalized_path)

            # 步骤1: 文档切分
            documents = document_splitter_service.split_document(content, normalized_path)
            logger.info(f"Split complete: {file_path} -> {len(documents)} chunks")

            if not documents:
                logger.warning(f"Markdown content is empty or produced no chunks: {file_path}")
                return

            # 步骤2: 🆕 质量检查与清洗
            cleaned_documents, quality_report = document_quality_checker.check_and_clean(
                documents=documents,
                source_path=normalized_path
            )

            # 记录质量报告
            logger.info(
                f"Quality check result: {file_path}, "
                f"input={quality_report.total_input}, "
                f"valid={quality_report.total_valid}, "
                f"filtered={quality_report.total_filtered}, "
                f"score={quality_report.quality_score:.1f}/100"
            )

            # 如果有严重问题，记录详情
            if quality_report.critical_count > 0:
                logger.warning(
                    f"Found {quality_report.critical_count} critical issues in {file_path}"
                )

            # 如果所有chunk都被过滤，记录警告
            if not cleaned_documents:
                logger.warning(
                    f"All chunks filtered due to quality issues: {file_path}, "
                    f"report={quality_report.to_dict()}"
                )
                return

            # 步骤3: 生成chunk_id（只为有效chunk生成）
            self._assign_chunk_ids(cleaned_documents, normalized_path)

            # 步骤4: 索引（只索引通过质量检查的chunk）
            chunk_ids = [str(doc.metadata["chunk_id"]) for doc in cleaned_documents]
            vector_store_manager.add_documents(cleaned_documents, ids=chunk_ids)
            keyword_index_service.add_documents(cleaned_documents)

            logger.info(
                f"Hybrid index complete: {file_path}, "
                f"indexed={len(cleaned_documents)}/{len(documents)} chunks, "
                f"quality_score={quality_report.quality_score:.1f}, "
                f"targets=Milvus+Elasticsearch"
            )

        except Exception as exc:
            logger.error(f"Failed to index file: {file_path}, error: {exc}")
            raise RuntimeError(f"Failed to index file: {exc}") from exc

    def _assign_chunk_ids(self, documents: List[Document], source_path: str) -> None:
        """Assign stable ids shared by Milvus and ES for RRF de-duplication."""
        for doc in documents:
            chunk_index = doc.metadata.get("chunk_index", 0)
            digest_source = f"{source_path}:{chunk_index}:{doc.page_content}"
            doc.metadata["chunk_id"] = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()


vector_index_service = VectorIndexService()

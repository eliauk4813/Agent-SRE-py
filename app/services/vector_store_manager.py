"""Milvus vector store manager."""

from __future__ import annotations

import time
from typing import List, Optional

from langchain_core.documents import Document
from langchain_milvus import Milvus
from loguru import logger

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service


COLLECTION_NAME = "biz"


class VectorStoreManager:
    """Wrap LangChain Milvus operations used by indexing and retrieval."""

    def __init__(self) -> None:
        self.vector_store: Milvus | None = None
        self.collection_name = COLLECTION_NAME
        self._initialize_vector_store()

    def _initialize_vector_store(self) -> None:
        try:
            _ = milvus_manager.connect()
            connection_args = {
                "host": config.milvus_host,
                "port": config.milvus_port,
            }
            self.vector_store = Milvus(
                embedding_function=vector_embedding_service,
                collection_name=self.collection_name,
                connection_args=connection_args,
                auto_id=False,
                drop_old=False,
                text_field="content",
                vector_field="vector",
                primary_field="id",
                metadata_field="metadata",
            )
            logger.info(
                f"VectorStore initialized: {config.milvus_host}:{config.milvus_port}, "
                f"collection={self.collection_name}"
            )
        except Exception as exc:
            logger.error(f"VectorStore initialization failed: {exc}")
            raise

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """Add documents to Milvus with stable ids shared by ES."""
        if not documents:
            return []
        if self.vector_store is None:
            raise RuntimeError("VectorStore is not initialized")

        start_time = time.time()
        if ids is None:
            ids = [str(doc.metadata.get("chunk_id", "")) for doc in documents]
        if len(ids) != len(documents) or any(not doc_id for doc_id in ids):
            raise ValueError("Document ids must be provided for every Milvus document")

        try:
            result_ids = self.vector_store.add_documents(documents, ids=ids)
            elapsed = time.time() - start_time
            logger.info(
                f"Added {len(documents)} documents to Milvus, "
                f"elapsed={elapsed:.2f}s, avg={elapsed / len(documents):.2f}s"
            )
            return result_ids
        except Exception as exc:
            logger.error(f"Failed to add documents to Milvus: {exc}")
            raise

    def delete_by_source(self, file_path: str) -> int:
        """Delete all Milvus chunks for one source file."""
        try:
            collection = milvus_manager.get_collection()
            expr = f'metadata["_source"] == "{file_path}"'
            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0
            logger.info(f"Deleted old Milvus chunks for {file_path}, deleted={deleted_count}")
            return int(deleted_count)
        except Exception as exc:
            logger.warning(f"Failed to delete old Milvus chunks for {file_path}: {exc}")
            return 0

    def get_vector_store(self) -> Milvus:
        if self.vector_store is None:
            raise RuntimeError("VectorStore is not initialized")
        return self.vector_store

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Run Milvus semantic similarity search."""
        try:
            docs = self.get_vector_store().similarity_search(query, k=k)
            logger.debug(f"Milvus similarity search finished, query={query!r}, count={len(docs)}")
            return docs
        except Exception as exc:
            logger.error(f"Milvus similarity search failed: {exc}")
            return []


vector_store_manager = VectorStoreManager()

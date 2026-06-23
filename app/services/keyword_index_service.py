"""Keyword index service backed by Elasticsearch."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.services.elasticsearch_client import elasticsearch_manager


@dataclass
class KeywordSearchHit:
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]

    def to_document(self) -> Document:
        return Document(page_content=self.content, metadata=self.metadata)


class KeywordIndexService:
    """Store and search chunk text in Elasticsearch."""

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return

        client = elasticsearch_manager.get_client()
        actions = []
        now = datetime.now(timezone.utc).isoformat()

        for doc in documents:
            chunk_id = str(doc.metadata.get("chunk_id", ""))
            if not chunk_id:
                raise ValueError("Document metadata must include chunk_id before ES indexing")

            actions.append({"index": {"_index": config.es_index_name, "_id": chunk_id}})
            actions.append(self._document_body(doc, chunk_id, now))

        response = client.bulk(operations=actions, refresh=True)
        if response.get("errors"):
            raise RuntimeError(f"Elasticsearch bulk index failed: {response}")
        logger.info(f"Indexed {len(documents)} chunks into Elasticsearch")

    def delete_by_source(self, file_path: str) -> int:
        try:
            client = elasticsearch_manager.get_client()
            response = client.delete_by_query(
                index=config.es_index_name,
                query={"term": {"source": file_path}},
                refresh=True,
                conflicts="proceed",
            )
            deleted = int(response.get("deleted", 0))
            logger.info(f"Deleted old Elasticsearch chunks for {file_path}, deleted={deleted}")
            return deleted
        except Exception as exc:
            logger.warning(f"Failed to delete old Elasticsearch chunks for {file_path}: {exc}")
            return 0

    def search(self, query: str, top_k: int) -> List[KeywordSearchHit]:
        try:
            client = elasticsearch_manager.get_client()
            response = client.search(
                index=config.es_index_name,
                size=top_k,
                query={
                    "multi_match": {
                        "query": query,
                        "fields": [
                            "h1^4",
                            "h2^3",
                            "h3^2",
                            "header_path^3",
                            "file_name^1.5",
                            "content",
                        ],
                        "type": "best_fields",
                        "operator": "or",
                    }
                },
            )
            hits = []
            for item in response.get("hits", {}).get("hits", []):
                source = item.get("_source", {})
                hits.append(
                    KeywordSearchHit(
                        id=str(item.get("_id") or source.get("id")),
                        content=str(source.get("content", "")),
                        score=float(item.get("_score") or 0.0),
                        metadata=self._metadata_from_source(source),
                    )
                )
            logger.info(f"Elasticsearch keyword search returned {len(hits)} chunks")
            return hits
        except Exception as exc:
            logger.error(f"Elasticsearch keyword search failed: {exc}")
            return []

    def _document_body(self, doc: Document, chunk_id: str, updated_at: str) -> Dict[str, Any]:
        metadata = doc.metadata
        return {
            "id": chunk_id,
            "content": doc.page_content,
            "header_path": metadata.get("header_path", ""),
            "h1": metadata.get("h1", ""),
            "h2": metadata.get("h2", ""),
            "h3": metadata.get("h3", ""),
            "source": metadata.get("_source", ""),
            "file_name": metadata.get("_file_name", ""),
            "extension": metadata.get("_extension", ""),
            "section_index": metadata.get("section_index", 0),
            "chunk_index": metadata.get("chunk_index", 0),
            "block_types": metadata.get("block_types", []),
            "content_length": len(doc.page_content),
            "updated_at": updated_at,
        }

    def _metadata_from_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        metadata = {
            "chunk_id": source.get("id", ""),
            "_source": source.get("source", ""),
            "_file_name": source.get("file_name", ""),
            "_extension": source.get("extension", ""),
            "section_index": source.get("section_index", 0),
            "chunk_index": source.get("chunk_index", 0),
            "block_types": source.get("block_types", []),
            "header_path": source.get("header_path", ""),
        }
        for key in ("h1", "h2", "h3"):
            if source.get(key):
                metadata[key] = source[key]
        return metadata


keyword_index_service = KeywordIndexService()

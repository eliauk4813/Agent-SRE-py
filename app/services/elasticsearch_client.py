"""Elasticsearch client and index bootstrap."""

from __future__ import annotations

from typing import Any

from loguru import logger

from app.config import config


class ElasticsearchManager:
    """Small wrapper around the Elasticsearch Python client."""

    def __init__(self) -> None:
        self._client: Any | None = None

    def connect(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from elasticsearch import Elasticsearch
        except ImportError as exc:
            raise RuntimeError(
                "Elasticsearch dependency is missing. Run dependency sync/install first."
            ) from exc

        self._client = Elasticsearch(
            config.es_url,
            request_timeout=config.es_request_timeout,
        )
        self.ensure_index()
        logger.info(f"Connected to Elasticsearch: {config.es_url}, index={config.es_index_name}")
        return self._client

    def ensure_index(self) -> None:
        client = self._client
        if client is None:
            return

        if client.indices.exists(index=config.es_index_name):
            return

        body = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "biz_text_analyzer": {
                            "type": config.es_analyzer,
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "content": {"type": "text", "analyzer": "biz_text_analyzer"},
                    "header_path": {
                        "type": "text",
                        "analyzer": "biz_text_analyzer",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "h1": {
                        "type": "text",
                        "analyzer": "biz_text_analyzer",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "h2": {
                        "type": "text",
                        "analyzer": "biz_text_analyzer",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "h3": {
                        "type": "text",
                        "analyzer": "biz_text_analyzer",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "source": {"type": "keyword"},
                    "file_name": {"type": "keyword"},
                    "extension": {"type": "keyword"},
                    "section_index": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
                    "block_types": {"type": "keyword"},
                    "content_length": {"type": "integer"},
                    "updated_at": {"type": "date"},
                }
            },
        }
        client.indices.create(index=config.es_index_name, body=body)
        logger.info(f"Created Elasticsearch index: {config.es_index_name}")

    def get_client(self) -> Any:
        return self.connect()

    def health_check(self) -> bool:
        try:
            client = self.connect()
            return bool(client.ping())
        except Exception as exc:
            logger.error(f"Elasticsearch health check failed: {exc}")
            return False

    def close(self) -> None:
        if self._client is None:
            return
        try:
            self._client.close()
        except Exception as exc:
            logger.warning(f"Failed to close Elasticsearch client: {exc}")
        finally:
            self._client = None


elasticsearch_manager = ElasticsearchManager()

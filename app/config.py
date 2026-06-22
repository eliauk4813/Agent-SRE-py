"""Application settings."""

from typing import Any, Dict

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "SuperBizAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    dashscope_api_key: str = ""
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"

    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000

    rag_top_k: int = 3
    rag_final_top_k: int = 5
    rag_milvus_top_k: int = 15
    rag_es_top_k: int = 15
    rag_rrf_k: int = 60
    rag_milvus_weight: float = 1.0
    rag_es_weight: float = 1.0
    rag_model: str = "qwen-max"

    es_url: str = "http://localhost:9200"
    es_index_name: str = "biz_chunks"
    es_request_timeout: int = 10
    es_analyzer: str = "standard"

    md_chunk_target_size: int = 900
    md_chunk_max_size: int = 1200
    md_chunk_overlap: int = 120
    md_chunk_min_size: int = 200

    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production"}:
                return False
            if normalized in {"debug", "dev", "development"}:
                return True
        return value

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            },
        }


config = Settings()

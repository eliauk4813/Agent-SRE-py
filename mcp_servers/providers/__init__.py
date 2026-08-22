"""Provider factory for MCP log tools."""

from __future__ import annotations

import os

from .elasticsearch_log_provider import ElasticsearchLogProvider
from .log_provider import LogProvider
from .mock_log_provider import MockLogProvider


_log_provider: LogProvider | None = None


def get_log_provider(force_new: bool = False) -> LogProvider:
    """Return the configured log provider."""
    global _log_provider

    if _log_provider is not None and not force_new:
        return _log_provider

    provider_type = os.getenv("AIOPS_LOG_PROVIDER", "mock").strip().lower()

    if provider_type in {"es", "elasticsearch"}:
        verify_certs = os.getenv("AIOPS_LOG_ES_VERIFY_CERTS", "true").lower() not in {
            "0",
            "false",
            "no",
        }
        _log_provider = ElasticsearchLogProvider(
            es_url=os.getenv("AIOPS_LOG_ES_URL", "http://localhost:9200"),
            index_pattern=os.getenv("AIOPS_LOG_INDEX_PATTERN", "springboot-logs-*"),
            username=os.getenv("AIOPS_LOG_ES_USERNAME") or None,
            password=os.getenv("AIOPS_LOG_ES_PASSWORD") or None,
            api_key=os.getenv("AIOPS_LOG_ES_API_KEY") or None,
            request_timeout=int(os.getenv("AIOPS_LOG_ES_TIMEOUT", "10")),
            verify_certs=verify_certs,
        )
    else:
        _log_provider = MockLogProvider()

    return _log_provider

"""Elasticsearch-backed provider for containerized Spring Boot service logs."""

from __future__ import annotations

from typing import Any, Optional

from .log_provider import LogProvider, normalize_time_range, to_iso


class ElasticsearchLogProvider(LogProvider):
    """Query structured application logs stored in Elasticsearch."""

    def __init__(
        self,
        es_url: str,
        index_pattern: str = "springboot-logs-*",
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout: int = 10,
        verify_certs: bool = True,
    ) -> None:
        try:
            from elasticsearch import Elasticsearch
        except ImportError as exc:
            raise RuntimeError("Install elasticsearch to use ElasticsearchLogProvider") from exc

        self.index_pattern = index_pattern
        kwargs: dict[str, Any] = {
            "request_timeout": request_timeout,
            "verify_certs": verify_certs,
        }
        if api_key:
            kwargs["api_key"] = api_key
        elif username and password:
            kwargs["basic_auth"] = (username, password)

        self.client = Elasticsearch(es_url, **kwargs)

    def search_logs(
        self,
        service_name: str,
        start_time: Optional[str | int] = None,
        end_time: Optional[str | int] = None,
        keyword: Optional[str] = None,
        level: Optional[str] = None,
        trace_id: Optional[str] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        status: Optional[int] = None,
        min_duration_ms: Optional[int] = None,
        env: Optional[str] = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        start_dt, end_dt = normalize_time_range(start_time, end_time)
        limit = max(1, min(limit, 500))

        query = self._build_query(
            service_name=service_name,
            start_time=to_iso(start_dt),
            end_time=to_iso(end_dt),
            keyword=keyword,
            level=level,
            trace_id=trace_id,
            request_id=request_id,
            path=path,
            status=status,
            min_duration_ms=min_duration_ms,
            env=env,
        )

        response = self.client.search(
            index=self.index_pattern,
            size=limit,
            sort=[{"@timestamp": {"order": "desc"}}],
            query=query,
        )

        hits = response.get("hits", {}).get("hits", [])
        logs = [self._normalize_hit(item) for item in hits]

        return {
            "provider": "elasticsearch",
            "index_pattern": self.index_pattern,
            "service_name": service_name,
            "start_time": to_iso(start_dt),
            "end_time": to_iso(end_dt),
            "keyword": keyword,
            "level": level,
            "trace_id": trace_id,
            "request_id": request_id,
            "path": path,
            "status": status,
            "min_duration_ms": min_duration_ms,
            "env": env,
            "limit": limit,
            "total": len(logs),
            "logs": logs,
        }

    def _build_query(
        self,
        service_name: str,
        start_time: str,
        end_time: str,
        keyword: Optional[str],
        level: Optional[str],
        trace_id: Optional[str],
        request_id: Optional[str],
        path: Optional[str],
        status: Optional[int],
        min_duration_ms: Optional[int],
        env: Optional[str],
    ) -> dict[str, Any]:
        filters: list[dict[str, Any]] = [
            {"range": {"@timestamp": {"gte": start_time, "lte": end_time}}},
            {
                "bool": {
                    "should": [
                        {"term": {"service_name": service_name}},
                        {"term": {"service.name": service_name}},
                        {"term": {"application": service_name}},
                        {"term": {"app": service_name}},
                    ],
                    "minimum_should_match": 1,
                }
            },
        ]

        if level:
            filters.append({"term": {"level": level.upper()}})
        if trace_id:
            filters.append({"term": {"trace_id": trace_id}})
        if request_id:
            filters.append({"term": {"request_id": request_id}})
        if path:
            filters.append(
                {
                    "bool": {
                        "should": [
                            {"term": {"path": path}},
                            {"term": {"url.path": path}},
                            {"match": {"path.text": path}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )
        if status is not None:
            filters.append(
                {
                    "bool": {
                        "should": [
                            {"term": {"status": status}},
                            {"term": {"http.response.status_code": status}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )
        if min_duration_ms is not None:
            filters.append({"range": {"duration_ms": {"gte": min_duration_ms}}})
        if env:
            filters.append({"term": {"env": env}})

        must: list[dict[str, Any]] = []
        if keyword:
            must.append(
                {
                    "multi_match": {
                        "query": keyword,
                        "fields": [
                            "message",
                            "exception",
                            "logger",
                            "path.text",
                            "log.original",
                        ],
                        "operator": "and",
                    }
                }
            )

        return {"bool": {"filter": filters, "must": must}}

    def _normalize_hit(self, item: dict[str, Any]) -> dict[str, Any]:
        source = item.get("_source", {})
        host = source.get("host")
        host_name = host.get("name") if isinstance(host, dict) else host
        service = source.get("service")
        service_name = service.get("name") if isinstance(service, dict) else None
        http = source.get("http")
        http_response = http.get("response") if isinstance(http, dict) else None

        return {
            "timestamp": source.get("@timestamp") or source.get("timestamp"),
            "service_name": source.get("service_name") or service_name or source.get("application"),
            "env": source.get("env") or source.get("environment"),
            "level": source.get("level") or source.get("log.level"),
            "message": source.get("message") or source.get("log.original"),
            "trace_id": source.get("trace_id") or source.get("trace.id"),
            "request_id": source.get("request_id"),
            "method": source.get("method") or source.get("http.request.method"),
            "path": source.get("path") or source.get("url.path"),
            "status": source.get("status")
            or (
                http_response.get("status_code")
                if isinstance(http_response, dict)
                else source.get("http.response.status_code")
            ),
            "duration_ms": source.get("duration_ms"),
            "exception": source.get("exception") or source.get("error.stack_trace"),
            "host": host_name,
            "container": source.get("container"),
            "raw": source,
        }

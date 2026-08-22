"""Mock log provider kept for local Agent workflow demos."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Optional

from .log_provider import LogProvider, normalize_time_range


class MockLogProvider(LogProvider):
    """Generate deterministic diagnostic logs without external services."""

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

        logs: list[dict[str, Any]] = []
        current = start_dt
        count = 0
        max_scan = 500

        while current <= end_dt and count < max_scan and len(logs) < limit:
            is_error = count % 7 == 3
            is_slow = count % 5 == 2
            log_level = "ERROR" if is_error else "WARN" if is_slow else "INFO"
            duration = 3200 if is_error else 1250 if is_slow else 86
            http_status = 500 if is_error else 200
            message = (
                "Database query timeout after 3000ms"
                if is_error
                else "HTTP request completed slowly"
                if is_slow
                else "HTTP request completed"
            )

            item = {
                "timestamp": current.isoformat(),
                "service_name": service_name,
                "env": env or "demo",
                "level": log_level,
                "message": message,
                "trace_id": trace_id or f"mock-trace-{count:04d}",
                "request_id": request_id or f"mock-request-{count:04d}",
                "method": "GET",
                "path": path or "/api/demo",
                "status": http_status,
                "duration_ms": duration,
                "exception": "java.sql.SQLTimeoutException: Query timeout" if is_error else "",
                "host": "mock-container",
            }

            if self._matches(item, keyword, level, status, min_duration_ms):
                logs.append(item)

            current += timedelta(minutes=1)
            count += 1

        return {
            "provider": "mock",
            "service_name": service_name,
            "start_time": start_dt.isoformat(),
            "end_time": end_dt.isoformat(),
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
            "message": f"Mock provider returned {len(logs)} logs",
        }

    def _matches(
        self,
        item: dict[str, Any],
        keyword: Optional[str],
        level: Optional[str],
        status: Optional[int],
        min_duration_ms: Optional[int],
    ) -> bool:
        if level and item.get("level") != level.upper():
            return False
        if status is not None and int(item.get("status") or 0) != status:
            return False
        if min_duration_ms is not None and int(item.get("duration_ms") or 0) < min_duration_ms:
            return False
        if keyword:
            haystack = " ".join(
                str(item.get(key, "")) for key in ("message", "exception", "path")
            ).lower()
            if keyword.lower() not in haystack:
                return False
        return True

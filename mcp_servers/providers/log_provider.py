"""Log provider interface used by the log MCP server."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, Optional


class LogProvider(ABC):
    """Backend-neutral interface for querying service logs."""

    @abstractmethod
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
        """Search logs by service and optional diagnostic filters."""
        raise NotImplementedError


def normalize_time_range(
    start_time: Optional[str | int],
    end_time: Optional[str | int],
    default_minutes: int = 15,
) -> tuple[datetime, datetime]:
    """Normalize timestamp inputs to timezone-aware UTC datetimes."""
    end_dt = parse_time(end_time) if end_time is not None else datetime.now(timezone.utc)
    start_dt = (
        parse_time(start_time)
        if start_time is not None
        else end_dt - timedelta(minutes=default_minutes)
    )
    return start_dt.astimezone(timezone.utc), end_dt.astimezone(timezone.utc)


def parse_time(value: str | int) -> datetime:
    """Parse epoch milliseconds, epoch seconds, ISO time, or common log time."""
    if isinstance(value, int):
        timestamp = value / 1000 if value > 10_000_000_000 else value
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    text = str(value).strip()
    if not text:
        return datetime.now(timezone.utc)

    if text.isdigit():
        return parse_time(int(text))

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def to_iso(dt: datetime) -> str:
    """Format a datetime for Elasticsearch range filters."""
    return dt.astimezone(timezone.utc).isoformat()

"""Persistent conversation memory service.

This module keeps chat history and rolling session summaries in SQLite. It is
intentionally small: the Agent can rebuild useful context from persisted memory
without depending on in-process LangGraph state.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from loguru import logger

from app.config import config


ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


@dataclass
class MemoryContext:
    """Memory snippets used to rebuild an agent prompt."""

    summary: str
    recent_messages: List[Dict[str, Any]]

    def to_prompt_text(self) -> str:
        parts: List[str] = []
        if self.summary:
            parts.append(f"## 会话摘要\n{self.summary.strip()}")

        if self.recent_messages:
            lines = []
            for message in self.recent_messages:
                role = "用户" if message.get("role") == ROLE_USER else "助手"
                content = str(message.get("content", "")).strip()
                if content:
                    lines.append(f"{role}: {content}")
            if lines:
                parts.append("## 最近对话\n" + "\n".join(lines))

        return "\n\n".join(parts).strip()


class ConversationMemoryService:
    """SQLite-backed chat memory with rolling summaries."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self._lock = Lock()
        self._initialize()
        logger.info(f"Conversation memory initialized: {self.db_path}")

    def _initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    summary TEXT NOT NULL DEFAULT '',
                    summarized_message_id INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id "
                "ON chat_messages(session_id, id)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def append_exchange(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist one user/assistant turn."""
        now = self._now()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._lock, self._connect() as conn:
            self._ensure_session(conn, session_id, now)
            conn.execute(
                """
                INSERT INTO chat_messages(session_id, role, content, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, ROLE_USER, user_content, metadata_json, now),
            )
            conn.execute(
                """
                INSERT INTO chat_messages(session_id, role, content, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, ROLE_ASSISTANT, assistant_content, metadata_json, now),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )

    def get_context(self, session_id: str, recent_turns: int) -> MemoryContext:
        """Return summary plus recent complete turns for prompt reconstruction."""
        limit = max(0, recent_turns) * 2
        with self._lock, self._connect() as conn:
            session = conn.execute(
                "SELECT summary FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            summary = str(session["summary"]) if session else ""
            rows = conn.execute(
                """
                SELECT id, role, content, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        recent = [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["created_at"],
            }
            for row in reversed(rows)
        ]
        return MemoryContext(summary=summary, recent_messages=recent)

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

        return [
            {
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["created_at"],
            }
            for row in rows
        ]

    def clear_session(self, session_id: str) -> bool:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        return True

    def should_update_summary(self, session_id: str, threshold_messages: int) -> bool:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COALESCE(MAX(m.id), 0) AS max_id,
                    COALESCE(s.summarized_message_id, 0) AS summarized_id
                FROM chat_messages m
                LEFT JOIN chat_sessions s ON s.session_id = m.session_id
                WHERE m.session_id = ?
                """,
                (session_id,),
            ).fetchone()

        if not row:
            return False
        return int(row["max_id"]) - int(row["summarized_id"]) >= threshold_messages

    def get_messages_for_summary(self, session_id: str) -> List[Dict[str, Any]]:
        """Return messages not yet covered by the rolling summary."""
        with self._lock, self._connect() as conn:
            session = conn.execute(
                """
                SELECT summarized_message_id
                FROM chat_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            summarized_id = int(session["summarized_message_id"]) if session else 0
            rows = conn.execute(
                """
                SELECT id, role, content, created_at
                FROM chat_messages
                WHERE session_id = ? AND id > ?
                ORDER BY id ASC
                """,
                (session_id, summarized_id),
            ).fetchall()

        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["created_at"],
            }
            for row in rows
        ]

    def get_summary(self, session_id: str) -> str:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT summary FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return str(row["summary"]) if row else ""

    def update_summary(self, session_id: str, summary: str, covered_message_id: int) -> None:
        now = self._now()
        with self._lock, self._connect() as conn:
            self._ensure_session(conn, session_id, now)
            conn.execute(
                """
                UPDATE chat_sessions
                SET summary = ?, summarized_message_id = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (summary.strip(), covered_message_id, now, session_id),
            )

    def _ensure_session(self, conn: sqlite3.Connection, session_id: str, now: str) -> None:
        conn.execute(
            """
            INSERT INTO chat_sessions(session_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (session_id, session_id, now, now),
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


conversation_memory_service = ConversationMemoryService(config.chat_memory_db_path)

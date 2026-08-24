"""Long-term memory storage for stable user/project facts."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

from app.config import config


TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


@dataclass
class LongTermMemory:
    id: int
    memory_type: str
    content: str
    importance: float
    metadata: Dict[str, Any]
    created_at: str


class LongTermMemoryService:
    """Store and retrieve durable facts extracted from conversations."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self._lock = Lock()
        self._initialize()
        logger.info(f"Long-term memory initialized: {self.db_path}")

    def _initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_key TEXT NOT NULL UNIQUE,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_long_term_memories_active "
                "ON long_term_memories(is_active, memory_type)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add_memories(
        self,
        memories: Iterable[Dict[str, Any]],
        session_id: str,
    ) -> int:
        """Insert or refresh extracted memory items."""
        now = self._now()
        saved = 0
        with self._lock, self._connect() as conn:
            for item in memories:
                content = str(item.get("content", "")).strip()
                if not content:
                    continue

                memory_type = str(item.get("type") or item.get("memory_type") or "fact")
                importance = self._normalize_importance(item.get("importance", 0.5))
                metadata = {
                    "source_session_id": session_id,
                    "raw": item,
                }
                memory_key = self._memory_key(memory_type, content)

                conn.execute(
                    """
                    INSERT INTO long_term_memories(
                        memory_key, memory_type, content, importance,
                        metadata, is_active, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                    ON CONFLICT(memory_key) DO UPDATE SET
                        importance = MAX(long_term_memories.importance, excluded.importance),
                        metadata = excluded.metadata,
                        is_active = 1,
                        updated_at = excluded.updated_at
                    """,
                    (
                        memory_key,
                        memory_type,
                        content,
                        importance,
                        json.dumps(metadata, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
                saved += 1

        if saved:
            logger.info(f"Saved {saved} long-term memories for session {session_id}")
        return saved

    def retrieve(self, query: str, top_k: int) -> List[LongTermMemory]:
        """Retrieve relevant memories using a lightweight lexical score."""
        query_tokens = self._tokens(query)
        if not query_tokens:
            return []

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, memory_type, content, importance, metadata, created_at
                FROM long_term_memories
                WHERE is_active = 1
                """
            ).fetchall()

        scored = []
        for row in rows:
            content = str(row["content"])
            memory_tokens = self._tokens(content)
            overlap = len(query_tokens & memory_tokens)
            if overlap == 0:
                continue
            score = overlap + float(row["importance"])
            scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        memories: List[LongTermMemory] = []
        for _, row in scored[: max(0, top_k)]:
            try:
                metadata = json.loads(row["metadata"] or "{}")
            except json.JSONDecodeError:
                metadata = {}
            memories.append(
                LongTermMemory(
                    id=int(row["id"]),
                    memory_type=str(row["memory_type"]),
                    content=str(row["content"]),
                    importance=float(row["importance"]),
                    metadata=metadata,
                    created_at=str(row["created_at"]),
                )
            )
        return memories

    def format_for_prompt(self, memories: List[LongTermMemory]) -> str:
        if not memories:
            return ""
        lines = []
        for memory in memories:
            lines.append(
                f"- [{memory.memory_type}] {memory.content} "
                f"(importance={memory.importance:.1f})"
            )
        return "\n".join(lines)

    def clear_all(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE long_term_memories SET is_active = 0")

    @staticmethod
    def _normalize_importance(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.5
        return max(0.0, min(1.0, number))

    @staticmethod
    def _memory_key(memory_type: str, content: str) -> str:
        normalized = " ".join(content.lower().split())
        return f"{memory_type}:{normalized}"

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token.lower() for token in TOKEN_RE.findall(text) if len(token.strip()) > 1}

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


long_term_memory_service = LongTermMemoryService(config.chat_memory_db_path)

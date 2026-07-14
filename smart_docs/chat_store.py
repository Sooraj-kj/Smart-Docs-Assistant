from __future__ import annotations

import sqlite3
from pathlib import Path

from smart_docs.config import CHAT_DB_PATH
from smart_docs.schemas import ChatSession, HistoryItem


class SQLiteChatStore:
    def __init__(self, db_path: Path = CHAT_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
                        ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
                ON chat_messages (session_id, id)
                """
            )

    def ensure_session(self, session_id: str, title: str | None = None) -> None:
        session_title = title or session_id
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_sessions (session_id, title)
                VALUES (?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (session_id, session_title),
            )

    def append_messages(self, session_id: str, messages: list[HistoryItem]) -> None:
        if not messages:
            return

        self.ensure_session(session_id, self._title_from_messages(messages) or session_id)
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO chat_messages (session_id, role, content)
                VALUES (?, ?, ?)
                """,
                [(session_id, message.role, message.content) for message in messages],
            )
            title = self._title_from_history(session_id, connection) or session_id
            connection.execute(
                """
                UPDATE chat_sessions
                SET title = ?, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
                """,
                (title, session_id),
            )

    def get_history(self, session_id: str) -> list[HistoryItem]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [HistoryItem(role=row["role"], content=row["content"]) for row in rows]

    def list_sessions(self) -> list[ChatSession]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    s.session_id,
                    s.title,
                    s.created_at,
                    s.updated_at,
                    COUNT(m.id) AS message_count
                FROM chat_sessions s
                LEFT JOIN chat_messages m ON m.session_id = s.session_id
                GROUP BY s.session_id, s.title, s.created_at, s.updated_at
                ORDER BY s.updated_at DESC, s.created_at DESC
                """
            ).fetchall()
        return [
            ChatSession(
                session_id=row["session_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=row["message_count"],
            )
            for row in rows
        ]

    @staticmethod
    def _title_from_messages(messages: list[HistoryItem]) -> str | None:
        for message in messages:
            if message.role == "user":
                return SQLiteChatStore._short_title(message.content)
        return None

    @staticmethod
    def _title_from_history(session_id: str, connection: sqlite3.Connection) -> str | None:
        row = connection.execute(
            """
            SELECT content
            FROM chat_messages
            WHERE session_id = ? AND role = 'user'
            ORDER BY id ASC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return SQLiteChatStore._short_title(row["content"])

    @staticmethod
    def _short_title(content: str) -> str:
        compact = " ".join(content.split())
        return compact[:57] + "..." if len(compact) > 60 else compact

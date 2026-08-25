"""Append-only SQLite record store for write-time memory."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import regex


TOKEN_RE = regex.compile(r"[\p{L}\p{N}]+")


@dataclass(frozen=True)
class Record:
    id: str
    session_id: str
    type: str
    content: str
    entities: list[str]
    created_at: str
    supersedes_id: str | None = None
    embedding: np.ndarray | None = None

    def search_text(self) -> str:
        entity_text = " ".join(self.entities)
        return f"{self.type} {entity_text} {self.content}".strip()

    def as_dict(self, include_embedding: bool = False) -> dict:
        value = {
            "id": self.id,
            "session_id": self.session_id,
            "type": self.type,
            "content": self.content,
            "entities": self.entities,
            "created_at": self.created_at,
            "supersedes_id": self.supersedes_id,
        }
        if include_embedding:
            value["embedding"] = (
                self.embedding.tolist() if self.embedding is not None else None
            )
        return value


class MemoryStore:
    """SQLite store whose public API cannot mutate or delete records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS records(
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                entities TEXT NOT NULL,
                created_at TEXT NOT NULL,
                supersedes_id TEXT NULL REFERENCES records(id),
                embedding BLOB
            );
            CREATE INDEX IF NOT EXISTS idx_records_session
              ON records(session_id);
            CREATE INDEX IF NOT EXISTS idx_records_supersedes
              ON records(supersedes_id);
            CREATE TRIGGER IF NOT EXISTS records_no_update
            BEFORE UPDATE ON records
            BEGIN
                SELECT RAISE(ABORT, 'records are append-only');
            END;
            CREATE TRIGGER IF NOT EXISTS records_no_delete
            BEFORE DELETE ON records
            BEGIN
                SELECT RAISE(ABORT, 'records are append-only');
            END;
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def add(self, record: Record) -> Record:
        vector = None
        if record.embedding is not None:
            vector = np.asarray(record.embedding, dtype=np.float32).tobytes()
        self.connection.execute(
            """INSERT OR IGNORE INTO records
               (id, session_id, type, content, entities, created_at,
                supersedes_id, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (record.id, record.session_id, record.type, record.content,
             json.dumps(record.entities, ensure_ascii=False),
             record.created_at, record.supersedes_id, vector),
        )
        self.connection.commit()
        stored = self.get(record.id)
        if stored is None:
            raise RuntimeError(f"record was not stored: {record.id}")
        if stored.as_dict() != record.as_dict():
            raise ValueError(f"record id collision with different content: {record.id}")
        return stored

    @staticmethod
    def new_id() -> str:
        return f"mem_{uuid.uuid4().hex[:16]}"

    def get(self, record_id: str) -> Record | None:
        row = self.connection.execute(
            "SELECT * FROM records WHERE id = ?", (record_id,)
        ).fetchone()
        return self._from_row(row) if row else None

    def all(self) -> list[Record]:
        rows = self.connection.execute(
            "SELECT * FROM records ORDER BY rowid"
        ).fetchall()
        return [self._from_row(row) for row in rows]

    def count(self) -> int:
        return int(self.connection.execute(
            "SELECT COUNT(*) FROM records"
        ).fetchone()[0])

    def candidates(self, query_or_session: str, k: int = 10) -> list[Record]:
        """Return bounded lexical/recency candidates for extractor context."""
        records = self.all()
        if not records:
            return []
        query_tokens = self._tokens(query_or_session)
        scored = []
        total = len(records)
        for index, record in enumerate(records):
            record_tokens = self._tokens(record.search_text())
            overlap = len(query_tokens & record_tokens)
            coverage = overlap / max(len(query_tokens), 1)
            recency = (index + 1) / total
            scored.append((overlap + coverage + 0.05 * recency, index, record))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        relevant = [item for item in scored if item[0] > 0.05]
        if len(relevant) < min(k, len(records)):
            seen = {item[2].id for item in relevant}
            relevant.extend(
                item for item in reversed(scored)
                if item[2].id not in seen
            )
        return [item[2] for item in relevant[:k]]

    def successors(self, record_id: str) -> list[Record]:
        rows = self.connection.execute(
            "SELECT * FROM records WHERE supersedes_id = ? ORDER BY rowid",
            (record_id,),
        ).fetchall()
        return [self._from_row(row) for row in rows]

    def successor_chain(self, record_id: str) -> list[Record]:
        chain: list[Record] = []
        frontier = [record_id]
        seen = {record_id}
        while frontier:
            current = frontier.pop(0)
            for successor in self.successors(current):
                if successor.id in seen:
                    continue
                chain.append(successor)
                frontier.append(successor.id)
                seen.add(successor.id)
        return chain

    def is_superseded(self, record_id: str) -> bool:
        return bool(self.connection.execute(
            "SELECT 1 FROM records WHERE supersedes_id = ? LIMIT 1",
            (record_id,),
        ).fetchone())

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token.casefold() for token in TOKEN_RE.findall(text)}

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Record:
        blob = row["embedding"]
        embedding = (np.frombuffer(blob, dtype=np.float32).copy()
                     if blob is not None else None)
        return Record(
            id=row["id"],
            session_id=row["session_id"],
            type=row["type"],
            content=row["content"],
            entities=json.loads(row["entities"]),
            created_at=row["created_at"],
            supersedes_id=row["supersedes_id"],
            embedding=embedding,
        )

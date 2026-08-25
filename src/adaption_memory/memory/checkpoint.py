"""Hash-addressed JSONL checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from adaption_memory.evals.common import append_jsonl, read_jsonl


def stable_hash(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def replay_usage(target, snapshot: dict | None) -> None:
    """Restore one checkpointed call delta without firing spend callbacks."""
    if not snapshot:
        return
    for field in ("calls", "prompt_tokens", "completion_tokens",
                  "reasoning_tokens", "latency_ms", "cost_usd"):
        setattr(target, field, getattr(target, field) + snapshot.get(field, 0))


class Checkpoint:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._rows = {
            row["input_hash"]: row
            for row in read_jsonl(self.path)
            if isinstance(row, dict) and row.get("input_hash")
        }

    def get(self, input_hash: str) -> dict | None:
        return self._rows.get(input_hash)

    def append(self, row: dict) -> None:
        input_hash = row.get("input_hash")
        if not input_hash:
            raise ValueError("checkpoint rows require input_hash")
        existing = self._rows.get(input_hash)
        if existing is not None:
            if existing != row:
                raise ValueError(f"checkpoint hash collision in {self.path}")
            return
        append_jsonl(self.path, row)
        self._rows[input_hash] = row

    def rows(self) -> list[dict]:
        return list(self._rows.values())

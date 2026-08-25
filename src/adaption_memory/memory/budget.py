"""Atomic API spend accounting and hard-cap enforcement."""

from __future__ import annotations

import json
import os
import fcntl
from datetime import datetime, timezone
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


class SpendTracker:
    def __init__(self, path: str | Path, cap_usd: float = 40.0):
        self.path = Path(path)
        self.cap_usd = cap_usd
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "schema": 1,
                "cap_usd": cap_usd,
                "total_usd": 0.0,
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "latency_ms": 0.0,
                "by_model": {},
                "events": [],
                "note": "Counts only API calls made by the overnight pipeline; pre-existing baselines are excluded.",
            }
            self._write()

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - float(self.data["total_usd"]))

    def before_call(self, call: dict) -> None:
        if call["model"] != "gpt-5.6-luna":
            return
        characters = sum(len(str(message.get("content", "")))
                         for message in call.get("messages", []))
        projected_input = max(1, characters // 4)
        projected = (
            projected_input * 0.20 / 1_000_000
            + int(call.get("max_tokens", 0)) * 1.20 / 1_000_000
        )
        with self._lock():
            self._reload()
            if projected > self.remaining_usd:
                raise BudgetExceeded(
                    f"projected call ${projected:.4f} exceeds remaining "
                    f"overnight budget ${self.remaining_usd:.4f}"
                )

    def on_call(self, event: dict) -> None:
        with self._lock():
            self._reload()
            self._apply_event(event)
            self._write()

    def _apply_event(self, event: dict) -> None:
        model = event["model"]
        self.data["calls"] += 1
        for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
            self.data[field] += int(event.get(field, 0))
        self.data["latency_ms"] = round(
            float(self.data["latency_ms"]) + float(event.get("latency_ms", 0)), 3
        )
        self.data["total_usd"] = round(
            float(self.data["total_usd"]) + float(event.get("cost_usd", 0)), 8
        )
        model_row = self.data["by_model"].setdefault(model, {
            "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "reasoning_tokens": 0, "latency_ms": 0.0, "cost_usd": 0.0,
        })
        model_row["calls"] += 1
        for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
            model_row[field] += int(event.get(field, 0))
        model_row["latency_ms"] = round(
            model_row["latency_ms"] + float(event.get("latency_ms", 0)), 3
        )
        model_row["cost_usd"] = round(
            model_row["cost_usd"] + float(event.get("cost_usd", 0)), 8
        )
        self.data["events"].append({
            "at": datetime.now(timezone.utc).isoformat(), **event,
        })

    def _reload(self) -> None:
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def _lock(self):
        tracker = self

        class FileLock:
            def __enter__(self):
                tracker.path.parent.mkdir(parents=True, exist_ok=True)
                self.handle = open(tracker.path.with_suffix(".lock"), "a+")
                fcntl.flock(self.handle, fcntl.LOCK_EX)
                return self

            def __exit__(self, exc_type, exc, traceback):
                fcntl.flock(self.handle, fcntl.LOCK_UN)
                self.handle.close()

        return FileLock()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(
            self.path.suffix + f".{os.getpid()}.tmp"
        )
        temporary.write_text(json.dumps(self.data, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(self.path)

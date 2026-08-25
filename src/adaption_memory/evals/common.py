"""Shared runner plumbing: resumable JSONL answer files and summaries."""

import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def done_ids(path: Path, key: str) -> set:
    return {r[key] for r in read_jsonl(path)}


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def usage_delta(before: dict, after: dict) -> dict:
    """Subtract two cumulative usage snapshots."""
    return {key: after.get(key, 0) - before.get(key, 0)
            for key in before.keys() | after.keys()}


def add_usage(*snapshots: dict) -> dict:
    """Combine independent usage deltas."""
    keys = set().union(*(snapshot.keys() for snapshot in snapshots))
    return {key: sum(snapshot.get(key, 0) for snapshot in snapshots)
            for key in keys}

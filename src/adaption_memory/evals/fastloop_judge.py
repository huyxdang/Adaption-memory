"""Build store-recall judging packets for a fastloop run.

Dumps one JSON packet per run directory: every scorable smoke question with
its reference answer and the full store contents of its conversation. An
external LLM judge (spawned by the operator) scores each question as
contained / partial / absent; verdicts are written back with `apply`.

Usage:
    uv run python -m adaption_memory.evals.fastloop_judge dump <run_dir>
    uv run python -m adaption_memory.evals.fastloop_judge apply <run_dir> \
        <verdicts.json>

The packet deliberately contains only store contents, never transcripts, so
the judge measures what survived extraction, not what it can re-extract.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from adaption_memory.overnight import load_conversations, recall_proxy


def store_contents(run_dir: Path, benchmark: str, conversation_id: str) -> str:
    path = run_dir / benchmark / "stores" / f"{conversation_id}.sqlite3"
    if not path.exists():
        return ""
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT type, content FROM records ORDER BY rowid"
        ).fetchall()
    return "\n".join(f"[{kind}] {content}" for kind, content in rows)


def dump(run_dir: Path) -> Path:
    items = []
    for benchmark in ("longmemeval", "locomo", "beam"):
        if not (run_dir / benchmark).exists():
            continue
        for conversation in load_conversations("smoke", benchmark):
            store = store_contents(
                run_dir, benchmark,
                "".join(c if c.isalnum() or c in "-_" else "_"
                        for c in conversation.id),
            )
            for question in conversation.questions:
                if recall_proxy(question.reference, question.category,
                                "") is None:
                    continue
                items.append({
                    "question_id": question.id,
                    "benchmark": benchmark,
                    "category": question.category,
                    "question": question.text,
                    "reference": question.reference,
                    "store": store,
                })
    out = run_dir / "judge_packet.json"
    out.write_text(json.dumps(items, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"{out}: {len(items)} questions")
    return out


def apply(run_dir: Path, verdicts_path: Path) -> None:
    verdicts = json.loads(verdicts_path.read_text())
    if isinstance(verdicts, list):
        verdicts = {row["question_id"]: row["verdict"] for row in verdicts}
    packet = json.loads((run_dir / "judge_packet.json").read_text())
    score = {"contained": 1.0, "partial": 0.5, "absent": 0.0}
    by_benchmark: dict[str, list[float]] = {}
    rows = []
    for item in packet:
        verdict = str(verdicts.get(item["question_id"], "absent")).lower()
        value = score.get(verdict)
        if value is None:
            raise SystemExit(f"bad verdict {verdict!r} "
                             f"for {item['question_id']}")
        by_benchmark.setdefault(item["benchmark"], []).append(value)
        rows.append({"question_id": item["question_id"],
                     "benchmark": item["benchmark"], "verdict": verdict})
    summary = {
        "scale": "contained=1, partial=0.5, absent=0",
        "benchmarks": {name: round(sum(values) / len(values), 4)
                       for name, values in sorted(by_benchmark.items())},
        "macro": round(sum(sum(v) / len(v) for v in by_benchmark.values())
                       / len(by_benchmark), 4),
        "verdicts": rows,
    }
    out = run_dir / "judged_store_recall.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "verdicts"},
                     indent=2))


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in {"dump", "apply"}:
        raise SystemExit(__doc__)
    run_dir = Path(sys.argv[2])
    if sys.argv[1] == "dump":
        dump(run_dir)
    else:
        apply(run_dir, Path(sys.argv[3]))


if __name__ == "__main__":
    main()

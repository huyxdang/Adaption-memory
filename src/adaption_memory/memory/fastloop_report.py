"""Static HTML inspector for fastloop runs.

Builds one self-contained page from the newest run directory of every arm
under ``results/fastloop/``: a per-question hit/miss comparison across arms,
and, per conversation, the full extraction trace — what the model was shown,
what it wrote, what validation rejected and why, and what ended up in the
store. No external assets, so the file can be opened or shared as-is.
"""

from __future__ import annotations

import html
import json
import sqlite3
from pathlib import Path

from adaption_memory.evals.common import read_jsonl

CSS = """
body{font:15px/1.5 -apple-system,'Segoe UI',sans-serif;margin:0;padding:2rem;
     background:#f6f7f9;color:#1c2733;max-width:1100px;margin-inline:auto}
h1{font-size:1.5rem}h2{margin-top:2.5rem;border-bottom:2px solid #d6dce3}
h3{margin:1.2rem 0 .4rem}
table{border-collapse:collapse;width:100%;background:#fff;font-size:.92rem}
th,td{border:1px solid #d6dce3;padding:.45rem .6rem;text-align:left;
      vertical-align:top}
th{background:#eef1f5}
.hit{background:#d9f2dd;font-weight:600}.miss{background:#fadbd8;font-weight:600}
.na{color:#8a97a5}
.rej{color:#a33;margin:.15rem 0}
.rec{margin:.15rem 0}
.type{display:inline-block;min-width:5.2rem;font-size:.78rem;font-weight:700;
      color:#456}
details{margin:.4rem 0;background:#fff;border:1px solid #d6dce3;
        border-radius:6px;padding:.5rem .8rem}
details details{background:#fafbfc}
summary{cursor:pointer;font-weight:600}
pre{white-space:pre-wrap;font-size:.82rem;background:#f0f2f5;padding:.6rem;
    border-radius:4px;max-height:24rem;overflow:auto}
.meta{color:#5a6775;font-size:.85rem}
"""


def newest_run(arm_dir: Path) -> Path | None:
    runs = [path for path in arm_dir.iterdir() if path.is_dir()]
    return max(runs, key=lambda path: path.stat().st_mtime) if runs else None


def store_records(store_path: Path) -> list[dict]:
    with sqlite3.connect(store_path) as connection:
        rows = connection.execute(
            "SELECT id, session_id, type, content, supersedes_id "
            "FROM records ORDER BY rowid"
        ).fetchall()
    superseded = {row[4] for row in rows if row[4]}
    return [{"id": row[0], "session_id": row[1], "type": row[2],
             "content": row[3], "supersedes_id": row[4],
             "superseded": row[0] in superseded} for row in rows]


def esc(value) -> str:
    return html.escape(str(value))


def load_arm(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    arm = {
        "run_dir": run_dir,
        "summary": (json.loads(summary_path.read_text())
                    if summary_path.exists() else {}),
        "benchmarks": {},
    }
    for benchmark_dir in sorted(path for path in run_dir.iterdir()
                                if path.is_dir()):
        misses = {row["question_id"]: row
                  for row in read_jsonl(benchmark_dir / "misses.jsonl")}
        conversations = {}
        checkpoint_root = benchmark_dir / "ckpt"
        if checkpoint_root.exists():
            for conversation_dir in sorted(checkpoint_root.iterdir()):
                conversations[conversation_dir.name] = read_jsonl(
                    conversation_dir / "extractions.jsonl"
                )
        stores = {path.stem: store_records(path)
                  for path in sorted(
                      (benchmark_dir / "stores").glob("*.sqlite3"))}
        arm["benchmarks"][benchmark_dir.name] = {
            "misses": misses, "conversations": conversations,
            "stores": stores,
        }
    return arm


def question_rows(benchmark: str) -> list[dict]:
    # Imported lazily: overnight imports this module's sibling packages, and
    # the benchmark question list lives next to the harness on purpose.
    from adaption_memory.overnight import load_conversations, recall_proxy

    rows = []
    for conversation in load_conversations("smoke", benchmark):
        for question in conversation.questions:
            scorable = recall_proxy(
                question.reference, question.category, ""
            ) is not None
            rows.append({
                "id": question.id, "conversation_id": conversation.id,
                "category": question.category, "question": question.text,
                "reference": question.reference, "scorable": scorable,
            })
    return rows


def status_cell(question: dict, benchmark_data: dict | None) -> str:
    if benchmark_data is None:
        return '<td class="na">not run</td>'
    if not question["scorable"]:
        return '<td class="na">n/a (abstention)</td>'
    miss = benchmark_data["misses"].get(question["id"])
    if miss is None:
        return '<td class="hit">saved</td>'
    label = ("missed at write" if miss["bucket"] == "fact_not_extracted"
             else "saved, lost in retrieval")
    return f'<td class="miss">{esc(label)}</td>'


def render_conversation(conversation_id: str, extractions: list[dict],
                        records: list[dict]) -> str:
    parts = [f"<details><summary>Conversation {esc(conversation_id)} — "
             f"{len(records)} records in store, "
             f"{len(extractions)} sessions</summary>"]
    parts.append("<h4>Final store</h4>")
    if records:
        for record in records:
            mark = " <s>(superseded)</s>" if record["superseded"] else ""
            parts.append(
                f'<div class="rec"><span class="type">{esc(record["type"])}'
                f"</span> {esc(record['content'])}{mark}</div>"
            )
    else:
        parts.append('<p class="meta">Store is empty.</p>')
    parts.append("<h4>Per-session extraction</h4>")
    for row in extractions:
        accepted = row.get("records", [])
        rejected = row.get("rejected", [])
        parts.append(
            f"<details><summary>Session {esc(row.get('session_id'))} — "
            f"{len(accepted)} accepted, {len(rejected)} rejected"
            f"{', repaired' if row.get('repaired') else ''}</summary>"
        )
        for record in accepted:
            parts.append(
                f'<div class="rec"><span class="type">{esc(record["type"])}'
                f"</span> {esc(record['content'])}</div>"
            )
        for rejection in rejected:
            content = rejection.get("record", {}).get("content", "")
            reasons = "; ".join(rejection.get("reasons", []))
            parts.append(f'<div class="rej">✗ {esc(content)}'
                         f'<br><span class="meta">{esc(reasons)}</span></div>')
        model_input = row.get("input", "")
        raw = row.get("raw_output", "")
        if model_input:
            parts.append(f"<details><summary>Model input</summary>"
                         f"<pre>{esc(model_input)}</pre></details>")
        if raw:
            parts.append(f"<details><summary>Raw model output</summary>"
                         f"<pre>{esc(raw)}</pre></details>")
        parts.append("</details>")
    parts.append("</details>")
    return "".join(parts)


def build_fastloop_report(fastloop_root: Path,
                          out_path: Path | None = None) -> Path:
    arms = {}
    for arm_dir in sorted(path for path in fastloop_root.iterdir()
                          if path.is_dir()):
        run_dir = newest_run(arm_dir)
        if run_dir is not None:
            arms[arm_dir.name] = load_arm(run_dir)
    if not arms:
        raise SystemExit(f"no fastloop runs under {fastloop_root}")

    benchmarks = sorted({name for arm in arms.values()
                         for name in arm["benchmarks"]})
    parts = [f"<style>{CSS}</style><h1>Fastloop extraction inspector</h1>"]
    parts.append("<table><tr><th>Arm</th><th>Run</th><th>Wall s</th>"
                 + "".join(f"<th>{esc(name)} recall</th>"
                           for name in benchmarks) + "</tr>")
    for arm_name, arm in arms.items():
        summary = arm["summary"]
        cells = []
        for name in benchmarks:
            value = (summary.get("benchmarks", {}).get(name, {})
                     .get("store_recall"))
            cells.append(f"<td>{value if value is not None else '—'}</td>")
        parts.append(f"<tr><td>{esc(arm_name)}</td>"
                     f"<td class='meta'>{esc(arm['run_dir'].name)}</td>"
                     f"<td>{esc(summary.get('wall_seconds', '—'))}</td>"
                     + "".join(cells) + "</tr>")
    parts.append("</table>")

    for benchmark in benchmarks:
        parts.append(f"<h2>{esc(benchmark)}</h2>")
        parts.append("<table><tr><th>Category</th><th>Question</th>"
                     "<th>Gold answer</th>"
                     + "".join(f"<th>{esc(arm_name)}</th>"
                               for arm_name in arms) + "</tr>")
        for question in question_rows(benchmark):
            cells = "".join(
                status_cell(question, arm["benchmarks"].get(benchmark))
                for arm in arms.values()
            )
            parts.append(
                f"<tr><td>{esc(question['category'])}</td>"
                f"<td>{esc(question['question'])}</td>"
                f"<td>{esc(question['reference'])}</td>{cells}</tr>"
            )
        parts.append("</table>")
        for arm_name, arm in arms.items():
            benchmark_data = arm["benchmarks"].get(benchmark)
            if benchmark_data is None:
                continue
            parts.append(f"<h3>{esc(arm_name)} — extraction trace</h3>")
            for conversation_id, extractions in sorted(
                benchmark_data["conversations"].items()
            ):
                records = benchmark_data["stores"].get(conversation_id, [])
                parts.append(render_conversation(
                    conversation_id, extractions, records
                ))

    out_path = out_path or fastloop_root / "report.html"
    out_path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>Fastloop inspector</title>" + "".join(parts),
        encoding="utf-8",
    )
    return out_path

"""Build filtered Northstar-style SFT artifacts from Luna extractor traces."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from adaption_memory.evals.common import read_jsonl
from adaption_memory.memory.prompts import production_prompt


def build_sft(*, root: Path, format_name: str, source_run: Path,
              prompt_revision: str = "base") -> dict:
    output_dir = root / "sft" / "northstar-style"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(source_run.glob("*/extractions.jsonl")):
        benchmark = path.parent.name
        for extraction in read_jsonl(path):
            rows.append((benchmark, extraction))
    accepted, rejected = [], []
    seen = set()
    for benchmark, row in rows:
        reason = None
        if not row.get("schema_valid"):
            reason = "initial output was not schema-valid"
        elif row.get("rejected"):
            reason = "one or more records failed validation"
        elif not row.get("scope_id"):
            reason = "missing auditable conversation scope"
        elif row["input_hash"] in seen:
            reason = "duplicate input hash"
        if reason:
            rejected.append({
                "input_hash": row.get("input_hash"),
                "benchmark": benchmark, "reason": reason,
            })
            continue
        seen.add(row["input_hash"])
        accepted.append({
            "messages": [
                {"role": "system", "content": production_prompt(
                    format_name, prompt_revision
                )},
                {"role": "user", "content": row["input"]},
                {"role": "assistant", "content": json.dumps(
                    row["accepted_output"], ensure_ascii=False,
                    separators=(",", ":"),
                )},
            ],
            "metadata": {
                "benchmark": benchmark,
                "scope_id": row["scope_id"],
                "session_id": row["session_id"],
                "input_hash": row["input_hash"],
                "format": format_name,
            },
        })
    train_path = output_dir / "train.jsonl"
    train_path.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in accepted
    ), encoding="utf-8")
    rejection_path = output_dir / "rejections.jsonl"
    rejection_path.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rejected
    ), encoding="utf-8")
    summary = {
        "schema": 1,
        "format": format_name,
        "prompt_revision": prompt_revision,
        "source_run": str(source_run.relative_to(root)),
        "input_pairs": len(rows),
        "accepted_pairs": len(accepted),
        "rejected_pairs": len(rejected),
        "rejection_rate": round(len(rejected) / len(rows), 4) if rows else None,
        "training_started": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        f"""# Northstar-style extractor SFT dataset

This dataset contains {len(accepted)} validated Luna-target extraction pairs
from `{source_run.relative_to(root)}` using format {format_name}. Each assistant
target contains only records accepted by the production validator; any session
with a schema, supersession-reference, or exact-value failure is excluded.

## Provenance and contamination boundary

Sources are the benchmark **signal-dev development subsample**, not official
benchmark train splits: LongMemEval, LoCoMo, and BEAM. This repository did not
contain separately identified official train-split files, so no additional
sessions were sampled from the flat full datasets. Do not describe these pairs
as official train-split data. They are suitable for pipeline prototyping, but a
future official full evaluation must keep its held-out questions isolated and
should prefer a replacement dataset built from verified train splits.

- Source run: `{source_run.relative_to(root)}`
- Extractor prompt: `{prompt_revision}`
- Accepted pairs: {len(accepted)}
- Rejected pairs: {len(rejected)}
- Rejection rate: {summary['rejection_rate']}
- Training started: no
""",
        encoding="utf-8",
    )
    config = {
        "model": "Qwen/Qwen3-4B",
        "method": "lora",
        "lora": {
            "rank": 64, "alpha": 128,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "training": {
            "epochs": 3,
            "learning_rate": 0.0001,
            "lr_schedule": "cosine",
            "mixture": {
                "memory_extraction_percent": 90,
                "general_purpose_percent": 10,
                "note": "Add general-purpose examples before training.",
            },
        },
        "data": {
            "train": "northstar-style/train.jsonl",
            "format": format_name,
            "prompt_revision": prompt_revision,
        },
    }
    (root / "sft" / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (root / "sft" / "EVAL_PLAN.md").write_text(
        f"""# Extractor SFT evaluation plan

The tuned Qwen3-4B extractor plugs into the same `overnight-memory` harness as
the zero-shot and few-shot arms. The frozen storage format is **{format_name}**.
The frozen extractor prompt revision is **{prompt_revision}**.
The answerer and judge remain `gpt-5.6-luna` with reasoning effort `none`.

## Gate order

1. Smoke: require at least 95% schema validity and complete pipeline output.
2. Signal-dev: compare extraction recall proxy and supersession accuracy with
   the untuned Qwen3-4B few-shot and Luna-target traces.
3. Signal-holdout: run once only after choosing a checkpoint.
4. Full: remains manual and prohibited by the overnight harness.

## Success criterion

Before any full benchmark run, the tuned extractor must close at least 70% of
the Qwen3-4B-fewshot versus Luna-target gap on both direct extractor metrics.
End-to-end judge accuracy is reported but does not replace that gate.

No training was performed by the overnight plan.
""",
        encoding="utf-8",
    )
    return summary

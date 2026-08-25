"""Curate the deterministic half tier from the full benchmark files.

User-authorized escalation past the signal tier (2026-08-26): half of each
full benchmark, selected deterministically and stratified, written in the
native formats under data/mini/half/ with a manifest of counts and hashes.
The full tier itself remains unsupported.

Selection rules (recorded in the manifest):
- LongMemEval_S: half of the 500 instances, stratified by question_type,
  ranked inside each stratum by sha256("half-v1:<type>:<question_id>").
- LoCoMo: 5 of 10 conversations by the same hash ranking; within each,
  half of the QA pairs, stratified by category and hash-ranked. (Full QA
  at 20K-token contexts alone exceeded the authorized budget.)
- BEAM 100K: 10 of 20 conversations by hash ranking; within each, one
  probing question per ability (of two), hash-ranked — same budget reason.

Usage: uv run python -m adaption_memory.evals.half
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "mini" / "half"
SEED = "half-v1"


def rank(*parts: str) -> str:
    return hashlib.sha256(":".join((SEED,) + parts).encode()).hexdigest()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def curate() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"schema": 1, "seed": SEED, "selection": {}}

    instances = json.loads((ROOT / "data" / "longmemeval_s_cleaned.json")
                          .read_text())
    by_type = collections.defaultdict(list)
    for instance in instances:
        by_type[instance["question_type"]].append(instance)
    chosen = []
    per_type = {}
    for question_type, group in sorted(by_type.items()):
        group.sort(key=lambda i: rank(question_type, str(i["question_id"])))
        keep = (len(group) + 1) // 2
        chosen.extend(group[:keep])
        per_type[question_type] = f"{keep}/{len(group)}"
    (OUT / "longmemeval.json").write_text(json.dumps(chosen))
    manifest["selection"]["longmemeval"] = {
        "instances": f"{len(chosen)}/{len(instances)}",
        "per_type": per_type,
        "abstention_kept": sum(1 for i in chosen
                               if str(i["question_id"]).endswith("_abs")),
    }

    samples = json.loads((ROOT / "data" / "locomo10.json").read_text())
    samples.sort(key=lambda s: rank("locomo", str(s["sample_id"])))
    kept = samples[:5]
    total_qa = 0
    for sample in kept:
        by_cat = collections.defaultdict(list)
        for index, item in enumerate(sample["qa"]):
            by_cat[item.get("category")].append((index, item))
        chosen_qa = []
        for category, group in sorted(by_cat.items(), key=lambda kv: str(kv[0])):
            group.sort(key=lambda pair: rank(
                "locomo-qa", str(sample["sample_id"]), str(category),
                str(pair[0])))
            chosen_qa.extend(group[:(len(group) + 1) // 2])
        chosen_qa.sort(key=lambda pair: pair[0])
        sample["qa"] = [item for _, item in chosen_qa]
        total_qa += len(sample["qa"])
    (OUT / "locomo.json").write_text(json.dumps(kept))
    manifest["selection"]["locomo"] = {
        "conversations": f"{len(kept)}/{len(samples)}",
        "sample_ids": [s["sample_id"] for s in kept],
        "qa_pairs": f"{total_qa} (half per category per conversation)",
    }

    import ast
    import pyarrow as pa
    table = pq.read_table(ROOT / "data" / "beam_100k.parquet")
    ids = [str(v) for v in table.column("conversation_id").to_pylist()]
    order = sorted(range(len(ids)), key=lambda i: rank("beam", ids[i]))
    keep_rows = sorted(order[:10])
    subset = table.take(keep_rows)
    trimmed, kept_questions = [], 0
    for row_index in range(subset.num_rows):
        conv_id = str(subset.column("conversation_id")[row_index].as_py())
        questions = ast.literal_eval(
            subset.column("probing_questions")[row_index].as_py())
        for ability, group in questions.items():
            indexed = list(enumerate(group))
            indexed.sort(key=lambda pair: rank(
                "beam-q", conv_id, ability, str(pair[0])))
            keep = indexed[:(len(indexed) + 1) // 2]
            keep.sort(key=lambda pair: pair[0])
            questions[ability] = [item for _, item in keep]
            kept_questions += len(questions[ability])
        trimmed.append(repr(questions))
    column_index = subset.schema.get_field_index("probing_questions")
    subset = subset.set_column(column_index, "probing_questions",
                               pa.array(trimmed))
    pq.write_table(subset, OUT / "beam.parquet")
    manifest["selection"]["beam"] = {
        "conversations": f"{len(keep_rows)}/{len(ids)}",
        "conversation_ids": [ids[i] for i in keep_rows],
        "questions": f"{kept_questions} (one per ability per conversation)",
    }

    manifest["files"] = {name: sha(OUT / name)
                         for name in ("longmemeval.json", "locomo.json",
                                      "beam.parquet")}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(curate(), indent=2))

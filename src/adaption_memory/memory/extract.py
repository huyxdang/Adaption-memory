"""LLM extractor with validation, repair, and durable checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import regex
from json_repair import repair_json

from adaption_memory.evals.common import usage_delta
from adaption_memory.interface import Session
from adaption_memory.llm import LLM
from adaption_memory.memory.checkpoint import Checkpoint, replay_usage, stable_hash
from adaption_memory.memory.embedding import LocalEmbedder
from adaption_memory.memory.prompts import (LINES_EMISSION_INSTRUCTION,
                                             REPLACES_EMISSION_INSTRUCTION,
                                             SIMPLE_EMISSION_INSTRUCTION,
                                             extraction_schema,
                                             fewshot_messages,
                                             production_prompt,
                                             simple_extraction_schema)
from adaption_memory.memory.store import MemoryStore, Record


NUMBER_OR_DATE_TOKEN = regex.compile(
    r"(?<![\p{L}\p{N}])(?:[$€£₫¥]?\d(?:[\d,]*\d)?(?:\.\d+)?"
    r"(?:[/-]\d(?:[\d,]*\d)?(?:\.\d+)?)*"
    r"|\d{1,2}:\d{2}|[A-Z]{1,6}-\d[\p{L}\p{N}-]*)"
)

LOCAL_MAX_RECORDS = 6
LOCAL_MAX_TOKENS = 512
LOCAL_INFERENCE_REVISION = "compact-v2"

LINE_PATTERN = regex.compile(
    r"^\s*(?:[-*]\s*)?(A|N)(?:\s+updates\s+(\d+))?\s*:\s*(.+?)\s*$",
    regex.IGNORECASE,
)
ENTITY_STOPWORDS = {
    "The", "A", "An", "In", "On", "At", "It", "He", "She", "They", "We",
    "His", "Her", "Their", "This", "That", "These", "Those", "When", "After",
    "Before", "During", "User", "Assistant", "Session", "I", "You",
}


def derive_entities(content: str, cap: int = 3) -> list[str]:
    """Capitalized-token heuristic replacing model-emitted entities for the
    simple/lines emissions; used only as a retrieval boost."""
    seen = []
    for token in regex.findall(r"\b\p{Lu}[\p{L}'-]+\b", content):
        if token in ENTITY_STOPWORDS or token in seen:
            continue
        seen.append(token)
        if len(seen) >= cap:
            break
    return seen
LOCAL_OUTPUT_INSTRUCTION = f"""
Local inference constraints:
- Use the compact transport {{"r":[{{"t":type_code,"c":content,
  "e":[entities],"s":supersedes_id}}]}}. Type codes are n=narrative,
  a=atomic, and m=memory; use only the codes allowed by the response schema.
- Candidate rows use i=id, t=type, c=content, and d=created_at. Use a shown i
  value as s only when the current session genuinely updates that memory.
- Return at most {LOCAL_MAX_RECORDS} highest-value new records.
- Keep each content field under 40 words.
- Return at most three explicit entity names per record.
- Return compact one-line JSON without indentation.
- Never copy a long list or passage; preserve only its durable conclusion.
- Never emit a date, number, or time by itself; attach it to the fact it qualifies.
- Prefer explicit user facts, decisions, changes, and constraints.
"""


@dataclass
class ExtractionResult:
    input_hash: str
    records: list[Record]
    rejected: list[dict]
    schema_valid: bool
    repaired: bool
    input_text: str


class MemoryExtractor:
    def __init__(self, llm: LLM, store: MemoryStore, embedder: LocalEmbedder,
                 checkpoint_path: str | Path, *, fewshot: bool,
                 format_name: str = "F1", arm: str = "luna-target",
                 prompt_revision: str = "base", emission: str = "pointer",
                 local_thinking: bool = False):
        if emission not in {"pointer", "simple", "lines", "replaces"}:
            raise ValueError(f"unknown emission: {emission}")
        if emission != "pointer" and fewshot:
            raise ValueError("simple/lines emissions run zero-shot; the "
                             "few-shot examples use the pointer format")
        self.emission = emission
        self.local_thinking = local_thinking
        self.llm = llm
        self.store = store
        self.embedder = embedder
        self.checkpoint = Checkpoint(checkpoint_path)
        self.fewshot = fewshot
        self.format_name = format_name
        self.arm = arm
        self.prompt_revision = prompt_revision
        self.local_qwen = llm.model.startswith("qwen3")
        self.max_tokens = (
            2048 if (self.local_qwen and local_thinking)
            else LOCAL_MAX_TOKENS if self.local_qwen else 1400
        )
        self.max_records = LOCAL_MAX_RECORDS if self.local_qwen else None
        self._usage_replayed: set[str] = set()

    def extract(self, session: Session) -> ExtractionResult:
        transcript = "\n".join(
            f"{turn.role}: {turn.content}" for turn in session.turns
        )
        session_fingerprint = stable_hash({
            "schema": 3 if self.local_qwen else 2,
            "scope": str(self.store.path.resolve()),
            "arm": self.arm,
            "model": self.llm.model,
            "format": self.format_name,
            "prompt_revision": self.prompt_revision,
            "fewshot": self.fewshot,
            "max_tokens": self.max_tokens,
            "max_records": self.max_records,
            "emission": self.emission,
            "thinking": self.local_thinking,
            "session_id": session.session_id,
            "session_date": session.date,
            "transcript": transcript,
        })
        completed = next(
            (row for row in self.checkpoint.rows()
             if row.get("session_fingerprint") == session_fingerprint),
            None,
        )
        if completed is not None:
            self._replay_checkpoint_usage(completed)
            records = [self._record_from_dict(row)
                       for row in completed["records"]]
            for record in records:
                self.store.add(record)
            return ExtractionResult(
                input_hash=completed["input_hash"],
                records=records,
                rejected=completed.get("rejected", []),
                schema_valid=completed.get("schema_valid", False),
                repaired=completed.get("repaired", False),
                input_text=completed.get("input", ""),
            )
        candidate_limit = 6 if self.local_qwen else 10
        candidates = self.store.candidates(transcript, k=candidate_limit)
        self._shown_candidates = candidates
        if self.emission != "pointer":
            candidate_text = "\n".join(
                f"{index + 1}. [{record.type}] {record.content} "
                f"({record.created_at})"
                for index, record in enumerate(candidates)
            ) or "(none yet)"
        elif self.local_qwen:
            type_codes = {"narrative": "n", "atomic": "a", "memory": "m"}
            candidate_text = json.dumps([{
                "i": record.id,
                "t": type_codes[record.type],
                "c": record.content,
                "d": record.created_at,
            } for record in candidates], ensure_ascii=False)
        else:
            candidate_text = json.dumps(
                [record.as_dict() for record in candidates],
                ensure_ascii=False,
            )
        user_input = (
            "Candidate memories:\n"
            f"{candidate_text}\n\n"
            f"Session id: {session.session_id}\n"
            f"Session date: {session.date or 'unknown'}\n"
            f"<session>\n{transcript}\n</session>"
        )
        system_prompt = production_prompt(self.format_name, self.prompt_revision)
        if self.emission == "simple":
            system_prompt += SIMPLE_EMISSION_INSTRUCTION
        elif self.emission == "replaces":
            system_prompt += REPLACES_EMISSION_INSTRUCTION
        elif self.emission == "lines":
            system_prompt += LINES_EMISSION_INSTRUCTION
        if self.local_qwen:
            prefix = "" if self.local_thinking else "/no_think\n"
            suffix = (LOCAL_OUTPUT_INSTRUCTION
                      if self.emission == "pointer" else "")
            system_prompt = prefix + system_prompt + suffix
        input_hash = stable_hash({
            "schema": 3 if self.local_qwen else 2,
            "arm": self.arm,
            "model": self.llm.model,
            "format": self.format_name,
            "prompt_revision": self.prompt_revision,
            "fewshot": self.fewshot,
            "max_tokens": self.max_tokens,
            "max_records": self.max_records,
            "emission": self.emission,
            "thinking": self.local_thinking,
            "system": system_prompt,
            "input": user_input,
        })
        cached = self.checkpoint.get(input_hash)
        if cached is not None:
            self._replay_checkpoint_usage(cached)
            records = [self._record_from_dict(row) for row in cached["records"]]
            for record in records:
                self.store.add(record)
            return ExtractionResult(
                input_hash=input_hash,
                records=records,
                rejected=cached.get("rejected", []),
                schema_valid=cached.get("schema_valid", False),
                repaired=cached.get("repaired", False),
                input_text=user_input,
            )

        messages = [{"role": "system", "content": system_prompt}]
        if self.fewshot:
            messages.extend(fewshot_messages(
                self.format_name, compact=self.local_qwen
            ))
        messages.append({"role": "user", "content": user_input})
        usage_before = self.llm.usage.snapshot()
        raw = self.llm.chat(
            messages, max_tokens=self.max_tokens,
            response_format=self._response_format(),
        )
        parsed, parse_error = self._parse_emission(raw)
        valid, rejected = self._validate(
            parsed, transcript, {record.id for record in candidates}, session
        )
        repaired = False
        # Structured output already enforces the JSON shape. Record-level
        # validation failures are deterministic and are dropped locally; asking
        # the same model to regenerate every valid record doubles latency and
        # often replaces good records with a worse answer. Reserve the single
        # model repair for a malformed top-level response only.
        if parse_error:
            repaired = True
            repair_input = (
                "Repair the extraction below. Return the complete corrected JSON.\n"
                f"Validation issues: {json.dumps(rejected or [parse_error], ensure_ascii=False)}\n"
                f"Original extractor input:\n{user_input}\n\n"
                f"Invalid output:\n{raw}"
            )
            repair_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": repair_input},
            ]
            repaired_raw = self.llm.chat(
                repair_messages, max_tokens=self.max_tokens,
                response_format=self._response_format(),
            )
            repaired_parsed, repaired_error = self._parse_emission(repaired_raw)
            repaired_valid, repaired_rejected = self._validate(
                repaired_parsed, transcript,
                {record.id for record in candidates}, session,
            )
            if repaired_error:
                repaired_rejected.append(repaired_error)
            # The repair is authoritative. Invalid repaired records are dropped.
            valid = repaired_valid
            rejected = repaired_rejected
            raw = repaired_raw

        contents = [self._canonical_content(item["content"]) for item in valid]
        vectors = self.embedder.encode(contents)
        records = []
        for item, content, vector in zip(valid, contents, vectors):
            record = Record(
                id=MemoryStore.new_id(),
                session_id=session.session_id,
                type=item["type"],
                content=content,
                entities=item["entities"],
                created_at=session.date or "unknown",
                supersedes_id=item["supersedes_id"],
                embedding=vector,
            )
            records.append(record)
        row = {
            "input_hash": input_hash,
            "session_fingerprint": session_fingerprint,
            "scope_id": self.store.path.stem,
            "arm": self.arm,
            "emission": self.emission,
            "thinking": self.local_thinking,
            "session_id": session.session_id,
            "format": self.format_name,
            "prompt_revision": self.prompt_revision,
            "inference": {
                "revision": (
                    LOCAL_INFERENCE_REVISION if self.local_qwen else "canonical"
                ),
                "max_tokens": self.max_tokens,
                "max_records": self.max_records,
                "repair_policy": "malformed-top-level-only",
            },
            "schema_valid": not parse_error,
            "repaired": repaired,
            "records": [record.as_dict(include_embedding=True) for record in records],
            "rejected": rejected,
            "input": user_input,
            "accepted_output": {"records": [
                {"type": record.type,
                 "content": json.loads(record.content)
                    if self.format_name == "F2" and record.type == "atomic"
                    else record.content,
                 "entities": record.entities,
                 "supersedes_id": record.supersedes_id}
                for record in records
            ]},
            "raw_output": raw,
            "usage": usage_delta(usage_before, self.llm.usage.snapshot()),
        }
        self.checkpoint.append(row)
        for record in records:
            self.store.add(record)
        return ExtractionResult(
            input_hash=input_hash, records=records, rejected=rejected,
            schema_valid=not parse_error, repaired=repaired, input_text=user_input,
        )

    def _response_format(self) -> dict | None:
        if self.emission == "lines":
            return None
        if self.emission == "simple":
            return simple_extraction_schema(max_records=self.max_records)
        if self.emission == "replaces":
            return simple_extraction_schema(max_records=self.max_records,
                                            link_field="replaces")
        return extraction_schema(
            self.format_name, max_records=self.max_records,
            compact=self.local_qwen,
        )

    def _resolve_updates(self, items: list[dict]) -> list[dict]:
        """Map 'updates: <1-based candidate number>' from the simple/lines
        emissions onto supersedes_id, and derive entities from content."""
        resolved = []
        for item in items:
            updates = item.pop("updates", None)
            supersedes_id = None
            if updates not in (None, ""):
                try:
                    index = int(updates) - 1
                except (TypeError, ValueError):
                    index = -1
                if 0 <= index < len(self._shown_candidates):
                    supersedes_id = self._shown_candidates[index].id
                else:
                    item["_updates_out_of_range"] = True
            content = item.get("content") or ""
            resolved.append({
                "type": item.get("type"),
                "content": content,
                "entities": derive_entities(str(content)),
                "supersedes_id": supersedes_id,
                "_updates_out_of_range": item.get("_updates_out_of_range",
                                                  False),
            })
        return resolved

    def _resolve_replaces(self, items: list[dict]) -> list[dict]:
        """Map 'replaces: <restated candidate text>' onto supersedes_id by
        matching against the candidates shown in this call. Exact normalized
        match first, then best token overlap (>= 0.6). Ties prefer a
        candidate that is not itself superseded, then the later-listed one.
        An unmatched replaces keeps the fact and drops the link."""
        superseded_ids = {record.supersedes_id
                          for record in self._shown_candidates
                          if record.supersedes_id}

        def tokens(text: str) -> set[str]:
            return {token for token in
                    regex.findall(r"[\p{L}\p{N}']+", text.lower())
                    if len(token) > 1}

        def match(target: str) -> str | None:
            wanted = " ".join(target.lower().split())
            scored = []
            for index, record in enumerate(self._shown_candidates):
                content = " ".join(str(record.content).lower().split())
                if content == wanted:
                    overlap = 1.01
                else:
                    a, b = tokens(target), tokens(str(record.content))
                    union = a | b
                    overlap = len(a & b) / len(union) if union else 0.0
                active = record.id not in superseded_ids
                scored.append((overlap, active, index, record.id))
            if not scored:
                return None
            overlap, active, index, record_id = max(scored)
            return record_id if overlap >= 0.6 else None

        resolved = []
        for item in items:
            target = item.pop("replaces", None)
            supersedes_id = None
            unmatched = False
            if isinstance(target, str) and target.strip():
                supersedes_id = match(target)
                unmatched = supersedes_id is None
            content = item.get("content") or ""
            resolved.append({
                "type": item.get("type"),
                "content": content,
                "entities": derive_entities(str(content)),
                "supersedes_id": supersedes_id,
                "_replaces_unmatched": unmatched,
            })
        return resolved

    def _parse_emission(self, raw: str) -> tuple[dict, dict | None]:
        if self.emission == "pointer":
            return self._parse(raw, compact=self.local_qwen)
        if self.emission in {"simple", "replaces"}:
            parsed, error = self._parse(raw, compact=False)
            items = parsed.get("records", []) if isinstance(parsed, dict) else []
            items = [item for item in items if isinstance(item, dict)]
            if self.emission == "replaces":
                return {"records": self._resolve_replaces(items)}, error
            return {"records": self._resolve_updates(items)}, error
        # lines
        text = raw.strip()
        if text.startswith("<think>") and "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        items = []
        for line in text.splitlines():
            match = LINE_PATTERN.match(line)
            if match is None:
                continue
            kind, updates, content = match.groups()
            items.append({
                "type": "atomic" if kind.upper() == "A" else "narrative",
                "content": content,
                "updates": int(updates) if updates else None,
            })
        if self.max_records is not None:
            items = items[:self.max_records]
        if not items and text and text.upper() != "NONE":
            return {"records": []}, {"reason": "no parseable memory lines"}
        return {"records": self._resolve_updates(items)}, None

    def _replay_checkpoint_usage(self, row: dict) -> None:
        input_hash = row["input_hash"]
        if input_hash in self._usage_replayed:
            return
        replay_usage(self.llm.usage, row.get("usage"))
        self._usage_replayed.add(input_hash)

    def _validate(self, parsed: dict, transcript: str,
                  candidate_ids: set[str], session: Session) -> tuple[list[dict], list[dict]]:
        items = parsed.get("records", []) if isinstance(parsed, dict) else []
        if not isinstance(items, list):
            return [], [{"reason": "records must be a list"}]
        valid, rejected = [], []
        allowed_types = {"memory"} if self.format_name == "F4" else {
            "narrative", "atomic"
        }
        for index, item in enumerate(items):
            issues = []
            if not isinstance(item, dict):
                rejected.append({"index": index, "reason": "record is not an object"})
                continue
            if item.get("type") not in allowed_types:
                issues.append("invalid type")
            content = item.get("content")
            if self.format_name == "F2" and item.get("type") == "atomic":
                required = {"subject", "attribute", "value", "unit", "as_of_date"}
                if not isinstance(content, dict) or set(content) != required:
                    issues.append("structured atomic content has wrong fields")
                elif not all(isinstance(content[key], str) and content[key].strip()
                             for key in ("subject", "attribute", "value")):
                    issues.append("structured atomic required values must be strings")
            elif not isinstance(content, str) or not content.strip():
                issues.append("content must be a non-empty string")
            entities = item.get("entities")
            if (not isinstance(entities, list)
                    or any(not isinstance(entity, str) or not entity.strip()
                           for entity in entities)):
                issues.append("entities must be strings")
            supersedes_id = item.get("supersedes_id")
            if supersedes_id is not None and supersedes_id not in candidate_ids:
                issues.append("supersedes_id was not shown in candidates")
            content_text = self._canonical_content(content)
            if item.get("type") in {"atomic", "memory"}:
                missing = [token for token in NUMBER_OR_DATE_TOKEN.findall(content_text)
                           if token not in transcript]
                if missing:
                    issues.append(f"atomic exact-string check failed: {missing}")
            if issues:
                rejected.append({"index": index, "reasons": issues, "record": item})
                continue
            valid.append({
                "type": item["type"],
                "content": content,
                "entities": [entity.strip() for entity in entities],
                "supersedes_id": supersedes_id,
            })
        return valid, rejected

    @staticmethod
    def _parse(raw: str, *, compact: bool = False) -> tuple[dict, dict | None]:
        try:
            value = json.loads(repair_json(raw))
        except Exception as exc:
            return {}, {"reason": f"unparseable JSON: {type(exc).__name__}"}
        if compact:
            if not isinstance(value, dict) or not isinstance(value.get("r"), list):
                return {}, {"reason": "compact object must contain r array"}
            type_names = {"n": "narrative", "a": "atomic", "m": "memory"}
            records = []
            required = {"t", "c", "e", "s"}
            for item in value["r"]:
                if not isinstance(item, dict) or not required <= item.keys():
                    # json_repair can recover every complete item before a
                    # token-truncated tail. Preserve those siblings and send
                    # only the incomplete tail through normal validation.
                    records.append({
                        "type": None, "content": None, "entities": None,
                        "supersedes_id": "__incomplete_compact_record__",
                    })
                    continue
                records.append({
                    "type": type_names.get(item["t"]),
                    "content": item["c"],
                    "entities": item["e"],
                    "supersedes_id": item["s"],
                })
            value = {"records": records}
        if not isinstance(value, dict) or "records" not in value:
            return {}, {"reason": "top-level object must contain records"}
        return value, None

    @staticmethod
    def _canonical_content(content) -> str:
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"))
        return str(content).strip()

    @staticmethod
    def _record_from_dict(row: dict) -> Record:
        vector = row.get("embedding")
        import numpy as np
        return Record(
            id=row["id"], session_id=row["session_id"], type=row["type"],
            content=row["content"], entities=row["entities"],
            created_at=row["created_at"], supersedes_id=row.get("supersedes_id"),
            embedding=np.asarray(vector, dtype=np.float32) if vector is not None else None,
        )

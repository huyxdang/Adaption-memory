"""Fixed Luna answer path over retrieved memory records."""

from __future__ import annotations

from pathlib import Path

from adaption_memory.evals.common import usage_delta
from adaption_memory.llm import LLM
from adaption_memory.memory.checkpoint import Checkpoint, replay_usage, stable_hash
from adaption_memory.memory.retrieve import Retrieved


ANSWER_SYSTEM_PROMPT = """Answer the question using only the supplied memory.
Prefer current records over records marked superseded. Use historical records
only for questions about an earlier state or change. Preserve exact dates,
figures, IDs, and constraints. If the memory does not contain enough evidence,
say "No information available." Be concise and do not mention the memory
system.
"""

TYPE_AWARE_ANSWER_PROMPT = """Answer the question using only the supplied
memory records.

Use the record types deliberately. Atomic records carry exact facts: when the
answer is a specific value — a time, date, name, quantity, price, URL, or
identifier — quote it verbatim from an atomic record; never round, rephrase,
or derive it. Narrative records carry decisions, reasons, preferences, and
context: use them for why/how questions and to connect events across
sessions.

Time and change: prefer records not marked superseded; a superseded record is
an earlier state, correct only for questions about the past or about what
changed. When several records describe the same fact, the unsuperseded, most
recent one is the present truth. Use record dates to order events.

If the memory does not contain enough evidence, say "No information
available." Be concise and do not mention the memory system.
"""

CHRONO_ANSWER_PROMPT = ANSWER_SYSTEM_PROMPT + """
The records are listed in chronological order, oldest first; use that order
and the record dates to reason about sequences, changes, and durations.
"""

ANSWER_PROMPTS = {"base": ANSWER_SYSTEM_PROMPT,
                  "type-aware": TYPE_AWARE_ANSWER_PROMPT,
                  "chrono": CHRONO_ANSWER_PROMPT}


class MemoryAnswerer:
    def __init__(self, llm: LLM, checkpoint_path: str | Path,
                 arm: str, format_name: str = "F1",
                 answer_revision: str = "base"):
        if llm.model != "gpt-5.6-luna" or llm.reasoning_effort != "none":
            raise ValueError("answerer must be gpt-5.6-luna with effort none")
        self.llm = llm
        self.checkpoint = Checkpoint(checkpoint_path)
        self.arm = arm
        self.format_name = format_name
        self.system_prompt = ANSWER_PROMPTS[answer_revision]
        self.chronological = answer_revision == "chrono"
        self._usage_replayed: set[str] = set()

    def answer(self, question: str, records: list[Retrieved],
               question_date: str | None = None,
               instruction: str | None = None) -> tuple[str, str]:
        if self.chronological:
            records = sorted(records,
                             key=lambda item: item.record.created_at or "")
        rendered = self.render(records)
        user_parts = ["<memory>", rendered, "</memory>"]
        if question_date:
            user_parts.append(f"Question date: {question_date}")
        user_parts.append(f"Question: {question}")
        if instruction:
            user_parts.append(f"Output instruction: {instruction}")
        user_input = "\n".join(user_parts)
        input_hash = stable_hash({
            "schema": 1, "arm": self.arm, "format": self.format_name,
            "model": self.llm.model, "effort": self.llm.reasoning_effort,
            "system": self.system_prompt, "input": user_input,
        })
        cached = self.checkpoint.get(input_hash)
        if cached is not None:
            if input_hash not in self._usage_replayed:
                replay_usage(self.llm.usage, cached.get("usage"))
                self._usage_replayed.add(input_hash)
            return cached["answer"], input_hash
        usage_before = self.llm.usage.snapshot()
        answer = self.llm.chat([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ], max_tokens=1200)
        self.checkpoint.append({
            "input_hash": input_hash,
            "arm": self.arm,
            "format": self.format_name,
            "question": question,
            "record_ids": [item.record.id for item in records],
            "answer": answer,
            "usage": usage_delta(usage_before, self.llm.usage.snapshot()),
        })
        return answer, input_hash

    @staticmethod
    def render(records: list[Retrieved]) -> str:
        if not records:
            return "(no records retrieved)"
        lines = []
        for item in records:
            record = item.record
            relation = (f"supersedes={record.supersedes_id}"
                        if record.supersedes_id else "new")
            lines.append(
                f"- [{record.id}; type={record.type}; at={record.created_at}; "
                f"{relation}] {record.content}"
            )
        return "\n".join(lines)

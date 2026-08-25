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


class MemoryAnswerer:
    def __init__(self, llm: LLM, checkpoint_path: str | Path,
                 arm: str, format_name: str = "F1"):
        if llm.model != "gpt-5.6-luna" or llm.reasoning_effort != "none":
            raise ValueError("answerer must be gpt-5.6-luna with effort none")
        self.llm = llm
        self.checkpoint = Checkpoint(checkpoint_path)
        self.arm = arm
        self.format_name = format_name
        self._usage_replayed: set[str] = set()

    def answer(self, question: str, records: list[Retrieved],
               question_date: str | None = None,
               instruction: str | None = None) -> tuple[str, str]:
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
            "system": ANSWER_SYSTEM_PROMPT, "input": user_input,
        })
        cached = self.checkpoint.get(input_hash)
        if cached is not None:
            if input_hash not in self._usage_replayed:
                replay_usage(self.llm.usage, cached.get("usage"))
                self._usage_replayed.add(input_hash)
            return cached["answer"], input_hash
        usage_before = self.llm.usage.snapshot()
        answer = self.llm.chat([
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
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

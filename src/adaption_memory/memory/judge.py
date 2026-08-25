"""Mem0-style semantic judge shared by all three benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

from json_repair import repair_json

from adaption_memory.evals.common import usage_delta
from adaption_memory.llm import LLM
from adaption_memory.memory.checkpoint import Checkpoint, replay_usage, stable_hash


JUDGE_PROMPT = """You are an impartial memory QA judge. Decide whether the
model response correctly answers the question according to the reference.
Accept equivalent wording and harmless extra detail. Reject contradictions,
answers that omit an essential reference fact, and fabricated specifics. If
the reference says the information is absent, accept only a response that
recognizes it is unavailable. Return JSON only with a boolean label and a
short reason.
"""

JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "memory_qa_judge",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "label": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["label", "reason"],
        },
    },
}


class MemoryJudge:
    def __init__(self, llm: LLM, checkpoint_path: str | Path):
        if llm.model != "gpt-5.6-luna" or llm.reasoning_effort != "none":
            raise ValueError("judge must be gpt-5.6-luna with effort none")
        self.llm = llm
        self.checkpoint = Checkpoint(checkpoint_path)
        self._usage_replayed: set[str] = set()

    def judge(self, *, question_id: str, benchmark: str, question: str,
              reference: str, response: str) -> dict:
        input_text = (
            f"Benchmark: {benchmark}\nQuestion: {question}\n"
            f"Reference answer or rubric: {reference}\n"
            f"Model response: {response}"
        )
        input_hash = stable_hash({
            "schema": 1, "prompt": JUDGE_PROMPT, "input": input_text,
            "model": self.llm.model, "effort": self.llm.reasoning_effort,
        })
        cached = self.checkpoint.get(input_hash)
        if cached is not None:
            if input_hash not in self._usage_replayed:
                replay_usage(self.llm.usage, cached.get("usage"))
                self._usage_replayed.add(input_hash)
            return cached
        usage_before = self.llm.usage.snapshot()
        raw = self.llm.chat([
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": input_text},
        ], max_tokens=250, response_format=JUDGE_SCHEMA)
        try:
            parsed = json.loads(repair_json(raw))
        except Exception:
            parsed = {"label": False, "reason": "unparseable judge response"}
        label = parsed.get("label") is True
        row = {
            "input_hash": input_hash,
            "question_id": question_id,
            "benchmark": benchmark,
            "label": label,
            "reason": str(parsed.get("reason", "")),
            "judge_response": raw,
            "usage": usage_delta(usage_before, self.llm.usage.snapshot()),
        }
        self.checkpoint.append(row)
        return row
